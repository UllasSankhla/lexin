"""Scheduling agent — presents slots, takes choice, confirms and books."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from pydantic import BaseModel

from app.agents.base import AgentBase, AgentStatus, SubagentResponse
from app.agents.llm_utils import llm_structured_call, llm_text_call, ConversationHistory
from app.agents.agent_schemas import SlotConfirmSignal, EventTypeMatch
from app.services.calendar_service import list_available_slots, book_time_slot


class _NeedsAnswerSignal(BaseModel):
    needs_answer: bool


class _SlotAction(BaseModel):
    """Single-call intent classification for slot choice and date preference."""
    action: str  # "pick" | "new_date" | "unclear"
    slot_index: Optional[int] = None   # 1-based, present when action=="pick"
    start_time: Optional[str] = None   # ISO UTC, present when action=="new_date"
    end_time: Optional[str] = None     # ISO UTC, present when action=="new_date"


logger = logging.getLogger(__name__)

_SCHEDULING_QUESTION_GATE_SYSTEM = (
    "A caller is in the middle of booking an appointment with a law firm.\n"
    "Determine whether the utterance requires a factual answer (something the "
    "scheduling agent cannot provide), or is scheduling-domain content the agent "
    "should handle (choosing a slot, confirming, requesting a different time).\n\n"
    "needs_answer=true  — utterance is a question about the firm, fees, legal process,\n"
    "                     documents, or anything outside picking/confirming a time slot.\n"
    "needs_answer=false — utterance is about choosing or confirming a time slot,\n"
    "                     requesting a different day/time, or confirming a booking.\n\n"
    "true  → 'What are your fees?' | 'Where are you located?' | "
    "'What should I bring?' | 'How long does a consultation last?'\n"
    "false → 'The second one.' | 'Can we do Tuesday instead?' | "
    "'Yes, confirm that.' | 'Do you have anything earlier?' | 'That works.'\n\n"
    "Reply ONLY valid JSON: {\"needs_answer\": true} or {\"needs_answer\": false}."
)

_SLOT_ACTION_SYSTEM = (
    "Given a list of available appointment slots and a caller's utterance, "
    "determine what the caller wants.\n\n"
    "Return one of:\n"
    "  {\"action\": \"pick\", \"slot_index\": <1-based int>}\n"
    "    — caller is selecting a specific slot from the current list\n"
    "    — Examples: \"The second one\", \"Monday at 10\", \"I'll take option 1\", "
    "\"That works\", \"The first one\", \"Option 2 please\"\n\n"
    "  {\"action\": \"new_date\", \"start_time\": \"<ISO UTC>\", \"end_time\": \"<ISO UTC>\"}\n"
    "    — caller wants to see slots on a different day or time, OR is asking whether "
    "a specific day/time is available\n"
    "    — Full-day range for a specific day (e.g. Thursday = 00:00–23:59 that Thursday)\n"
    "    — 7-day window for vague week preferences (e.g. 'next week', 'later this month')\n"
    "    — 14-day window for time-of-day preferences without a specific day "
    "(e.g. 'afternoon', 'morning', 'earlier', 'later in the day')\n"
    "    — Examples: \"Do you have Thursday?\", \"Any Friday morning?\", "
    "\"What about next week?\", \"Something earlier?\", "
    "\"Do you have anything in the afternoon?\", \"How about a different day?\"\n\n"
    "  {\"action\": \"unclear\"}\n"
    "    — cannot determine what the caller wants\n\n"
    "Today's date and current slots are in the message. "
    "Compute relative dates from today. "
    "Reply ONLY valid JSON."
)

_SPEAK_SLOTS_SYSTEM = (
    "Present available appointment slots to a caller over the phone. "
    "Friendly and concise. Plain text only — no markdown, no asterisks, no numbers, no special characters. "
    "List the slots naturally as a comma-separated list. Under three sentences. "
    "Ask which one they prefer."
)

_CONFIRM_INTENT_SYSTEM = """\
The caller was just asked to confirm a specific appointment slot (e.g. "I'll book you for Monday at 10 AM. Shall I confirm?").
Classify their response into exactly one of:

confirm  — caller accepts the slot as-is.
           Examples: "Yes", "That works", "Go ahead", "Perfect", "Sounds good", "Sure",
                     "Let's do it", "That's fine", "Confirmed"

reject   — caller declines and wants to see other options (no specific new time given).
           Examples: "No", "That doesn't work", "Change it", "Pick a different one",
                     "Actually no", "I'd rather not", "Let me think", "Never mind"

new_slot — caller rejects the slot AND specifies a new time preference.
           Examples: "Do you have anything later?", "Let's do 1pm", "Can we do Tuesday afternoon?",
                     "The second one", "How about Wednesday?", "After noon if possible"

Respond with ONLY valid JSON: {"intent": "confirm"} or {"intent": "reject"} or {"intent": "new_slot"}
"""


def _detect_confirmation(utterance: str, slot_desc: str = "") -> str:
    """Return 'confirm', 'reject', or 'new_slot' via LLM classification."""
    context = f'Slot offered: "{slot_desc}"\n' if slot_desc else ""
    user_msg = f'{context}Caller said: "{utterance}"'
    try:
        result = llm_structured_call(_CONFIRM_INTENT_SYSTEM, user_msg, SlotConfirmSignal, max_tokens=64, tag="scheduling_confirm_intent")
        logger.debug("SchedulingAgent: confirm_intent=%r for %r", result.intent, utterance[:60])
        return result.intent
    except Exception as exc:
        logger.warning("SchedulingAgent: _detect_confirmation LLM failed: %s — defaulting to new_slot", exc)
    return "new_slot"


class SchedulingAgent(AgentBase):
    is_primary_interactive = True

    """
    Internal state keys:
      stage: "presenting" | "awaiting_choice" | "awaiting_confirm" | "done"
      available_slots: list of slot dicts (serializable)
      chosen_slot_id: int | None  (index into available_slots)
      retry_count: int
      matched_event_type_uri: str | None
    """

    MAX_RETRIES = 3

    def process(
        self,
        utterance: str,
        internal_state: dict,
        config: dict,
        history: list[dict],
    ) -> SubagentResponse:
        stage = internal_state.get("stage", "presenting")

        # ── Resume: re-surface pending state after an interrupt (utterance="") ──
        if not utterance:
            if stage == "awaiting_choice":
                slots_data = internal_state.get("available_slots", [])
                if slots_data:
                    speak = self._speak_slots(slots_data)
                    return SubagentResponse(
                        status=AgentStatus.IN_PROGRESS,
                        speak=speak,
                        internal_state=internal_state,
                    )
            elif stage == "awaiting_confirm":
                slots_data = internal_state.get("available_slots", [])
                chosen_idx = internal_state.get("chosen_slot_id")
                if slots_data and chosen_idx is not None and chosen_idx < len(slots_data):
                    chosen = slots_data[chosen_idx]
                    speak = (
                        f"Just to confirm — I'll book you for {chosen['description']}. "
                        "Shall I go ahead?"
                    )
                    return SubagentResponse(
                        status=AgentStatus.WAITING_CONFIRM,
                        speak=speak,
                        pending_confirmation={"slot": chosen["description"]},
                        internal_state=internal_state,
                    )
            # presenting or no slots yet — fall through to _present_slots
            return self._present_slots(utterance, internal_state, config)

        # ── Domain gate — return UNHANDLED for questions needing a factual answer ─
        # Only gate during active choice/confirm stages — not during presenting,
        # which runs once automatically before any caller input.
        if stage in ("awaiting_choice", "awaiting_confirm"):
            if self._needs_answer(utterance):
                logger.info(
                    "SchedulingAgent: UNHANDLED — utterance needs a factual answer "
                    "| stage=%s | %r", stage, utterance[:80],
                )
                return SubagentResponse(
                    status=AgentStatus.UNHANDLED,
                    speak="",
                    internal_state=internal_state,
                    confidence=0.0,
                )

        if stage == "presenting":
            return self._present_slots(utterance, internal_state, config)

        if stage == "awaiting_choice":
            return self._handle_choice(utterance, internal_state, config)

        if stage == "awaiting_confirm":
            return self._handle_confirmation(utterance, internal_state, config)

        return SubagentResponse(
            status=AgentStatus.COMPLETED,
            speak="Your appointment has been confirmed. Have a great day!",
            internal_state=internal_state,
        )

    def _speak_slots(self, slots_data: list) -> str:
        """Build a spoken slot list from a list of slot dicts or TimeSlot objects."""
        slot_list = ", ".join(
            s.description if hasattr(s, "description") else s["description"]
            for s in slots_data
        )
        return llm_text_call(_SPEAK_SLOTS_SYSTEM, f"Slots: {slot_list}", tag="scheduling_slots_speak")

    def _present_slots(self, utterance: str, internal_state: dict, config: dict) -> SubagentResponse:
        # Use pre-fetched slots if CalendarPrefetchTool already ran concurrently
        # with the intake_qualification LLM call — avoids a redundant API round-trip.
        prefetched = config.get("_tool_results", {}).get("prefetched_slots")
        if prefetched is not None:
            slots = prefetched
            internal_state["matched_event_type_uri"] = None
        else:
            # Pull collected data from graph state via config passthrough
            # (the handler injects collected params into config["_collected"])
            collected = config.get("_collected", {})

            # Derive purpose
            purpose = (
                collected.get("purpose") or collected.get("reason") or
                collected.get("service") or collected.get("appointment_type") or
                config.get("assistant", {}).get("persona_description", "appointment")
            )

            # Match event type via LLM if event types configured
            event_types = config.get("calendly_event_types") or []
            matched_uri = None
            if event_types:
                try:
                    result = llm_structured_call(
                        "Match a booking purpose to an event type. Return JSON: {\"index\": <0-based int>} or {\"index\": null}.",
                        f"Purpose: \"{purpose}\"\nEvent types:\n" +
                        "\n".join(f"{i}. {et['name']}" + (f" — {et['description']}" if et.get('description') else "") for i, et in enumerate(event_types)),
                        EventTypeMatch,
                        tag="scheduling_event_match",
                    )
                    if result.index is not None and 0 <= int(result.index) < len(event_types):
                        matched_uri = event_types[int(result.index)]["event_type_uri"]
                except Exception as exc:
                    logger.warning("SchedulingAgent event type match failed: %s", exc)
                if not matched_uri:
                    matched_uri = event_types[0]["event_type_uri"]

            internal_state["matched_event_type_uri"] = matched_uri

            now = datetime.now(timezone.utc)
            search_start = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            search_end = search_start + timedelta(days=7)

            slots = list_available_slots(purpose, config, matched_uri, search_start=search_start, search_end=search_end)

        # Store slots as serializable dicts — use isoformat for datetime fields
        internal_state["available_slots"] = [
            {
                "slot_id": s.slot_id,
                "description": s.description,
                "start_time": s.start.isoformat() if s.start else "",
                "end_time": s.end.isoformat() if s.end else "",
                "event_type_uri": s.event_type_uri,
            }
            for s in slots
        ]
        internal_state["stage"] = "awaiting_choice"
        internal_state["retry_count"] = 0

        if not slots:
            speak = (
                "I don't have any openings in the next seven days. "
                "When would work best for you? For example, you could say 'next Tuesday morning'."
            )
            return SubagentResponse(
                status=AgentStatus.IN_PROGRESS,
                speak=speak,
                internal_state=internal_state,
            )

        speak = self._speak_slots(slots)
        return SubagentResponse(
            status=AgentStatus.IN_PROGRESS,
            speak=speak,
            internal_state=internal_state,
        )

    def _handle_choice(self, utterance: str, internal_state: dict, config: dict) -> SubagentResponse:
        slots_data = internal_state.get("available_slots", [])
        llm_history = ConversationHistory.from_list(internal_state.get("llm_history"))

        # Single LLM call: classify as slot pick, new date range, or unclear.
        # Combining slot matching and date extraction in one call avoids the
        # two-step fragility where question-form utterances ("Do you have Thursday?")
        # or time-of-day preferences ("afternoon") fail the standalone DateRangePreference
        # extraction but carry a clear user intent.
        current_date = datetime.now(timezone.utc).strftime("%A, %Y-%m-%d")
        numbered = (
            "\n".join(f"{i+1}. {s['description']}" for i, s in enumerate(slots_data))
            if slots_data else "(no slots currently loaded)"
        )
        user_msg = (
            f"Today is {current_date}.\n"
            f"Available slots:\n{numbered}\n"
            f"Caller said: \"{utterance}\""
        )
        try:
            action = llm_structured_call(
                _SLOT_ACTION_SYSTEM,
                user_msg,
                _SlotAction,
                max_tokens=256,
                history=llm_history,
                tag="scheduling_slot_action",
            )
            logger.debug(
                "SchedulingAgent: slot_action=%s slot_index=%s for %r",
                action.action, action.slot_index, utterance[:60],
            )

            # ── Pick: caller selected a slot from the current list ────────────
            if action.action == "pick" and action.slot_index is not None and slots_data:
                idx = int(action.slot_index) - 1
                # ── Steering: slot choice bounds guard ────────────────────────
                bounds_issue = self._slot_choice_bounds_guard(idx, slots_data)
                if bounds_issue:
                    logger.warning(
                        "Steering[slot_choice_bounds]: %s | slot_index=%s len=%d",
                        bounds_issue, action.slot_index, len(slots_data),
                    )
                    # Retry with explicit correction context
                    correction_msg = (
                        f"STEERING CORRECTION: {bounds_issue} "
                        f"Valid slot numbers are 1 to {len(slots_data)}. "
                        f"Re-read the list and return the correct slot number.\n\n"
                        + user_msg
                    )
                    try:
                        action = llm_structured_call(
                            _SLOT_ACTION_SYSTEM,
                            correction_msg,
                            _SlotAction,
                            max_tokens=256,
                            tag="scheduling_slot_action_steered",
                        )
                        idx = int(action.slot_index) - 1 if action.slot_index is not None else -1
                    except Exception as exc:
                        logger.warning("Steering slot action retry failed: %s", exc)
                        action = _SlotAction(action="unclear")
                        idx = -1

                if action.action == "pick" and action.slot_index is not None and 0 <= idx < len(slots_data):
                    chosen = slots_data[idx]
                    internal_state["chosen_slot_id"] = idx
                    internal_state["stage"] = "awaiting_confirm"
                    internal_state["retry_count"] = 0
                    speak = llm_text_call(
                        "Generate a single voice sentence confirming a chosen appointment slot "
                        "and asking for final confirmation.",
                        f"Slot: {chosen['description']}\nPattern: 'I'll book you for [slot]. Shall I confirm?'",
                        history=llm_history,
                        tag="scheduling_slot_confirm_speak",
                    )
                    llm_history.add("user", user_msg)
                    llm_history.add("assistant", speak)
                    internal_state["llm_history"] = llm_history.to_list()
                    return SubagentResponse(
                        status=AgentStatus.WAITING_CONFIRM,
                        speak=speak,
                        pending_confirmation={"slot": chosen["description"]},
                        internal_state=internal_state,
                    )

            # ── New date: caller wants a different day or time ────────────────
            if action.action == "new_date" and action.start_time and action.end_time:
                try:
                    s = datetime.fromisoformat(action.start_time.replace("Z", "+00:00"))
                    e = datetime.fromisoformat(action.end_time.replace("Z", "+00:00"))
                except ValueError as exc:
                    logger.warning("SchedulingAgent: bad date range from LLM: %s — %s", action, exc)
                else:
                    collected = config.get("_collected", {})
                    purpose = collected.get("purpose") or collected.get("reason") or "appointment"
                    matched_uri = internal_state.get("matched_event_type_uri")
                    new_slots = list_available_slots(purpose, config, matched_uri, search_start=s, search_end=e)
                    internal_state["available_slots"] = [
                        {
                            "slot_id": sl.slot_id,
                            "description": sl.description,
                            "start_time": sl.start.isoformat() if sl.start else "",
                            "end_time": sl.end.isoformat() if sl.end else "",
                            "event_type_uri": sl.event_type_uri,
                        }
                        for sl in new_slots
                    ]
                    internal_state["retry_count"] = 0
                    if not new_slots:
                        speak = (
                            "I'm sorry, I don't see any openings in that time range. "
                            "Would you like to try a different day?"
                        )
                        return SubagentResponse(
                            status=AgentStatus.IN_PROGRESS,
                            speak=speak,
                            internal_state=internal_state,
                        )
                    speak = self._speak_slots(new_slots)
                    llm_history.add("user", user_msg)
                    llm_history.add("assistant", speak)
                    internal_state["llm_history"] = llm_history.to_list()
                    return SubagentResponse(
                        status=AgentStatus.IN_PROGRESS,
                        speak=speak,
                        internal_state=internal_state,
                    )

        except Exception as exc:
            logger.warning("SchedulingAgent slot action LLM failed: %s", exc)

        # ── Retry / fallback ──────────────────────────────────────────────────
        retry = internal_state.get("retry_count", 0) + 1
        internal_state["retry_count"] = retry
        if retry >= self.MAX_RETRIES and slots_data:
            chosen = slots_data[0]
            internal_state["chosen_slot_id"] = 0
            internal_state["stage"] = "awaiting_confirm"
            speak = llm_text_call(
                "Generate a voice sentence confirming an appointment slot and asking for final confirmation.",
                f"Slot: {chosen['description']}",
                tag="scheduling_max_retry_speak",
            )
            return SubagentResponse(
                status=AgentStatus.WAITING_CONFIRM,
                speak=speak,
                pending_confirmation={"slot": chosen["description"]},
                internal_state=internal_state,
            )

        if slots_data:
            speak = self._speak_slots(slots_data)
        else:
            speak = "I don't have any slots loaded. When would work best for you?"
        return SubagentResponse(
            status=AgentStatus.IN_PROGRESS,
            speak=speak,
            internal_state=internal_state,
        )

    def _handle_confirmation(self, utterance: str, internal_state: dict, config: dict) -> SubagentResponse:
        slots_data = internal_state.get("available_slots", [])
        chosen_idx = internal_state.get("chosen_slot_id", 0)
        chosen_data = slots_data[chosen_idx] if slots_data and chosen_idx < len(slots_data) else None
        slot_desc = chosen_data["description"] if chosen_data else ""

        intent = _detect_confirmation(utterance, slot_desc)
        logger.info("SchedulingAgent: confirmation intent=%r | utterance=%r", intent, utterance[:60])

        if intent == "confirm":
            # ── Steering: pre-booking preflight ──────────────────────────────
            # Runs before book_time_slot — the only irreversible action in the
            # pipeline. Validates slot data integrity and confirms the slot
            # being booked matches what the caller was presented.
            preflight_issue = self._booking_preflight(chosen_data, internal_state)
            if preflight_issue:
                logger.error(
                    "Steering[booking_preflight]: blocked booking | reason=%s | chosen_idx=%s slot=%s",
                    preflight_issue,
                    internal_state.get("chosen_slot_id"),
                    chosen_data,
                )
                speak = (
                    "I'm sorry, something doesn't look right with that slot. "
                    "Let me pull up the available times again."
                )
                internal_state["stage"] = "awaiting_choice"
                internal_state["retry_count"] = 0
                return SubagentResponse(
                    status=AgentStatus.IN_PROGRESS,
                    speak=speak,
                    internal_state=internal_state,
                )
            if not chosen_data:
                return SubagentResponse(
                    status=AgentStatus.COMPLETED,
                    speak="Your appointment has been confirmed. We'll see you then!",
                    internal_state=internal_state,
                )
            from app.services.calendar_service import TimeSlot
            start_dt = datetime.fromisoformat(chosen_data["start_time"]) if chosen_data.get("start_time") else datetime.now(timezone.utc)
            end_dt = datetime.fromisoformat(chosen_data["end_time"]) if chosen_data.get("end_time") else start_dt + timedelta(hours=1)
            slot = TimeSlot(
                slot_id=chosen_data.get("slot_id", ""),
                description=chosen_data["description"],
                start=start_dt,
                end=end_dt,
                event_type_uri=chosen_data.get("event_type_uri", ""),
            )
            collected = config.get("_collected", {})
            try:
                booking = book_time_slot(slot, collected, config)
            except Exception as exc:
                logger.error("SchedulingAgent booking failed: %s", exc)
                booking = {"booking_id": "N/A", "error": str(exc)}

            has_email = bool(collected.get("email_address"))
            email_note = " We've sent the details to your email." if has_email else ""
            speak = f"Your meeting has been scheduled.{email_note} We look forward to speaking with you!"
            internal_state["stage"] = "done"
            return SubagentResponse(
                status=AgentStatus.COMPLETED,
                speak=speak,
                booking=booking,
                internal_state=internal_state,
            )

        if intent == "reject":
            internal_state["stage"] = "awaiting_choice"
            internal_state["retry_count"] = 0
            speak = self._speak_slots(slots_data) if slots_data else "No problem. When would work best for you?"
            return SubagentResponse(
                status=AgentStatus.IN_PROGRESS,
                speak=speak,
                internal_state=internal_state,
            )

        # new_slot — caller wants a different time; let _handle_choice parse the preference
        return self._handle_choice(utterance, internal_state, config)

    # ── Steering handlers ─────────────────────────────────────────────────────

    def _slot_choice_bounds_guard(
        self,
        idx: int,
        slots_data: list,
    ) -> str | None:
        """
        Steering handler: slot choice bounds guard.

        After the LLM returns a slot index, verify it falls within the
        available_slots list before accessing it. Returns None if valid,
        or a correction string if out of bounds.

        Prevents IndexError and silent retry loops when Calendly returns
        fewer slots than the LLM expects (e.g. after a narrow date search).
        """
        if not slots_data:
            return "No slots are available — cannot pick from an empty list."
        if idx < 0 or idx >= len(slots_data):
            return (
                f"Slot index {idx + 1} is out of range — "
                f"only {len(slots_data)} slot(s) available (1 to {len(slots_data)})."
            )
        return None

    def _booking_preflight(
        self,
        chosen_data: dict | None,
        internal_state: dict,
    ) -> str | None:
        """
        Steering handler: pre-booking preflight.

        Runs before book_time_slot() — the only irreversible action in the
        pipeline. Validates that the slot in state is complete and consistent
        with what was presented to the caller for confirmation.

        Returns None if safe to proceed, or a reason string if the booking
        should be blocked and the agent should re-present slots.

        Checks:
          1. A slot is actually selected (chosen_data is not None)
          2. slot_id is present (Calendly needs it for the API call)
          3. start_time is present and parses as a valid ISO datetime
          4. The slot description matches pending_confirmation["slot"] —
             i.e. the slot being booked is the one the caller confirmed,
             not a different slot at the same index position
        """
        if chosen_data is None:
            return "No slot selected in state — chosen_slot_id points to nothing."

        if not chosen_data.get("slot_id"):
            return f"Slot is missing slot_id: {chosen_data}"

        start_time_str = chosen_data.get("start_time", "")
        if not start_time_str:
            return f"Slot is missing start_time: {chosen_data}"
        try:
            datetime.fromisoformat(start_time_str)
        except ValueError:
            return f"Slot start_time is not a valid ISO datetime: {start_time_str!r}"

        # Guard: description must match what was presented to the caller
        pending = internal_state.get("pending_confirmation") or {}
        confirmed_desc = pending.get("slot", "")
        slot_desc = chosen_data.get("description", "")
        if confirmed_desc and slot_desc and confirmed_desc != slot_desc:
            return (
                f"Slot mismatch: caller confirmed '{confirmed_desc}' but "
                f"chosen_slot_id points to '{slot_desc}'. "
                f"These must be the same slot."
            )

        return None  # all clear — safe to book

    def _needs_answer(self, utterance: str) -> bool:
        """Return True if utterance requires a factual answer outside scheduling domain."""
        try:
            result = llm_structured_call(
                _SCHEDULING_QUESTION_GATE_SYSTEM,
                f"Caller said: \"{utterance}\"",
                _NeedsAnswerSignal,
                max_tokens=64,
                tag="scheduling_question_gate",
            )
            logger.debug(
                "SchedulingAgent: question_gate needs_answer=%s for %r",
                result.needs_answer, utterance[:60],
            )
            return result.needs_answer
        except Exception as exc:
            logger.warning("SchedulingAgent: question gate failed: %s — defaulting to False", exc)
            return False
