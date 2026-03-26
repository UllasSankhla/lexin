"""Scheduling agent — presents slots, takes choice, confirms and books."""
from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone

from app.agents.base import AgentBase, AgentStatus, SubagentResponse
from app.agents.llm_utils import llm_json_call, llm_text_call, ConversationHistory
from app.services.calendar_service import list_available_slots, book_time_slot

logger = logging.getLogger(__name__)

_SPEAK_SLOTS_SYSTEM = (
    "Present available appointment slots to a caller over the phone. "
    "Friendly and concise. Plain text only — no markdown, no asterisks, no numbers, no special characters. "
    "List the slots naturally as a comma-separated list. Under three sentences. "
    "Ask which one they prefer."
)

_YES_RE = re.compile(
    r"\b(yes|yeah|yep|yup|correct|right|confirmed|confirm|sure|ok|okay|"
    r"perfect|great|sounds good|go ahead|absolutely|definitely)\b", re.IGNORECASE
)
_NO_RE = re.compile(
    r"\b(no|nope|nah|wrong|incorrect|change|different|actually|wait|"
    r"hold on|not right|not quite)\b", re.IGNORECASE
)


def _detect_confirmation(text: str):
    has_yes = bool(_YES_RE.search(text))
    has_no = bool(_NO_RE.search(text))
    if has_yes and not has_no:
        return True
    if has_no and not has_yes:
        return False
    return None


class SchedulingAgent(AgentBase):
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

        if stage == "presenting" or (stage == "presenting" and not utterance):
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
        return llm_text_call(_SPEAK_SLOTS_SYSTEM, f"Slots: {slot_list}")

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
                    result = llm_json_call(
                        "Match a booking purpose to an event type. Return JSON: {\"index\": <0-based int>} or {\"index\": null}.",
                        f"Purpose: \"{purpose}\"\nEvent types:\n" +
                        "\n".join(f"{i}. {et['name']}" + (f" — {et['description']}" if et.get('description') else "") for i, et in enumerate(event_types)),
                    )
                    idx = result.get("index")
                    if idx is not None and 0 <= int(idx) < len(event_types):
                        matched_uri = event_types[int(idx)]["event_type_uri"]
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

        if slots_data:
            # Numbered list used internally only for LLM choice identification — never spoken to caller
            numbered = "\n".join(f"{i+1}. {s['description']}" for i, s in enumerate(slots_data))
            user_msg = f"Available slots:\n{numbered}\nCaller said: \"{utterance}\""
            try:
                result = llm_json_call(
                    "Identify which slot the caller chose based on their description. "
                    "Return JSON: {\"slot\": <1-based int>} or {\"slot\": null}.",
                    user_msg,
                    history=llm_history,
                )
                slot_num = result.get("slot")
                if slot_num is not None:
                    idx = int(slot_num) - 1
                    if 0 <= idx < len(slots_data):
                        chosen = slots_data[idx]
                        internal_state["chosen_slot_id"] = idx
                        internal_state["stage"] = "awaiting_confirm"
                        internal_state["retry_count"] = 0
                        speak = llm_text_call(
                            "Generate a single voice sentence confirming a chosen appointment slot and asking for final confirmation.",
                            f"Slot: {chosen['description']}\nPattern: 'I'll book you for [slot]. Shall I confirm?'",
                            history=llm_history,
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
            except Exception as exc:
                logger.warning("SchedulingAgent slot choice failed: %s", exc)

        # Check for date preference
        current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        try:
            pref = llm_json_call(
                f"Today is {current_date}. Convert a natural language time preference to a UTC date range. "
                "Return JSON: {{\"start_time\": \"ISO\", \"end_time\": \"ISO\"}} or {{\"found\": false}}.",
                f"Caller said: \"{utterance}\"",
            )
            if pref.get("start_time") and pref.get("end_time"):
                from datetime import datetime as dt
                s = dt.fromisoformat(pref["start_time"].replace("Z", "+00:00"))
                e = dt.fromisoformat(pref["end_time"].replace("Z", "+00:00"))
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
                    return SubagentResponse(
                        status=AgentStatus.IN_PROGRESS,
                        speak="I'm sorry, I don't see any openings in that time range. Would you like to try a different time?",
                        internal_state=internal_state,
                    )
                speak = self._speak_slots(new_slots)
                return SubagentResponse(
                    status=AgentStatus.IN_PROGRESS,
                    speak=speak,
                    internal_state=internal_state,
                )
        except Exception as exc:
            logger.warning("SchedulingAgent date preference failed: %s", exc)

        # Retry
        retry = internal_state.get("retry_count", 0) + 1
        internal_state["retry_count"] = retry
        if retry >= self.MAX_RETRIES and slots_data:
            chosen = slots_data[0]
            internal_state["chosen_slot_id"] = 0
            internal_state["stage"] = "awaiting_confirm"
            speak = llm_text_call(
                "Generate a voice sentence confirming an appointment slot and asking for final confirmation.",
                f"Slot: {chosen['description']}",
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
        intent = _detect_confirmation(utterance)
        slots_data = internal_state.get("available_slots", [])
        chosen_idx = internal_state.get("chosen_slot_id", 0)

        if intent is True:
            if not slots_data or chosen_idx >= len(slots_data):
                return SubagentResponse(
                    status=AgentStatus.COMPLETED,
                    speak="Your appointment has been confirmed. We'll see you then!",
                    internal_state=internal_state,
                )
            chosen_data = slots_data[chosen_idx]

            # Reconstruct TimeSlot object for book_time_slot
            from app.services.calendar_service import TimeSlot
            # Parse stored ISO datetime strings back to datetime objects
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

        if intent is False:
            internal_state["stage"] = "awaiting_choice"
            internal_state["retry_count"] = 0
            if slots_data:
                speak = self._speak_slots(slots_data)
            else:
                speak = "No problem. When would work best for you?"
            return SubagentResponse(
                status=AgentStatus.IN_PROGRESS,
                speak=speak,
                internal_state=internal_state,
            )

        # Unclear
        chosen_data = slots_data[chosen_idx] if slots_data and chosen_idx < len(slots_data) else None
        desc = chosen_data["description"] if chosen_data else "the selected time"
        return SubagentResponse(
            status=AgentStatus.WAITING_CONFIRM,
            speak=f"Should I book {desc} for you? Please say yes to confirm or no to pick a different time.",
            pending_confirmation={"slot": desc},
            internal_state=internal_state,
        )
