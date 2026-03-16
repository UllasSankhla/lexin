"""Explicit state-machine workflow for collecting booking parameters
and scheduling an appointment.

HIGH-LEVEL FLOW
===============
  COLLECTING
    For each parameter (in collection_order):
      1. ASKING         – generate & speak a question
      2. AWAITING_VALUE – wait for caller utterance
      3. (extract + validate in-process)
      4. SPELLING_BACK  – speak the value back to caller
      5. AWAITING_CONFIRM – wait for yes / no
      6. If confirmed → persist → next field
      7. If rejected  → re-ask (up to max_retries)

  SCHEDULING
      1. PRESENTING_SLOTS – fetch calendar slots, speak them
      2. AWAITING_CHOICE  – wait for caller to choose a slot
      3. SPELLING_BACK    – speak the chosen slot back
      4. AWAITING_CONFIRM – wait for yes / no
      5. If confirmed → BOOKING → speak confirmation
      6. If rejected  → re-present slots

  COMPLETING
      Booking done; call_complete flag set on final WorkflowResult.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from datetime import datetime, timedelta, timezone

from app.pipeline.parameter_collector import CollectionState, ParameterSpec
from app.services.calendar_service import TimeSlot, book_time_slot, list_available_slots

logger = logging.getLogger(__name__)


# ── Stage enums ───────────────────────────────────────────────────────────────

class WorkflowStage(str, Enum):
    COLLECTING = "collecting"
    SCHEDULING = "scheduling"
    COMPLETING  = "completing"
    DONE        = "done"


class FieldStage(str, Enum):
    ASKING           = "asking"
    AWAITING_VALUE   = "awaiting_value"
    SPELLING_BACK    = "spelling_back"
    AWAITING_CONFIRM = "awaiting_confirm"


class ScheduleStage(str, Enum):
    PRESENTING_SLOTS = "presenting_slots"
    AWAITING_CHOICE  = "awaiting_choice"
    SPELLING_BACK    = "spelling_back"
    AWAITING_CONFIRM = "awaiting_confirm"


# ── Result ────────────────────────────────────────────────────────────────────

@dataclass
class WorkflowResult:
    """Returned by process_utterance / get_opening for the handler to act on."""
    speak: str
    # Set when a field has just been confirmed and should be persisted to DB
    field_confirmed: Optional[tuple[str, str, str]] = None  # (name, raw, normalized)
    # Set when a derived field (e.g. case_type) should also be persisted
    extra_field_confirmed: Optional[tuple[str, str, str]] = None
    # Set when a booking was just made
    booking_details: Optional[dict] = None
    # True when the conversation is over
    call_complete: bool = False


# ── Workflow ──────────────────────────────────────────────────────────────────

class BookingWorkflow:
    """
    Drives the booking conversation as a deterministic state machine.

    The handler calls:
        text = workflow.get_opening()          # after greeting
        result = workflow.process_utterance(text)   # on every caller utterance

    All LLM calls are synchronous (via LLMToolkit) so the handler should run
    this inside asyncio.get_event_loop().run_in_executor(None, ...).
    """

    MAX_RETRIES = 3

    def __init__(
        self,
        collection_state: CollectionState,
        llm_toolkit: "LLMToolkit",   # noqa: F821  (forward ref avoids circular import)
        config: dict,
    ) -> None:
        self.collection_state = collection_state
        self.llm = llm_toolkit
        self.config = config

        # Workflow stage
        self.stage: WorkflowStage = WorkflowStage.COLLECTING
        self.field_stage: FieldStage = FieldStage.ASKING
        self.schedule_stage: Optional[ScheduleStage] = None

        # Transient per-field state
        self._pending_value: Optional[str] = None    # normalised value awaiting confirm
        self._pending_raw: Optional[str] = None      # raw caller utterance
        self._retry_count: int = 0

        # Scheduling state
        self._available_slots: list[TimeSlot] = []
        self._chosen_slot: Optional[TimeSlot] = None
        self._matched_event_type_uri: Optional[str] = None
        # Tracks whether we are currently in an initial (7-day) search or a
        # user-directed search so we can give a tailored "no slots" message.
        self._initial_slot_search: bool = True

    # ── Public API ────────────────────────────────────────────────────────────

    def get_opening(self) -> str:
        """
        Returns the first thing to say after the greeting.
        Asks for the first uncollected field, or jumps to scheduling if there
        are no parameters configured.
        """
        field = self.collection_state.next_uncollected
        if not field:
            return self._enter_scheduling()
        return self._ask_for_field(field)

    def process_utterance(self, caller_text: str) -> WorkflowResult:
        """Process one caller utterance and return what to say / do next."""
        if self.stage == WorkflowStage.COLLECTING:
            return self._handle_collecting(caller_text)
        if self.stage == WorkflowStage.SCHEDULING:
            return self._handle_scheduling(caller_text)
        # COMPLETING / DONE – shouldn't normally be called
        return WorkflowResult(speak="Thank you. Your appointment is confirmed. Goodbye!", call_complete=True)

    # ── COLLECTING stage ──────────────────────────────────────────────────────

    def _handle_collecting(self, text: str) -> WorkflowResult:
        field = self.collection_state.next_uncollected
        if not field:
            return WorkflowResult(speak=self._enter_scheduling())

        if self.field_stage == FieldStage.AWAITING_VALUE:
            return self._process_field_value(field, text)

        if self.field_stage == FieldStage.AWAITING_CONFIRM:
            return self._process_field_confirmation(field, text)

        # Unexpected state – recover by re-asking
        logger.warning("Unexpected field_stage=%s; re-asking field %s", self.field_stage, field.name)
        return WorkflowResult(speak=self._ask_for_field(field))

    def _ask_for_field(self, field: ParameterSpec) -> str:
        question = self.llm.generate_field_question(field.display_label, field.data_type)
        self.field_stage = FieldStage.AWAITING_VALUE
        self._retry_count = 0
        logger.debug("Asking for field %s", field.name)
        return question

    def _process_field_value(self, field: ParameterSpec, text: str) -> WorkflowResult:
        """Extract, validate, then spell back OR retry."""
        extracted = self.llm.extract_value(field.display_label, field.data_type, text)

        if not extracted:
            return self._retry_or_skip(field, f"Sorry, I didn't catch that. Could you please repeat your {field.display_label}?")

        is_valid, result = self.collection_state.validate_value(field, extracted)
        if not is_valid:
            # result is the validation error message
            return self._retry_or_skip(field, f"{result} Could you please try again?")

        # Value looks good — spell it back
        self._pending_value = result      # normalised
        self._pending_raw = text
        self.field_stage = FieldStage.AWAITING_CONFIRM
        spell_back_text = self.llm.spell_back(field.display_label, field.data_type, result)
        logger.debug("Spelling back field %s = %r", field.name, result)
        return WorkflowResult(speak=spell_back_text)

    def _process_field_confirmation(self, field: ParameterSpec, text: str) -> WorkflowResult:
        """Handle yes / no / correction in response to spell-back."""
        intent = _detect_confirmation(text)

        if intent is True:
            return self._confirm_field(field)

        if intent is False:
            # Caller rejected – check if they embedded a correction in the same utterance
            correction = self.llm.extract_correction(field.display_label, field.data_type, text)
            if correction:
                is_valid, result = self.collection_state.validate_value(field, correction)
                if is_valid:
                    self._pending_value = result
                    self._pending_raw = text
                    spell_back_text = self.llm.spell_back(field.display_label, field.data_type, result)
                    return WorkflowResult(speak=spell_back_text)

            # No embedded correction – ask again from scratch
            self.field_stage = FieldStage.AWAITING_VALUE
            self._pending_value = None
            self._pending_raw = None
            return WorkflowResult(speak=f"No problem. Could you please say your {field.display_label} again?")

        # Unclear — repeat the spell-back
        return WorkflowResult(
            speak=f"Sorry, I didn't catch that. Is your {field.display_label} {self._pending_value}? Please say yes or no."
        )

    def _confirm_field(self, field: ParameterSpec) -> WorkflowResult:
        """Persist the pending value and advance."""
        confirmed_value = self._pending_value
        raw = self._pending_raw
        self._pending_value = None
        self._pending_raw = None

        self.collection_state.record_collected(field.name, raw, confirmed_value)
        logger.info("Field %s confirmed: %r", field.name, confirmed_value)

        # For case description fields, derive a case type and store it alongside.
        extra_confirmed: Optional[tuple[str, str, str]] = None
        if field.data_type == "case_description" and confirmed_value:
            try:
                case_type = self.llm.classify_case_type(confirmed_value)
                self.collection_state.record_collected("case_type", confirmed_value, case_type)
                logger.info("Case type classified: %r", case_type)
                extra_confirmed = ("case_type", confirmed_value, case_type)
            except Exception as exc:
                logger.warning("Case type classification failed: %s", exc)

        next_text = self._advance_field()
        result = WorkflowResult(
            speak=next_text,
            field_confirmed=(field.name, raw, confirmed_value),
        )
        # Attach case_type as a second confirmed field so the handler persists it too
        if extra_confirmed:
            result.extra_field_confirmed = extra_confirmed
        return result

    def _advance_field(self) -> str:
        """Move to the next field or start scheduling."""
        next_field = self.collection_state.next_uncollected
        if next_field:
            return self._ask_for_field(next_field)
        return self._enter_scheduling()

    def _retry_or_skip(self, field: ParameterSpec, retry_message: str) -> WorkflowResult:
        """Increment retry counter; skip optional field after too many failures."""
        self._retry_count += 1
        if self._retry_count >= self.MAX_RETRIES:
            self._retry_count = 0
            if not field.required:
                logger.info("Skipping optional field %s after %d retries", field.name, self.MAX_RETRIES)
                self.collection_state.record_collected(field.name, "", "")
                return WorkflowResult(speak=self._advance_field())
            logger.warning("Required field %s still unresolved after %d retries", field.name, self.MAX_RETRIES)
            return WorkflowResult(
                speak=f"I'm having difficulty capturing your {field.display_label}. "
                      f"Could you say it slowly and clearly, one more time?"
            )
        return WorkflowResult(speak=retry_message)

    # ── SCHEDULING stage ──────────────────────────────────────────────────────

    def _enter_scheduling(self) -> str:
        """Transition to SCHEDULING and present slots for the next 7 days."""
        self.stage = WorkflowStage.SCHEDULING
        self.schedule_stage = ScheduleStage.PRESENTING_SLOTS
        self._retry_count = 0
        self._initial_slot_search = True

        purpose = self._derive_purpose()

        # LLM-match caller intent to a Calendly event type when configured
        event_types = self.config.get("calendly_event_types") or []
        if event_types:
            matched_uri = self.llm.match_event_type(purpose, event_types)
            if matched_uri:
                self._matched_event_type_uri = matched_uri
                logger.info(
                    "Scheduling: matched event_type_uri=%s for purpose=%r",
                    matched_uri, purpose,
                )
            else:
                self._matched_event_type_uri = event_types[0]["event_type_uri"]
                logger.warning(
                    "Scheduling: no event type matched for purpose=%r — defaulting to %s",
                    purpose, self._matched_event_type_uri,
                )

        # Initial search: next 7 days only
        now = datetime.now(timezone.utc)
        search_start = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        search_end = search_start + timedelta(days=7)

        logger.info(
            "Scheduling: initial 7-day search | start=%s end=%s",
            search_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            search_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        )

        self._available_slots = list_available_slots(
            purpose, self.config, self._matched_event_type_uri,
            search_start=search_start, search_end=search_end,
        )
        logger.info(
            "Scheduling: %d slot(s) in initial window for purpose=%r",
            len(self._available_slots), purpose,
        )

        if not self._available_slots:
            # Nothing in the next 7 days — ask for preference immediately
            self.schedule_stage = ScheduleStage.AWAITING_CHOICE
            return (
                "I don't have any openings in the next seven days. "
                "When would work best for you? For example, you can say 'next Tuesday morning' "
                "or 'next week'."
            )

        self.schedule_stage = ScheduleStage.AWAITING_CHOICE
        return self.llm.present_slots(self._available_slots)

    def _fetch_and_present_slots(self, date_pref: dict) -> str:
        """
        Query Calendly for the user-specified date range and present the results.

        date_pref is {"start_time": ISO, "end_time": ISO} from extract_date_preference.
        Returns a voice string (either slot list or "no openings" prompt).
        """
        def _parse(iso: str) -> datetime:
            return datetime.fromisoformat(iso.replace("Z", "+00:00"))

        search_start = _parse(date_pref["start_time"])
        search_end = _parse(date_pref["end_time"])

        logger.info(
            "Scheduling: user-directed search | start=%s end=%s",
            date_pref["start_time"], date_pref["end_time"],
        )

        purpose = self._derive_purpose()
        self._available_slots = list_available_slots(
            purpose, self.config, self._matched_event_type_uri,
            search_start=search_start, search_end=search_end,
        )
        self._initial_slot_search = False

        logger.info(
            "Scheduling: %d slot(s) in user-directed window for purpose=%r",
            len(self._available_slots), purpose,
        )

        if not self._available_slots:
            return (
                "I'm sorry, I don't see any openings in that time range. "
                "Would you like to try a different time? You can say something like "
                "'the following week' or 'early next month'."
            )

        return self.llm.present_slots(self._available_slots)

    def _handle_scheduling(self, text: str) -> WorkflowResult:
        if self.schedule_stage == ScheduleStage.AWAITING_CHOICE:
            return self._process_slot_choice(text)
        if self.schedule_stage == ScheduleStage.AWAITING_CONFIRM:
            return self._process_slot_confirmation(text)
        return WorkflowResult(speak=self.llm.present_slots(self._available_slots))

    def _process_slot_choice(self, text: str) -> WorkflowResult:
        # First try to match a specific slot from what the caller said
        slot = self.llm.extract_slot_choice(text, self._available_slots) if self._available_slots else None

        if slot:
            self._chosen_slot = slot
            self._retry_count = 0
            self.schedule_stage = ScheduleStage.AWAITING_CONFIRM
            confirm_text = self.llm.confirm_slot(slot)
            logger.debug("User chose slot: %s", slot.slot_id)
            return WorkflowResult(speak=confirm_text)

        # No slot match — check if the caller is expressing a date preference
        current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        date_pref = self.llm.extract_date_preference(text, current_date)

        if date_pref:
            logger.info(
                "Scheduling: caller expressed date preference | start=%s end=%s",
                date_pref.get("start_time"), date_pref.get("end_time"),
            )
            new_slots_text = self._fetch_and_present_slots(date_pref)
            self._retry_count = 0
            # Stay in AWAITING_CHOICE so the caller can pick from the new list
            return WorkflowResult(speak=new_slots_text)

        # Neither — retry or default to first slot after too many attempts
        self._retry_count += 1
        if self._retry_count >= self.MAX_RETRIES:
            if self._available_slots:
                slot = self._available_slots[0]
                self._chosen_slot = slot
                self._retry_count = 0
                self.schedule_stage = ScheduleStage.AWAITING_CONFIRM
                logger.info("Defaulting to first slot after %d failed choices", self.MAX_RETRIES)
                return WorkflowResult(speak=self.llm.confirm_slot(slot))
            # No slots at all — shouldn't normally happen
            self.stage = WorkflowStage.COMPLETING
            return WorkflowResult(
                speak="I'm sorry, I wasn't able to find an available slot. Please call back and we'll help you schedule.",
                call_complete=True,
            )

        if self._available_slots:
            re_present = self.llm.present_slots(self._available_slots)
            return WorkflowResult(
                speak=f"Sorry, I didn't catch that. {re_present}"
            )
        return WorkflowResult(
            speak=(
                "I don't have any slots loaded right now. "
                "Could you tell me what time works best for you? For example, 'next Tuesday morning'."
            )
        )

    def _process_slot_confirmation(self, text: str) -> WorkflowResult:
        intent = _detect_confirmation(text)

        if intent is True:
            booking = book_time_slot(
                self._chosen_slot,
                self.collection_state.collected,
                self.config,
            )
            self.stage = WorkflowStage.COMPLETING
            speak = self.llm.confirm_booking(
                self._chosen_slot, booking, self.collection_state.collected
            )
            logger.info("Booking confirmed: %s", booking.get("booking_id"))
            return WorkflowResult(speak=speak, booking_details=booking, call_complete=True)

        if intent is False:
            self.schedule_stage = ScheduleStage.AWAITING_CHOICE
            self._retry_count = 0
            re_present = self.llm.present_slots(self._available_slots)
            return WorkflowResult(speak=f"No problem. {re_present}")

        # Unclear
        slot_desc = self._chosen_slot.description if self._chosen_slot else "the selected slot"
        return WorkflowResult(
            speak=f"Should I book {slot_desc} for you? Please say yes to confirm or no to pick a different time."
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _derive_purpose(self) -> str:
        """Infer booking purpose from collected params or assistant config."""
        for key in ("purpose", "reason", "service", "appointment_type", "visit_type"):
            if key in self.collection_state.collected:
                return self.collection_state.collected[key]
        return self.config.get("assistant", {}).get("persona_description", "appointment")


# ── Confirmation detector (regex — no LLM overhead needed) ───────────────────

_YES_RE = re.compile(
    r"\b(yes|yeah|yep|yup|correct|right|exactly|that'?s right|that is right|"
    r"confirmed|confirm|sure|ok|okay|perfect|great|sounds good|go ahead|"
    r"proceed|affirmative|absolutely|definitely|of course)\b",
    re.IGNORECASE,
)
_NO_RE = re.compile(
    r"\b(no|nope|nah|wrong|incorrect|that'?s wrong|that is wrong|not right|"
    r"change|different|actually|wait|hold on|negative|not quite|not exactly)\b",
    re.IGNORECASE,
)


def _detect_confirmation(text: str) -> Optional[bool]:
    """
    Returns True (confirmed), False (rejected), or None (unclear).
    Requires an unambiguous signal — both patterns together → unclear.
    """
    has_yes = bool(_YES_RE.search(text))
    has_no  = bool(_NO_RE.search(text))

    if has_yes and not has_no:
        return True
    if has_no and not has_yes:
        return False
    return None
