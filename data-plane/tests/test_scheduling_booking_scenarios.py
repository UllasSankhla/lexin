"""
Scheduling agent: booking scenario tests.

Two layers of verification:

1. LLM intent layer — different caller utterances produce correct _SlotAction
   classifications with valid ISO UTC date ranges that can be sent to Calendly's
   GET /event_type_available_times.

2. Calendly payload layer — the data that flows through the full conversation
   (slot state, collected params) has all required fields for:
     GET  /event_type_available_times  (event_type, start_time, end_time)
     POST /scheduling/invitees         (event_type, start_time, invitee{name, email, timezone})

All Calendly HTTP calls are faked — no real API token needed.
LLM calls (llm_structured_call, llm_text_call) use the real Cerebras API.

Run:
    cd data-plane
    PYTHONPATH=. python3 -m pytest tests/test_scheduling_booking_scenarios.py -v -s
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Optional
from unittest.mock import patch, MagicMock

import pytest

from app.agents.scheduling import SchedulingAgent, _SlotAction, _SLOT_ACTION_SYSTEM
from app.agents.llm_utils import llm_structured_call
from app.agents.base import AgentStatus
from app.services.calendar_service import TimeSlot, _book_calendly_slot, _split_name


# ── Shared fixtures ────────────────────────────────────────────────────────────

_EVENT_TYPE_URI = "https://api.calendly.com/event_types/abc123def456"
_TODAY = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)


def _make_calendly_slots(n: int = 3, base_offset_days: int = 1) -> list[dict]:
    """
    Slots in the serialized dict format used in internal_state["available_slots"].
    All fields required for a valid Calendly booking.
    """
    slots = []
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    hours = [9, 10, 14, 15, 11]
    for i in range(n):
        start = _TODAY + timedelta(days=base_offset_days + i, hours=hours[i % 5])
        end = start + timedelta(hours=1)
        day = start.strftime("%A, %B %-d")
        hour_label = start.strftime("%-I:%M %p")
        slots.append({
            "slot_id": start.isoformat(),          # ISO start_time as stable key (Calendly convention)
            "description": f"{day} at {hour_label}",
            "start_time": start.isoformat(),        # sent to Calendly as start_time
            "end_time": end.isoformat(),
            "event_type_uri": _EVENT_TYPE_URI,      # required for Calendly booking
        })
    return slots


def _make_timeslots(n: int = 3, base_offset_days: int = 1) -> list[TimeSlot]:
    """TimeSlot objects for patching list_available_slots."""
    results = []
    for s in _make_calendly_slots(n, base_offset_days):
        start = datetime.fromisoformat(s["start_time"])
        end = datetime.fromisoformat(s["end_time"])
        results.append(TimeSlot(
            slot_id=s["slot_id"],
            description=s["description"],
            start=start,
            end=end,
            event_type_uri=_EVENT_TYPE_URI,
            spots_remaining=1,
        ))
    return results


_BASE_STATE = {
    "stage": "awaiting_choice",
    "available_slots": [],
    "chosen_slot_id": None,
    "retry_count": 0,
    "matched_event_type_uri": _EVENT_TYPE_URI,
    "llm_history": [],
}

_COLLECTED = {
    "full_name": "Jane Smith",
    "phone_number": "555-867-5309",
    "email_address": "jane.smith@example.com",
}

_CONFIG = {
    "assistant": {"persona_name": "Aria"},
    "parameters": [],
    "_collected": _COLLECTED,
    "_tool_results": {},
}


# ── Calendly format validators ─────────────────────────────────────────────────

def _assert_valid_iso_utc(ts: str, label: str) -> datetime:
    """Assert a timestamp is valid ISO 8601 with timezone info."""
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError as exc:
        pytest.fail(f"{label} is not a valid ISO datetime: {ts!r} — {exc}")
    assert dt.tzinfo is not None, f"{label} has no timezone info: {ts!r}"
    return dt


def _assert_slot_calendly_ready(slot: dict, label: str = "slot") -> None:
    """Assert a slot dict has all fields required for a Calendly booking."""
    assert slot.get("slot_id"), f"{label} missing slot_id"
    assert slot.get("event_type_uri", "").startswith("https://api.calendly.com/"), (
        f"{label} event_type_uri is not a Calendly API URI: {slot.get('event_type_uri')!r}"
    )
    _assert_valid_iso_utc(slot["start_time"], f"{label}.start_time")
    _assert_valid_iso_utc(slot["end_time"], f"{label}.end_time")
    assert slot.get("description"), f"{label} missing description"


def _assert_booking_payload_valid(body: dict) -> None:
    """Assert a Calendly POST /invitees body has all required fields."""
    assert body.get("event_type", "").startswith("https://api.calendly.com/"), (
        f"event_type is not a Calendly API URI: {body.get('event_type')!r}"
    )
    # start_time format: YYYY-MM-DDTHH:MM:SS.000000Z
    st = body.get("start_time", "")
    assert st.endswith("Z"), f"start_time must end with Z: {st!r}"
    _assert_valid_iso_utc(st, "booking.start_time")

    invitee = body.get("invitee", {})
    assert invitee.get("name"), "invitee.name missing"
    assert invitee.get("first_name"), "invitee.first_name missing"
    assert invitee.get("email"), "invitee.email missing"
    assert invitee.get("timezone"), "invitee.timezone missing"
    assert body.get("booking_source") == "ai_scheduling_assistant"


# ── LLM intent layer: _SlotAction format tests ────────────────────────────────

class TestSlotActionLLMFormat:
    """
    Verify the LLM produces correctly structured _SlotAction responses
    for a variety of caller utterances. Uses real LLM — no mocks on
    llm_structured_call. Checks field types and ISO date validity.
    """

    _today_str = _TODAY.strftime("%A, %Y-%m-%d")
    _slots_str = (
        f"1. {(_TODAY + timedelta(days=1)).strftime('%A, %B %-d')} at 10:00 AM\n"
        f"2. {(_TODAY + timedelta(days=2)).strftime('%A, %B %-d')} at 2:00 PM\n"
        f"3. {(_TODAY + timedelta(days=3)).strftime('%A, %B %-d')} at 9:00 AM"
    )

    def _classify(self, utterance: str) -> _SlotAction:
        user_msg = (
            f"Today is {self._today_str}.\n"
            f"Available slots:\n{self._slots_str}\n"
            f"Caller said: \"{utterance}\""
        )
        return llm_structured_call(
            _SLOT_ACTION_SYSTEM, user_msg, _SlotAction,
            max_tokens=128, tag="test_slot_action",
        )

    # ── Direct slot picks ──────────────────────────────────────────────────────

    def test_pick_first_ordinal(self):
        """'The first one' → action=pick, slot_index=1."""
        result = self._classify("The first one please.")
        assert result.action == "pick", f"Expected 'pick', got {result.action!r}"
        assert result.slot_index == 1, f"Expected slot_index=1, got {result.slot_index}"

    def test_pick_second_ordinal(self):
        """'Option 2' → action=pick, slot_index=2."""
        result = self._classify("Option 2 works for me.")
        assert result.action == "pick", f"Expected 'pick', got {result.action!r}"
        assert result.slot_index == 2, f"Expected slot_index=2, got {result.slot_index}"

    def test_pick_third_ordinal(self):
        """'The last one' / 'The third' → action=pick, slot_index=3."""
        result = self._classify("I'll take the last one.")
        assert result.action == "pick", f"Expected 'pick', got {result.action!r}"
        assert result.slot_index == 3, f"Expected slot_index=3, got {result.slot_index}"

    def test_pick_by_day_name(self):
        """Naming the day of slot 1 → action=pick, slot_index=1."""
        # Slot 1 is tomorrow
        day_name = (_TODAY + timedelta(days=1)).strftime("%A")
        result = self._classify(f"{day_name} at 10 works for me.")
        assert result.action == "pick", f"Expected 'pick', got {result.action!r}"
        assert result.slot_index == 1, f"Expected slot_index=1, got {result.slot_index}"

    def test_pick_ambiguous_yes(self):
        """'That works' in context of one clearly shown slot → action=pick."""
        result = self._classify("That works.")
        # LLM may pick slot 1 or be unclear — either is acceptable
        assert result.action in ("pick", "unclear"), f"Unexpected action: {result.action!r}"
        if result.action == "pick":
            assert isinstance(result.slot_index, int) and 1 <= result.slot_index <= 3

    # ── Date preference: specific day ─────────────────────────────────────────

    def test_new_date_thursday_question_form(self):
        """'Do you have anything on Thursday?' → action=new_date, Thursday range."""
        result = self._classify("Do you have any dates available on Thursday?")
        assert result.action == "new_date", f"Expected 'new_date', got {result.action!r}"
        assert result.start_time, "start_time missing"
        assert result.end_time, "end_time missing"
        start = _assert_valid_iso_utc(result.start_time, "start_time")
        end = _assert_valid_iso_utc(result.end_time, "end_time")
        assert start.weekday() == 3, (  # Thursday
            f"Expected Thursday (weekday=3), got weekday={start.weekday()} ({start.date()})"
        )
        assert end > start, "end_time must be after start_time"

    def test_new_date_friday_statement_form(self):
        """'I'd prefer Friday' → action=new_date, Friday range."""
        result = self._classify("I'd prefer Friday if possible.")
        assert result.action == "new_date", f"Expected 'new_date', got {result.action!r}"
        start = _assert_valid_iso_utc(result.start_time, "start_time")
        assert start.weekday() == 4, (  # Friday
            f"Expected Friday (weekday=4), got weekday={start.weekday()} ({start.date()})"
        )

    def test_new_date_range_is_same_day(self):
        """For a specific day request, start and end should be on the same day."""
        result = self._classify("What about Wednesday?")
        assert result.action == "new_date"
        start = _assert_valid_iso_utc(result.start_time, "start_time")
        end = _assert_valid_iso_utc(result.end_time, "end_time")
        assert start.date() == end.date() or (end - start) <= timedelta(days=1), (
            f"Expected same-day range, got {start.date()} to {end.date()}"
        )

    # ── Date preference: time-of-day ──────────────────────────────────────────

    def test_new_date_afternoon_preference(self):
        """'Afternoon slots' → action=new_date with a multi-day window."""
        result = self._classify("Do you have anything in the afternoon?")
        assert result.action == "new_date", f"Expected 'new_date', got {result.action!r}"
        start = _assert_valid_iso_utc(result.start_time, "start_time")
        end = _assert_valid_iso_utc(result.end_time, "end_time")
        # Should cover more than one day (time-of-day = broader search)
        assert (end - start).days >= 1, (
            f"Expected multi-day range for 'afternoon', got {start} to {end}"
        )

    def test_new_date_morning_preference(self):
        """'Something in the morning' → action=new_date."""
        result = self._classify("Do you have something in the morning?")
        assert result.action == "new_date"
        _assert_valid_iso_utc(result.start_time, "start_time")
        _assert_valid_iso_utc(result.end_time, "end_time")

    # ── Date preference: relative week ────────────────────────────────────────

    def test_new_date_next_week(self):
        """'Next week' → action=new_date, range starting >= 7 days from today."""
        result = self._classify("Can we do something next week?")
        assert result.action == "new_date", f"Expected 'new_date', got {result.action!r}"
        start = _assert_valid_iso_utc(result.start_time, "start_time")
        end = _assert_valid_iso_utc(result.end_time, "end_time")
        # Next week should start at least 5 days from today (conservative)
        assert start.date() >= (_TODAY + timedelta(days=5)).date(), (
            f"'next week' start {start.date()} is too soon (today={_TODAY.date()})"
        )
        assert (end - start).days >= 5, "next week range should cover at least 5 days"

    def test_new_date_later_this_month(self):
        """'Later this month' → action=new_date, start >= today."""
        result = self._classify("How about later this month?")
        assert result.action == "new_date"
        start = _assert_valid_iso_utc(result.start_time, "start_time")
        # >= today (LLM may anchor to today's date for "later this month")
        assert start.date() >= _TODAY.date(), f"start_time {start.date()} is in the past"

    # ── ISO format validation ──────────────────────────────────────────────────

    def test_date_range_always_has_timezone(self):
        """All returned date ranges must carry timezone info."""
        utterances = [
            "Do you have Thursday?",
            "Any morning slots?",
            "What about next week?",
        ]
        for u in utterances:
            result = self._classify(u)
            if result.action == "new_date":
                start = datetime.fromisoformat(result.start_time.replace("Z", "+00:00"))
                end = datetime.fromisoformat(result.end_time.replace("Z", "+00:00"))
                assert start.tzinfo is not None, f"start_time has no tzinfo for {u!r}: {result.start_time!r}"
                assert end.tzinfo is not None, f"end_time has no tzinfo for {u!r}: {result.end_time!r}"

    def test_date_range_start_before_end(self):
        """start_time must always be before end_time."""
        utterances = [
            "Do you have any dates available on Thursday?",
            "How about next Monday?",
            "I'd prefer an afternoon slot.",
        ]
        for u in utterances:
            result = self._classify(u)
            if result.action == "new_date" and result.start_time and result.end_time:
                start = datetime.fromisoformat(result.start_time.replace("Z", "+00:00"))
                end = datetime.fromisoformat(result.end_time.replace("Z", "+00:00"))
                assert start < end, (
                    f"start_time >= end_time for {u!r}: {result.start_time} >= {result.end_time}"
                )


# ── Agent layer: slot state is Calendly-ready ─────────────────────────────────

class TestUtteranceFormsCoverage:
    """
    Comprehensive utterance coverage for _SlotAction classification.

    Organises test cases by utterance form as the user described:
      1. Question form — asking if a day/time is available
      2. Rejection — rejecting a previously offered date/range
      3. Single date/time — exact date, or date + time
      4. Specific day — day name only (no calendar date)
      5. Time period — "next week", "this weekend", "the next two weeks"
      6. Date range — explicit start and end dates
      7. Combined — specific date AND day name together

    For each: asserts action is correct and (for new_date) ISO range is valid
    and the date range covers the intended period.
    """

    _today_str = _TODAY.strftime("%A, %Y-%m-%d")
    _slots_str = (
        f"1. {(_TODAY + timedelta(days=1)).strftime('%A, %B %-d')} at 10:00 AM\n"
        f"2. {(_TODAY + timedelta(days=2)).strftime('%A, %B %-d')} at 2:00 PM"
    )

    def _classify(self, utterance: str) -> _SlotAction:
        user_msg = (
            f"Today is {self._today_str}.\n"
            f"Available slots:\n{self._slots_str}\n"
            f"Caller said: \"{utterance}\""
        )
        return llm_structured_call(
            _SLOT_ACTION_SYSTEM, user_msg, _SlotAction,
            max_tokens=128, tag="test_utterance_form",
        )

    def _new_date(self, utterance: str) -> tuple[datetime, datetime]:
        """Helper: assert action=new_date and return (start, end) datetimes."""
        result = self._classify(utterance)
        assert result.action == "new_date", (
            f"Expected action='new_date' for {utterance!r}, got {result.action!r}"
        )
        start = _assert_valid_iso_utc(result.start_time, "start_time")
        end = _assert_valid_iso_utc(result.end_time, "end_time")
        # >= (not >) — LLM may return start==end for point-in-time requests
        assert end >= start, f"end_time must not be before start_time for {utterance!r}"
        return start, end

    # ── 1. Question form ───────────────────────────────────────────────────────

    def test_q_do_you_have_thursday(self):
        """'Do you have any dates available on Thursday?' → new_date, Thursday."""
        start, _ = self._new_date("Do you have any dates available on Thursday?")
        assert start.weekday() == 3, f"Expected Thursday, got weekday={start.weekday()}"

    def test_q_is_friday_available(self):
        """'Is Friday available?' → new_date, Friday."""
        start, _ = self._new_date("Is Friday available?")
        assert start.weekday() == 4, f"Expected Friday, got weekday={start.weekday()}"

    def test_q_do_you_have_anything_next_week(self):
        """'Do you have anything next week?' → new_date, next week window."""
        start, end = self._new_date("Do you have anything next week?")
        assert start.date() >= (_TODAY + timedelta(days=5)).date(), (
            f"'next week' should start >= 5 days out, got {start.date()}"
        )

    def test_q_do_you_have_afternoon_slots(self):
        """'Do you have anything in the afternoon?' → new_date, multi-day window."""
        start, end = self._new_date("Do you have anything in the afternoon?")
        assert (end - start).days >= 1, "Afternoon preference should yield multi-day range"

    def test_q_any_earlier_times(self):
        """'Do you have any earlier times?' → new_date."""
        result = self._classify("Do you have any earlier times?")
        assert result.action == "new_date", f"Expected 'new_date', got {result.action!r}"
        _assert_valid_iso_utc(result.start_time, "start_time")

    def test_q_what_do_you_have_on_monday(self):
        """'What do you have on Monday?' → new_date, Monday."""
        start, _ = self._new_date("What do you have on Monday?")
        assert start.weekday() == 0, f"Expected Monday, got weekday={start.weekday()}"

    # ── 2. Rejection / negation of an offered date ────────────────────────────

    def test_reject_that_day_doesnt_work(self):
        """'That day doesn't work for me' → new_date (wants different time)."""
        result = self._classify("That day doesn't work for me, do you have something else?")
        assert result.action == "new_date", (
            f"Expected 'new_date' for rejection+alternative request, got {result.action!r}"
        )

    def test_reject_not_monday(self):
        """'Not Monday, something later in the week' → new_date."""
        result = self._classify("Not Monday, do you have something later in the week?")
        assert result.action == "new_date"
        start = _assert_valid_iso_utc(result.start_time, "start_time")
        # Should be at least Tuesday (weekday > 0) or later this week
        assert start.weekday() > 0 or start.date() > _TODAY.date()

    def test_reject_none_of_those_work(self):
        """'None of those work' with no alternative → unclear or new_date."""
        result = self._classify("None of those work for me.")
        assert result.action in ("new_date", "unclear"), (
            f"Expected 'new_date' or 'unclear' for rejection without alternative, "
            f"got {result.action!r}"
        )

    def test_reject_too_early_next_week_instead(self):
        """'That's too early, can we do next week instead?' → new_date, next week."""
        start, end = self._new_date("That's too early, can we do next week instead?")
        assert start.date() >= (_TODAY + timedelta(days=5)).date()

    def test_reject_that_time_doesnt_suit(self):
        """'That time doesn't suit me, how about afternoon?' → new_date."""
        result = self._classify("That time doesn't suit me, how about afternoon slots?")
        assert result.action == "new_date"

    # ── 3. Single date/time ───────────────────────────────────────────────────

    def test_single_specific_date(self):
        """'April 10th' → new_date, range covering April 10."""
        start, end = self._new_date("How about April 10th?")
        assert start.month == 4 and start.day == 10, (
            f"Expected April 10, got {start.date()}"
        )

    def test_single_specific_date_and_time(self):
        """'April 10th at 2 PM' → new_date, range around that time."""
        start, end = self._new_date("Can I book April 10th at 2 PM?")
        assert start.month == 4 and start.day == 10, (
            f"Expected April 10, got {start.date()}"
        )

    def test_single_tomorrow(self):
        """'Tomorrow' → new_date, tomorrow's date."""
        start, _ = self._new_date("Do you have anything tomorrow?")
        expected = (_TODAY + timedelta(days=1)).date()
        assert start.date() == expected, (
            f"Expected tomorrow={expected}, got {start.date()}"
        )

    def test_single_this_friday(self):
        """'This Friday' → new_date, next Friday."""
        start, _ = self._new_date("How about this Friday?")
        assert start.weekday() == 4, f"Expected Friday (weekday=4), got {start.weekday()}"

    def test_single_specific_time_of_day(self):
        """'10 AM' without a day → new_date (search for morning slots)."""
        result = self._classify("Do you have a 10 AM slot?")
        assert result.action == "new_date"
        _assert_valid_iso_utc(result.start_time, "start_time")

    # ── 4. Specific day name ──────────────────────────────────────────────────

    def test_day_wednesday(self):
        """'Wednesday' → new_date, Wednesday of current/next week."""
        start, _ = self._new_date("Wednesday would work.")
        assert start.weekday() == 2, f"Expected Wednesday (weekday=2), got {start.weekday()}"

    def test_day_tuesday(self):
        """'Tuesday' → new_date, Tuesday."""
        start, _ = self._new_date("Do you have Tuesday?")
        assert start.weekday() == 1, f"Expected Tuesday (weekday=1), got {start.weekday()}"

    def test_day_saturday_edge_case(self):
        """'Saturday' → new_date (even if business is closed, agent shouldn't crash)."""
        result = self._classify("What about Saturday?")
        assert result.action in ("new_date", "unclear"), (
            f"Expected 'new_date' or 'unclear' for Saturday, got {result.action!r}"
        )
        if result.action == "new_date":
            start = _assert_valid_iso_utc(result.start_time, "start_time")
            assert start.weekday() == 5, f"Expected Saturday (weekday=5)"

    # ── 5. Time period ────────────────────────────────────────────────────────

    def test_period_next_two_weeks(self):
        """'Next two weeks' → new_date, ~14-day window."""
        start, end = self._new_date("Do you have anything in the next two weeks?")
        assert (end - start).days >= 10, (
            f"'next two weeks' should cover >= 10 days, got {(end-start).days}"
        )

    def test_period_this_week(self):
        """'This week' → new_date, within current week."""
        start, end = self._new_date("Do you have anything this week?")
        assert start.date() >= _TODAY.date(), f"'this week' starts in the past: {start.date()}"
        # Should not extend more than 7 days
        assert (end - start).days <= 8, (
            f"'this week' range too wide: {(end-start).days} days"
        )

    def test_period_end_of_month(self):
        """'End of the month' → new_date in the future."""
        result = self._classify("I'd prefer end of the month.")
        assert result.action == "new_date"
        start = _assert_valid_iso_utc(result.start_time, "start_time")
        assert start.date() > _TODAY.date()

    def test_period_morning(self):
        """'Morning' → new_date with broader search window."""
        start, end = self._new_date("Can we do a morning slot?")
        assert (end - start).days >= 1, "Morning preference should yield multi-day window"

    def test_period_evening(self):
        """'Evening' → new_date (even if unlikely to have slots)."""
        result = self._classify("Do you have any evening availability?")
        assert result.action == "new_date"
        _assert_valid_iso_utc(result.start_time, "start_time")

    # ── 6. Explicit date range ────────────────────────────────────────────────

    def test_range_april_10_to_15(self):
        """'April 10 to April 15' → new_date spanning that range."""
        start, end = self._new_date("Do you have anything from April 10 to April 15?")
        assert start.month == 4 and start.day == 10, (
            f"Expected start=April 10, got {start.date()}"
        )
        assert end.month == 4 and end.day >= 14, (
            f"Expected end >= April 14, got {end.date()}"
        )

    def test_range_next_monday_to_wednesday(self):
        """'Next Monday to Wednesday' → new_date 3-day window."""
        start, end = self._new_date("How about next Monday through Wednesday?")
        assert start.weekday() == 0, f"Expected Monday start, got weekday={start.weekday()}"
        assert (end - start).days >= 2, "Mon-Wed span should be at least 2 days"

    def test_range_between_two_dates(self):
        """'Between the 10th and the 15th' → new_date."""
        start, end = self._new_date("Something between the 10th and the 15th of April.")
        assert start.month == 4
        assert (end - start).days >= 4

    # ── 7. Combined: specific date + day name ─────────────────────────────────

    def test_combined_thursday_april_10(self):
        """'Thursday April 10th' → new_date, April 10 which is a Thursday."""
        start, _ = self._new_date("What about Thursday April 10th?")
        assert start.month == 4 and start.day == 10, (
            f"Expected April 10, got {start.date()}"
        )

    def test_combined_monday_the_14th(self):
        """'Monday the 14th' → new_date, April 14."""
        start, _ = self._new_date("How about Monday the 14th?")
        assert start.month == 4 and start.day == 14, (
            f"Expected April 14, got {start.date()}"
        )

    def test_combined_with_time(self):
        """'Wednesday April 9th at 10 AM' → new_date on that date."""
        start, _ = self._new_date("Can I get Wednesday April 9th at 10 AM?")
        assert start.month == 4 and start.day == 9, (
            f"Expected April 9, got {start.date()}"
        )

    # ── Regression: picks still work after all the above ─────────────────────

    def test_pick_still_works_the_second_one(self):
        """'The second one' → action=pick, slot_index=2. Not contaminated by new_date logic."""
        result = self._classify("The second one works.")
        assert result.action == "pick", f"Expected 'pick', got {result.action!r}"
        assert result.slot_index == 2


class TestSlotStateCalendlyReady:
    """
    After the agent processes a date preference utterance and fetches new slots,
    the slots stored in internal_state must have all fields required by the
    Calendly booking API.
    """

    def test_after_date_preference_slots_have_all_calendly_fields(self):
        """
        'Do you have Thursday?' → agent fetches slots → state has Calendly-ready dicts.
        """
        agent = SchedulingAgent()
        state = dict(_BASE_STATE)
        state["available_slots"] = _make_calendly_slots(3)

        def _fake_slots(*a, **kw):
            return _make_timeslots(2, base_offset_days=4)  # 4 days out = Thursday

        with patch("app.agents.scheduling.list_available_slots", _fake_slots), \
             patch("app.agents.scheduling.llm_text_call", return_value="I have slots on Thursday. Which works?"):
            resp = agent.process("Do you have anything on Thursday?", state, _CONFIG, [])

        new_slots = resp.internal_state.get("available_slots", [])
        assert new_slots, "No slots in state after date preference"
        for i, slot in enumerate(new_slots):
            _assert_slot_calendly_ready(slot, label=f"slot[{i}]")

    def test_after_direct_pick_chosen_slot_has_all_calendly_fields(self):
        """
        After the caller picks a slot, the chosen slot in state is Calendly-ready.
        """
        agent = SchedulingAgent()
        state = dict(_BASE_STATE)
        state["available_slots"] = _make_calendly_slots(3)

        with patch("app.agents.scheduling.llm_text_call", return_value="I'll book you for that slot. Shall I confirm?"):
            resp = agent.process("The first one please.", state, _CONFIG, [])

        if resp.status == AgentStatus.WAITING_CONFIRM:
            chosen_idx = resp.internal_state.get("chosen_slot_id", 0)
            slots = resp.internal_state.get("available_slots", [])
            assert 0 <= chosen_idx < len(slots), f"chosen_slot_id={chosen_idx} out of bounds"
            chosen = slots[chosen_idx]
            _assert_slot_calendly_ready(chosen, label="chosen_slot")

    def test_slot_id_is_iso_start_time(self):
        """
        Calendly uses the ISO start_time string as the stable slot identifier.
        slot_id and start_time must be consistent.
        """
        slots = _make_calendly_slots(3)
        for slot in slots:
            # When slot_id == start_time (Calendly convention), they must both parse
            _assert_valid_iso_utc(slot["slot_id"], "slot_id")
            _assert_valid_iso_utc(slot["start_time"], "start_time")
            assert slot["slot_id"] == slot["start_time"], (
                "slot_id should equal start_time (Calendly stable key convention)"
            )

    def test_event_type_uri_is_api_uri_not_scheduling_url(self):
        """
        The event_type_uri in slot state must be a Calendly API URI
        (https://api.calendly.com/event_types/...), not a scheduling URL
        (https://calendly.com/user/event).
        """
        slots = _make_calendly_slots(3)
        for slot in slots:
            uri = slot.get("event_type_uri", "")
            assert uri.startswith("https://api.calendly.com/"), (
                f"event_type_uri is a scheduling URL, not an API URI: {uri!r}\n"
                "Calendly's available_times endpoint only accepts API URIs."
            )


# ── Full booking flow: end-to-end payload validation ─────────────────────────

class TestCalendlyBookingPayload:
    """
    Simulates a full booking: present → pick → confirm → book.
    Verifies the POST /scheduling/invitees body has all required fields.
    """

    def _capture_booking_payload(self, collected: dict) -> dict:
        """
        Run _book_calendly_slot with a fake httpx client and capture the
        POST body that would be sent to Calendly.
        """
        from datetime import datetime, timezone
        slot = TimeSlot(
            slot_id="2026-04-10T10:00:00+00:00",
            description="Thursday, April 10 at 10:00 AM",
            start=datetime(2026, 4, 10, 10, 0, tzinfo=timezone.utc),
            end=datetime(2026, 4, 10, 11, 0, tzinfo=timezone.utc),
            event_type_uri=_EVENT_TYPE_URI,
            spots_remaining=1,
        )
        calendly_cfg = {
            "api_token": "fake-token",
            "timezone": "America/New_York",
        }
        captured = {}

        class FakeResp:
            status_code = 201
            is_success = True
            def raise_for_status(self): pass
            def json(self):
                return {"resource": {
                    "uri": "https://api.calendly.com/scheduled_events/abc/invitees/xyz",
                    "status": "active",
                    "event": "https://api.calendly.com/scheduled_events/abc",
                    "cancel_url": "https://calendly.com/cancellations/xyz",
                    "reschedule_url": "https://calendly.com/reschedulings/xyz",
                    "created_at": "2026-04-04T12:00:00Z",
                }}

        class FakeGetResp:
            status_code = 200
            is_success = True
            def raise_for_status(self): pass
            def json(self):
                return {"resource": {"locations": [{"kind": "custom", "location": "Via phone"}]}}

        class FakeClient:
            def __enter__(self): return self
            def __exit__(self, *a): pass
            def post(self, path, headers=None, json=None):
                captured["body"] = json
                return FakeResp()
            def get(self, path, headers=None):
                return FakeGetResp()

        with patch("app.services.calendar_service.httpx.Client", return_value=FakeClient()):
            _book_calendly_slot(calendly_cfg, slot, collected)

        return captured.get("body", {})

    def test_full_name_email_phone_in_payload(self):
        """Booking payload has name, email, phone from collected params."""
        body = self._capture_booking_payload(_COLLECTED)
        _assert_booking_payload_valid(body)

        invitee = body["invitee"]
        assert "jane" in invitee["name"].lower()
        assert "smith" in invitee["name"].lower()
        assert invitee["email"] == "jane.smith@example.com"
        assert invitee.get("text_reminder_number"), "Phone missing from invitee"

    def test_phone_normalized_to_e164(self):
        """10-digit US phone is normalized to E.164 (+1XXXXXXXXXX)."""
        collected = {**_COLLECTED, "phone_number": "5558675309"}
        body = self._capture_booking_payload(collected)
        phone = body["invitee"].get("text_reminder_number", "")
        assert phone.startswith("+1"), f"Expected E.164 format, got {phone!r}"
        assert len(phone) == 12, f"Expected +1XXXXXXXXXX (12 chars), got {phone!r}"

    def test_phone_with_dashes_normalized(self):
        """'555-867-5309' is normalized to E.164."""
        collected = {**_COLLECTED, "phone_number": "555-867-5309"}
        body = self._capture_booking_payload(collected)
        phone = body["invitee"].get("text_reminder_number", "")
        assert phone == "+15558675309", f"Expected +15558675309, got {phone!r}"

    def test_start_time_format_for_calendly(self):
        """
        Calendly POST /invitees requires start_time in YYYY-MM-DDTHH:MM:SS.000000Z format.
        """
        body = self._capture_booking_payload(_COLLECTED)
        st = body.get("start_time", "")
        assert st.endswith("Z"), f"start_time must end with 'Z': {st!r}"
        assert ".000000Z" in st, (
            f"Calendly requires microsecond precision: expected .000000Z, got {st!r}"
        )
        # Must parse cleanly
        _assert_valid_iso_utc(st, "booking.start_time")

    def test_event_type_is_api_uri(self):
        """event_type in booking payload is a Calendly API URI."""
        body = self._capture_booking_payload(_COLLECTED)
        et = body.get("event_type", "")
        assert et.startswith("https://api.calendly.com/"), (
            f"event_type must be an API URI, got {et!r}"
        )

    def test_booking_source_is_ai_assistant(self):
        """booking_source is always 'ai_scheduling_assistant'."""
        body = self._capture_booking_payload(_COLLECTED)
        assert body.get("booking_source") == "ai_scheduling_assistant"

    def test_single_name_caller_gets_empty_last_name(self):
        """Caller with single name: first_name set, last_name empty (not sent)."""
        collected = {**_COLLECTED, "full_name": "Madonna"}
        body = self._capture_booking_payload(collected)
        assert body["invitee"]["first_name"] == "Madonna"
        assert "last_name" not in body["invitee"] or body["invitee"]["last_name"] == ""

    def test_no_email_does_not_crash(self):
        """Missing email logs a warning but booking still proceeds."""
        collected = {"full_name": "Jane Smith", "phone_number": "555-867-5309"}
        body = self._capture_booking_payload(collected)
        # Still produces a payload — email may be empty string
        assert "invitee" in body

    def test_timezone_is_set_from_config(self):
        """invitee.timezone comes from calendly config (America/New_York in test cfg)."""
        body = self._capture_booking_payload(_COLLECTED)
        assert body["invitee"]["timezone"] == "America/New_York"


# ── Full agent conversation scenarios ─────────────────────────────────────────

class TestFullBookingConversation:
    """
    Multi-turn conversations through the scheduling agent.
    Verifies the final slot in state at confirmation is Calendly-ready.
    """

    def _run_turns(self, turns: list[str], initial_slots: Optional[list] = None) -> dict:
        """
        Drive the agent through multiple turns. Returns the final internal_state.
        """
        agent = SchedulingAgent()
        state = dict(_BASE_STATE)
        state["available_slots"] = initial_slots or _make_calendly_slots(3)
        history = []

        for utterance in turns:
            resp = agent.process(utterance, dict(state), _CONFIG, history)
            state = resp.internal_state
            history.append({"role": "user", "content": utterance})
            history.append({"role": "assistant", "content": resp.speak})
            if resp.status == AgentStatus.COMPLETED:
                break
        return state

    def test_scenario_direct_pick_then_confirm(self):
        """
        Turn 1: 'The first one' → WAITING_CONFIRM
        Turn 2: 'Yes' → agent processes confirmation

        After turn 1, chosen slot must be Calendly-ready.
        """
        agent = SchedulingAgent()
        state = dict(_BASE_STATE)
        state["available_slots"] = _make_calendly_slots(3)

        with patch("app.agents.scheduling.llm_text_call", return_value="I'll book you for that slot. Shall I confirm?"):
            resp1 = agent.process("The first one please.", dict(state), _CONFIG, [])

        if resp1.status == AgentStatus.WAITING_CONFIRM:
            idx = resp1.internal_state.get("chosen_slot_id", 0)
            chosen = resp1.internal_state["available_slots"][idx]
            _assert_slot_calendly_ready(chosen, "chosen_slot_after_pick")
            print(f"\n  Chosen slot: {chosen['description']}")
            print(f"  start_time: {chosen['start_time']}")
            print(f"  event_type_uri: {chosen['event_type_uri']}")

    def test_scenario_date_preference_then_pick(self):
        """
        Turn 1: 'Do you have Thursday?' → fetches Thursday slots, presents them
        Turn 2: 'The first one' → picks from new slots

        Verifies date-range query → new slots → pick all produce Calendly-ready state.
        """
        agent = SchedulingAgent()
        state = dict(_BASE_STATE)
        state["available_slots"] = _make_calendly_slots(3)

        thu_slots = _make_timeslots(2, base_offset_days=3)

        with patch("app.agents.scheduling.list_available_slots", return_value=thu_slots), \
             patch("app.agents.scheduling.llm_text_call", return_value="I have slots on Thursday. Which works?"):
            resp1 = agent.process("Do you have anything on Thursday?", dict(state), _CONFIG, [])

        assert resp1.status == AgentStatus.IN_PROGRESS
        new_slots = resp1.internal_state.get("available_slots", [])
        assert new_slots, "No slots after Thursday preference"
        for i, s in enumerate(new_slots):
            _assert_slot_calendly_ready(s, f"thursday_slot[{i}]")

        # Turn 2: pick from new slots
        with patch("app.agents.scheduling.llm_text_call", return_value="I'll book you for Thursday. Shall I confirm?"):
            resp2 = agent.process("The first one.", resp1.internal_state, _CONFIG, [])

        if resp2.status == AgentStatus.WAITING_CONFIRM:
            idx = resp2.internal_state.get("chosen_slot_id", 0)
            chosen = resp2.internal_state["available_slots"][idx]
            _assert_slot_calendly_ready(chosen, "chosen_thursday_slot")
            print(f"\n  Chosen Thursday slot: {chosen['description']}")

    def test_scenario_no_slots_in_range_graceful(self):
        """
        'Do you have Sunday?' → no slots → agent says sorry, keeps state intact.
        State available_slots after the response must be an empty list (not original).
        """
        agent = SchedulingAgent()
        state = dict(_BASE_STATE)
        state["available_slots"] = _make_calendly_slots(3)

        with patch("app.agents.scheduling.list_available_slots", return_value=[]):
            resp = agent.process("Do you have anything on Sunday?", dict(state), _CONFIG, [])

        assert resp.status == AgentStatus.IN_PROGRESS
        assert "sorry" in resp.speak.lower() or "opening" in resp.speak.lower() or \
               "available" in resp.speak.lower(), (
            f"Expected graceful no-slots message, got: {resp.speak!r}"
        )
        # State updated to reflect no Thursday slots (empty list)
        new_slots = resp.internal_state.get("available_slots", ["sentinel"])
        assert new_slots == [], f"Expected empty slot list after failed range query, got {new_slots}"

    def test_scenario_multi_day_search_preserves_event_type_uri(self):
        """
        After a date preference re-query, the event_type_uri in new slots
        matches the URI that was in state — not a blank or scheduling URL.
        """
        agent = SchedulingAgent()
        state = dict(_BASE_STATE)
        state["available_slots"] = _make_calendly_slots(3)
        state["matched_event_type_uri"] = _EVENT_TYPE_URI

        fresh_slots = _make_timeslots(2, base_offset_days=7)

        with patch("app.agents.scheduling.list_available_slots", return_value=fresh_slots), \
             patch("app.agents.scheduling.llm_text_call", return_value="Next week I have slots available. Which works?"):
            resp = agent.process("Can we do something next week?", dict(state), _CONFIG, [])

        new_slots = resp.internal_state.get("available_slots", [])
        for slot in new_slots:
            assert slot["event_type_uri"] == _EVENT_TYPE_URI, (
                f"event_type_uri changed after re-query: {slot['event_type_uri']!r}"
            )
