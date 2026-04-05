"""
Simulation: scheduling agent handling natural-language date preference utterances.

Tests that the scheduling agent correctly handles:
  1. Question-form date preferences: "Do you have any dates available on Thursday?"
  2. Statement-form date preferences: "Can we do something next week?"
  3. Time-of-day preferences: "Do you have anything in the afternoon?"
  4. Direct slot picks: "The first one please."

Root cause of the original bug: _handle_choice ran two sequential LLM calls:
  1. SlotChoice — returns null (not picking from current list)
  2. DateRangePreference — failed on question-form and time-of-day utterances
     because the prompt asked to "convert a time preference", not handle questions
  Falls through to retry → re-presents original slots without acknowledging request.

Fix: Single _SlotAction LLM call that classifies intent as:
  "pick" → selecting a specific slot (returns slot_index)
  "new_date" → wants a different day/time (returns start_time, end_time)
  "unclear" → cannot determine

Run:
    cd data-plane
    PYTHONPATH=. python3 -m pytest tests/simulate_scheduling_date_preference.py -v -s
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock, call as mock_call

import pytest

from app.agents.scheduling import SchedulingAgent
from app.agents.base import AgentStatus

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(name)s — %(message)s")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _monday_slots() -> list[dict]:
    """Initial slots — all on Monday/Tuesday, no Thursday."""
    base = datetime(2026, 4, 6, 10, 0, tzinfo=timezone.utc)  # Monday
    return [
        {
            "slot_id": "slot-mon-1",
            "description": "Monday, April 6 at 10:00 AM",
            "start_time": base.isoformat(),
            "end_time": (base + timedelta(hours=1)).isoformat(),
            "event_type_uri": "https://api.calendly.com/event_types/fake",
        },
        {
            "slot_id": "slot-tue-1",
            "description": "Tuesday, April 7 at 2:00 PM",
            "start_time": (base + timedelta(days=1, hours=4)).isoformat(),
            "end_time": (base + timedelta(days=1, hours=5)).isoformat(),
            "event_type_uri": "https://api.calendly.com/event_types/fake",
        },
    ]


def _thursday_slots() -> list:
    """Slots returned when Thursday is queried."""
    from app.services.calendar_service import TimeSlot
    base = datetime(2026, 4, 10, 11, 0, tzinfo=timezone.utc)  # Thursday
    return [
        TimeSlot(
            slot_id="slot-thu-1",
            description="Thursday, April 10 at 11:00 AM",
            start=base,
            end=base + timedelta(hours=1),
            event_type_uri="https://api.calendly.com/event_types/fake",
            spots_remaining=1,
        )
    ]


CONFIG = {
    "assistant": {"persona_name": "Aria"},
    "parameters": [],
    "calendly": {"api_token": "fake-token", "user_uri": "fake-uri"},
    "_collected": {"full_name": "Jane Smith"},
    "_tool_results": {},
}

_AWAITING_CHOICE_STATE = {
    "stage": "awaiting_choice",
    "available_slots": _monday_slots(),
    "chosen_slot_id": None,
    "retry_count": 0,
    "matched_event_type_uri": None,
}


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestDatePreferenceQuestionForm:
    """
    Caller asks "Do you have any dates available on Thursday?" —
    question-form utterance that should trigger a new calendar query.
    """

    def test_thursday_question_triggers_calendar_query(self):
        """
        Bug reproduction: question-form date preference must trigger
        list_available_slots, not fall through to re-presenting original slots.
        """
        agent = SchedulingAgent()
        state = dict(_AWAITING_CHOICE_STATE)
        state["available_slots"] = _monday_slots()

        calendar_called_with = []

        def _fake_list_slots(purpose, config, event_type_uri=None, search_start=None, search_end=None):
            calendar_called_with.append((search_start, search_end))
            return _thursday_slots()

        with patch("app.agents.scheduling.list_available_slots", _fake_list_slots), \
             patch("app.agents.scheduling.llm_text_call", return_value="I have Thursday, April 10 at 11 AM. Would that work?"):
            resp = agent.process(
                "Do you have any dates available on Thursday?",
                state, CONFIG, [],
            )

        assert calendar_called_with, (
            "list_available_slots was NOT called — date preference was not extracted. "
            "The agent likely fell through to retry/re-present original slots."
        )
        # Should have queried a Thursday range
        start, end = calendar_called_with[0]
        assert start is not None
        # Thursday is April 10 (today=April 4, so next Thursday)
        assert start.weekday() == 3, f"Expected Thursday (weekday=3), got weekday={start.weekday()} ({start})"

        assert resp.status == AgentStatus.IN_PROGRESS
        assert resp.speak
        # Response should reference Thursday or April 10
        assert "thursday" in resp.speak.lower() or "april 10" in resp.speak.lower() or "11" in resp.speak.lower()

    def test_thursday_question_new_slots_presented(self):
        """When Thursday has slots, those slots (not original slots) are presented."""
        agent = SchedulingAgent()
        state = dict(_AWAITING_CHOICE_STATE)
        state["available_slots"] = _monday_slots()
        original_slot_ids = {s["slot_id"] for s in _monday_slots()}

        def _fake_list_slots(*a, **kw):
            return _thursday_slots()

        with patch("app.agents.scheduling.list_available_slots", _fake_list_slots), \
             patch("app.agents.scheduling.llm_text_call", return_value="I have Thursday at 11 AM. Which would you prefer?"):
            resp = agent.process(
                "Do you have any dates available on Thursday?",
                state, CONFIG, [],
            )

        # available_slots in returned state should contain Thursday slot
        new_slots = resp.internal_state.get("available_slots", [])
        new_slot_ids = {s["slot_id"] for s in new_slots}
        assert new_slot_ids != original_slot_ids, (
            f"Slots were not updated — still showing original: {new_slot_ids}"
        )
        assert "slot-thu-1" in new_slot_ids

    def test_thursday_no_slots_gives_helpful_message(self):
        """When no slots found on Thursday, response is helpful, not a loop."""
        agent = SchedulingAgent()
        state = dict(_AWAITING_CHOICE_STATE)
        state["available_slots"] = _monday_slots()

        def _fake_list_slots(*a, **kw):
            return []  # No Thursday slots

        with patch("app.agents.scheduling.list_available_slots", _fake_list_slots):
            resp = agent.process(
                "Do you have any dates available on Thursday?",
                state, CONFIG, [],
            )

        # Must not be COMPLETED or FAILED
        assert resp.status == AgentStatus.IN_PROGRESS
        # Response should NOT just re-present the original Monday/Tuesday slots
        # without acknowledging the Thursday request
        speak = resp.speak.lower()
        assert speak  # must say something
        # Should not silently loop — must acknowledge no Thursday availability
        # (either mentions no openings, or asks for alternative, not just "Monday..." again)
        assert "opening" in speak or "available" in speak or "different" in speak or \
               "thursday" in speak or "try" in speak or "range" in speak, (
            f"Response doesn't acknowledge Thursday query: {resp.speak!r}"
        )


class TestStatementFormDatePreference:
    """
    Caller states a time preference directly (not question form).
    These should already work but are included as regression guards.
    """

    def test_next_week_preference(self):
        """'I'd prefer next week' triggers a new calendar query."""
        agent = SchedulingAgent()
        state = dict(_AWAITING_CHOICE_STATE)
        state["available_slots"] = _monday_slots()

        calendar_called = []

        def _fake_list_slots(*a, **kw):
            calendar_called.append(True)
            return _thursday_slots()

        with patch("app.agents.scheduling.list_available_slots", _fake_list_slots), \
             patch("app.agents.scheduling.llm_text_call", return_value="Next week I have Thursday at 11 AM. Does that work?"):
            resp = agent.process(
                "Can we do something next week instead?",
                state, CONFIG, [],
            )

        assert calendar_called, "list_available_slots not called for 'next week' preference"
        assert resp.status == AgentStatus.IN_PROGRESS

    def test_afternoon_preference(self):
        """'Do you have anything in the afternoon?' triggers a new query."""
        agent = SchedulingAgent()
        state = dict(_AWAITING_CHOICE_STATE)
        state["available_slots"] = _monday_slots()

        calendar_called = []

        def _fake_list_slots(*a, **kw):
            calendar_called.append(True)
            from app.services.calendar_service import TimeSlot
            base = datetime(2026, 4, 7, 14, 0, tzinfo=timezone.utc)
            return [TimeSlot(
                slot_id="slot-pm-1",
                description="Tuesday, April 7 at 2:00 PM",
                start=base, end=base + timedelta(hours=1),
                event_type_uri="https://api.calendly.com/event_types/fake",
                spots_remaining=1,
            )]

        with patch("app.agents.scheduling.list_available_slots", _fake_list_slots), \
             patch("app.agents.scheduling.llm_text_call", return_value="I have Tuesday at 2 PM. Shall I book that?"):
            resp = agent.process(
                "Do you have anything in the afternoon?",
                state, CONFIG, [],
            )

        assert calendar_called, "list_available_slots not called for afternoon preference"


class TestSlotPickStillWorks:
    """
    After the fix, direct slot selection must still work correctly.
    Uses the real LLM — tests that the _SlotAction prompt correctly classifies
    direct slot picks.
    """

    def test_pick_by_ordinal(self):
        """'The first one' selects a slot and moves to WAITING_CONFIRM."""
        agent = SchedulingAgent()
        state = dict(_AWAITING_CHOICE_STATE)
        state["available_slots"] = _monday_slots()

        with patch("app.agents.scheduling.llm_text_call", return_value="I'll book you for Monday at 10 AM. Shall I confirm?"):
            resp = agent.process("The first one please.", state, CONFIG, [])

        assert resp.status == AgentStatus.WAITING_CONFIRM, (
            f"Expected WAITING_CONFIRM for 'The first one please', got {resp.status.value}: {resp.speak!r}"
        )
        assert resp.pending_confirmation is not None
        chosen_idx = resp.internal_state.get("chosen_slot_id")
        assert chosen_idx == 0, f"Expected slot index 0, got {chosen_idx}"

    def test_pick_by_day_name(self):
        """'Monday works for me' selects the Monday slot."""
        agent = SchedulingAgent()
        state = dict(_AWAITING_CHOICE_STATE)
        state["available_slots"] = _monday_slots()

        with patch("app.agents.scheduling.llm_text_call", return_value="I'll book you for Monday at 10 AM. Shall I confirm?"):
            resp = agent.process("Monday works for me.", state, CONFIG, [])

        assert resp.status == AgentStatus.WAITING_CONFIRM, (
            f"Expected WAITING_CONFIRM for 'Monday works for me', got {resp.status.value}: {resp.speak!r}"
        )
        chosen_idx = resp.internal_state.get("chosen_slot_id")
        assert chosen_idx == 0, f"Expected Monday slot (index 0), got {chosen_idx}"
