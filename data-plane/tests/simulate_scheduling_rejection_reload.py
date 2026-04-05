"""
Simulation: Calendly slot reload after caller rejection.

Scenarios tested:
  1. reject_then_new_date
     — Caller confirms a slot, rejects confirmation, then requests a different
       week. Calendly must be called to fetch the new slots.

  2. unclear_then_new_date
     — Caller is in awaiting_choice, says "none of these work" (unclear intent),
       then says "how about next week?". Calendly must be called for the new date.

  3. reject_confirm_re_presents_same_slots
     — After rejecting confirmation (no new date given), the agent re-presents the
       same (cached) slots. No Calendly call expected on the rejection turn itself.

Run:
    cd data-plane
    PYTHONPATH=. python3 -m pytest tests/simulate_scheduling_rejection_reload.py -v -s
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock, call

import pytest

from app.agents.scheduling import SchedulingAgent, _SlotAction, _NeedsAnswerSignal
from app.agents.base import AgentStatus

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(name)s — %(message)s")

# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_slot_dicts(n: int = 3, base_day_offset: int = 1) -> list[dict]:
    base = datetime(2026, 4, 7, 9, 0, tzinfo=timezone.utc) + timedelta(days=base_day_offset)
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    return [
        {
            "slot_id": f"slot-{base_day_offset}-{i}",
            "description": f"{days[i % 5]} at {9 + i} AM",
            "start_time": (base + timedelta(days=i, hours=i)).isoformat(),
            "end_time":   (base + timedelta(days=i, hours=i + 1)).isoformat(),
            "event_type_uri": "https://api.calendly.com/event_types/fake",
        }
        for i in range(n)
    ]


def _make_time_slot_objects(slot_dicts: list[dict]):
    """Convert slot dicts to TimeSlot objects (as returned by list_available_slots)."""
    from app.services.calendar_service import TimeSlot
    return [
        TimeSlot(
            slot_id=s["slot_id"],
            description=s["description"],
            start=datetime.fromisoformat(s["start_time"]),
            end=datetime.fromisoformat(s["end_time"]),
            event_type_uri=s["event_type_uri"],
        )
        for s in slot_dicts
    ]


CONFIG = {
    "assistant": {"persona_name": "Aria"},
    "parameters": [],
    "calendly": {"api_token": "fake-token", "user_uri": "fake-uri"},
    "_collected": {"full_name": "Test Caller", "email_address": "test@example.com"},
}

INITIAL_SLOTS = _make_slot_dicts(3, base_day_offset=1)
NEXT_WEEK_SLOTS = _make_slot_dicts(3, base_day_offset=8)


# ── Scenario 1: reject confirmation, then ask for next week ──────────────────

class TestRejectThenNewDate:
    """
    Flow:
      Turn 1 (awaiting_choice): caller picks slot 1 → awaiting_confirm
      Turn 2 (awaiting_confirm): caller says "no" → reject → re-presents same slots
      Turn 3 (awaiting_choice): caller says "how about next week?" → Calendly fetched
    """

    def test_calendly_called_after_reject_then_new_date(self):
        agent = SchedulingAgent()
        state = {
            "stage": "awaiting_choice",
            "available_slots": INITIAL_SLOTS,
            "retry_count": 0,
            "matched_event_type_uri": "https://api.calendly.com/event_types/fake",
            "llm_history": [],
        }

        # ── Turn 1: caller picks slot 1 ──────────────────────────────────────
        with patch("app.agents.scheduling.llm_structured_call") as mock_llm, \
             patch("app.agents.scheduling.llm_text_call", return_value="I'll book you for Monday at 9 AM. Shall I confirm?"):

            mock_llm.side_effect = [
                _NeedsAnswerSignal(needs_answer=False),
                _SlotAction(action="pick", slot_index=1),
            ]
            resp1 = agent.process("I'll take the first one", state, CONFIG, [])
            state = resp1.internal_state

        print(f"\nTurn 1 — status={resp1.status.value} speak={resp1.speak!r}")
        assert resp1.status == AgentStatus.WAITING_CONFIRM
        assert state["stage"] == "awaiting_confirm"

        # ── Turn 2: caller rejects confirmation ───────────────────────────────
        calendly_calls_turn2 = []

        with patch("app.agents.scheduling.list_available_slots",
                   side_effect=lambda *a, **kw: calendly_calls_turn2.append(kw) or []) as mock_cal, \
             patch("app.agents.scheduling._detect_confirmation", return_value="reject"), \
             patch("app.agents.scheduling.llm_text_call", return_value="Here are the times again."):

            resp2 = agent.process("Actually, no", state, CONFIG, [])
            state = resp2.internal_state

        print(f"Turn 2 — status={resp2.status.value} speak={resp2.speak!r}")
        print(f"  Calendly calls on rejection turn: {len(calendly_calls_turn2)}")

        assert resp2.status == AgentStatus.IN_PROGRESS
        assert state["stage"] == "awaiting_choice"
        # Calendly should NOT be called on the rejection turn itself —
        # the agent re-presents the cached slot list
        assert len(calendly_calls_turn2) == 0, (
            f"Expected 0 Calendly calls on rejection turn, got {len(calendly_calls_turn2)}"
        )
        # Same slots must still be in state (not cleared)
        assert len(state["available_slots"]) == len(INITIAL_SLOTS), (
            f"available_slots should be unchanged after reject, got {len(state['available_slots'])}"
        )

        # ── Turn 3: caller asks for next week → Calendly MUST be called ──────
        calendly_calls_turn3 = []
        next_week_objects = _make_time_slot_objects(NEXT_WEEK_SLOTS)

        def fake_calendly(*args, **kwargs):
            calendly_calls_turn3.append(kwargs)
            return next_week_objects

        with patch("app.agents.scheduling.list_available_slots", side_effect=fake_calendly), \
             patch("app.agents.scheduling.llm_structured_call") as mock_llm, \
             patch("app.agents.scheduling.llm_text_call", return_value="Next week I have Monday, Tuesday, Wednesday."):

            mock_llm.side_effect = [
                _NeedsAnswerSignal(needs_answer=False),
                _SlotAction(
                    action="new_date",
                    start_time="2026-04-13T00:00:00Z",
                    end_time="2026-04-20T00:00:00Z",
                ),
            ]
            resp3 = agent.process("How about next week?", state, CONFIG, [])
            state = resp3.internal_state

        print(f"Turn 3 — status={resp3.status.value} speak={resp3.speak!r}")
        print(f"  Calendly calls on new_date turn: {len(calendly_calls_turn3)}")

        assert len(calendly_calls_turn3) == 1, (
            f"Expected exactly 1 Calendly call after new_date request, got {len(calendly_calls_turn3)}"
        )
        assert resp3.status == AgentStatus.IN_PROGRESS
        # State should now hold the new slots
        assert len(state["available_slots"]) == len(NEXT_WEEK_SLOTS), (
            f"available_slots should be updated to next-week slots, got {len(state['available_slots'])}"
        )
        assert state["available_slots"][0]["slot_id"].startswith("slot-8-"), (
            f"Slots should be next-week slots, got {state['available_slots'][0]['slot_id']!r}"
        )


# ── Scenario 2: unclear then new_date ────────────────────────────────────────

class TestUnclearThenNewDate:
    """
    Flow:
      State: awaiting_choice with initial slots loaded
      Turn 1: caller says "none of these work" → action=unclear → same slots re-presented
      Turn 2: caller says "do you have anything Friday?" → action=new_date → Calendly fetched
    """

    def test_calendly_called_after_unclear_then_new_date(self):
        agent = SchedulingAgent()
        state = {
            "stage": "awaiting_choice",
            "available_slots": INITIAL_SLOTS,
            "retry_count": 0,
            "matched_event_type_uri": "https://api.calendly.com/event_types/fake",
            "llm_history": [],
        }

        # ── Turn 1: unclear intent ────────────────────────────────────────────
        calendly_calls_turn1 = []

        with patch("app.agents.scheduling.list_available_slots",
                   side_effect=lambda *a, **kw: calendly_calls_turn1.append(kw) or []), \
             patch("app.agents.scheduling.llm_structured_call") as mock_llm, \
             patch("app.agents.scheduling.llm_text_call", return_value="Here are the available times."):

            mock_llm.side_effect = [
                _NeedsAnswerSignal(needs_answer=False),
                _SlotAction(action="unclear"),
            ]
            resp1 = agent.process("None of these work for me", state, CONFIG, [])
            state = resp1.internal_state

        print(f"\nTurn 1 (unclear) — status={resp1.status.value} speak={resp1.speak!r}")
        print(f"  Calendly calls: {len(calendly_calls_turn1)}")

        assert resp1.status == AgentStatus.IN_PROGRESS
        assert state["stage"] == "awaiting_choice"
        assert len(calendly_calls_turn1) == 0, (
            f"Expected 0 Calendly calls on unclear turn, got {len(calendly_calls_turn1)}"
        )
        # Slots must still be populated — agent must not clear them
        assert len(state["available_slots"]) > 0, (
            "available_slots should not be cleared after unclear intent"
        )

        # ── Turn 2: specific day request → Calendly MUST be called ───────────
        calendly_calls_turn2 = []
        friday_objects = _make_time_slot_objects(_make_slot_dicts(2, base_day_offset=4))

        def fake_calendly(*args, **kwargs):
            calendly_calls_turn2.append(kwargs)
            return friday_objects

        with patch("app.agents.scheduling.list_available_slots", side_effect=fake_calendly), \
             patch("app.agents.scheduling.llm_structured_call") as mock_llm, \
             patch("app.agents.scheduling.llm_text_call", return_value="Friday I have 9 AM and 10 AM."):

            mock_llm.side_effect = [
                _NeedsAnswerSignal(needs_answer=False),
                _SlotAction(
                    action="new_date",
                    start_time="2026-04-10T00:00:00Z",
                    end_time="2026-04-10T23:59:00Z",
                ),
            ]
            resp2 = agent.process("Do you have anything on Friday?", state, CONFIG, [])
            state = resp2.internal_state

        print(f"Turn 2 (new_date) — status={resp2.status.value} speak={resp2.speak!r}")
        print(f"  Calendly calls: {len(calendly_calls_turn2)}")

        assert len(calendly_calls_turn2) == 1, (
            f"Expected 1 Calendly call after Friday request, got {len(calendly_calls_turn2)}"
        )
        assert resp2.status == AgentStatus.IN_PROGRESS
        assert len(state["available_slots"]) == len(friday_objects), (
            f"available_slots should be updated to Friday slots after Calendly fetch"
        )


# ── Scenario 3: new_slot from confirm stage ───────────────────────────────────

class TestNewSlotFromConfirm:
    """
    Flow:
      State: awaiting_confirm with slot 0 chosen
      Turn: caller says "actually do you have Thursday instead?" → new_slot intent
            → _handle_choice called → action=new_date → Calendly fetched
    """

    def test_calendly_called_on_new_slot_from_confirm(self):
        agent = SchedulingAgent()
        state = {
            "stage": "awaiting_confirm",
            "available_slots": INITIAL_SLOTS,
            "chosen_slot_id": 0,
            "retry_count": 0,
            "matched_event_type_uri": "https://api.calendly.com/event_types/fake",
            "llm_history": [],
            "pending_confirmation": {"slot": INITIAL_SLOTS[0]["description"]},
        }

        calendly_calls = []
        thursday_objects = _make_time_slot_objects(_make_slot_dicts(2, base_day_offset=3))

        def fake_calendly(*args, **kwargs):
            calendly_calls.append(kwargs)
            return thursday_objects

        with patch("app.agents.scheduling.list_available_slots", side_effect=fake_calendly), \
             patch("app.agents.scheduling._detect_confirmation", return_value="new_slot"), \
             patch("app.agents.scheduling.llm_structured_call") as mock_llm, \
             patch("app.agents.scheduling.llm_text_call", return_value="Thursday I have 9 AM and 10 AM."):

            mock_llm.side_effect = [
                _NeedsAnswerSignal(needs_answer=False),
                _SlotAction(
                    action="new_date",
                    start_time="2026-04-09T00:00:00Z",
                    end_time="2026-04-09T23:59:00Z",
                ),
            ]
            resp = agent.process("Actually, do you have anything Thursday?", state, CONFIG, [])

        print(f"\nNew-slot from confirm — status={resp.status.value} speak={resp.speak!r}")
        print(f"  Calendly calls: {len(calendly_calls)}")

        assert len(calendly_calls) == 1, (
            f"Expected 1 Calendly call after new_slot→new_date, got {len(calendly_calls)}"
        )
        assert resp.status == AgentStatus.IN_PROGRESS
        assert len(resp.internal_state["available_slots"]) == len(thursday_objects)


# ── Scenario 4: event_type_uri preserved from prefetched slots ────────────────

class TestPrefetchedUriPreserved:
    """
    Regression for: _present_slots used to set matched_event_type_uri=None when
    using prefetched slots, causing post-rejection Calendly calls to re-discover
    the event type (extra API call, failure falls back to dummy slots).

    The fix: extract event_type_uri from the first prefetched TimeSlot.
    """

    def test_event_type_uri_extracted_from_prefetched_slots(self):
        """
        When _present_slots uses prefetched slots, matched_event_type_uri must
        be set to the event_type_uri on those slots — not None.
        """
        from app.services.calendar_service import TimeSlot

        agent = SchedulingAgent()
        prefetched_uri = "https://api.calendly.com/event_types/discovered-123"
        prefetched_objects = [
            TimeSlot(
                slot_id="pre-0",
                description="Monday at 9 AM",
                start=datetime(2026, 4, 7, 9, 0, tzinfo=timezone.utc),
                end=datetime(2026, 4, 7, 10, 0, tzinfo=timezone.utc),
                event_type_uri=prefetched_uri,
            ),
        ]

        config_with_prefetch = {
            **CONFIG,
            "_tool_results": {"prefetched_slots": prefetched_objects},
        }
        state = {"stage": "presenting", "retry_count": 0, "llm_history": []}

        with patch("app.agents.scheduling.llm_text_call", return_value="I have Monday at 9 AM."):
            resp = agent.process("", state, config_with_prefetch, [])

        print(f"\nPrefetch URI preservation:")
        print(f"  matched_event_type_uri = {resp.internal_state.get('matched_event_type_uri')!r}")

        assert resp.internal_state.get("matched_event_type_uri") == prefetched_uri, (
            f"matched_event_type_uri should be {prefetched_uri!r}, "
            f"got {resp.internal_state.get('matched_event_type_uri')!r}"
        )

    def test_new_date_after_prefetch_uses_correct_event_type_and_date_range(self):
        """
        After initial prefetch presentation, a post-rejection new_date request
        must pass BOTH the preserved event_type_uri AND the LLM-extracted date
        range to list_available_slots.

        Regression for two bugs:
          1. event_type_uri was None (fixed above)
          2. date range must come from the LLM's _SlotAction, not a hardcoded window
        """
        agent = SchedulingAgent()
        prefetched_uri = "https://api.calendly.com/event_types/discovered-123"

        # LLM will return this date range for "next week"
        llm_start = "2026-04-13T00:00:00Z"
        llm_end   = "2026-04-20T00:00:00Z"
        expected_start = datetime(2026, 4, 13, 0, 0, 0, tzinfo=timezone.utc)
        expected_end   = datetime(2026, 4, 20, 0, 0, 0, tzinfo=timezone.utc)

        state = {
            "stage": "awaiting_choice",
            "available_slots": INITIAL_SLOTS,
            "retry_count": 0,
            "matched_event_type_uri": prefetched_uri,
            "llm_history": [],
        }

        received = []

        def fake_calendly(purpose, config, event_type_uri, search_start=None, search_end=None):
            received.append({
                "event_type_uri": event_type_uri,
                "search_start": search_start,
                "search_end": search_end,
            })
            return _make_time_slot_objects(_make_slot_dicts(2, base_day_offset=7))

        with patch("app.agents.scheduling.list_available_slots", side_effect=fake_calendly), \
             patch("app.agents.scheduling.llm_structured_call") as mock_llm, \
             patch("app.agents.scheduling.llm_text_call", return_value="Next week: Monday and Tuesday."):

            mock_llm.side_effect = [
                _NeedsAnswerSignal(needs_answer=False),
                _SlotAction(action="new_date", start_time=llm_start, end_time=llm_end),
            ]
            resp = agent.process("How about next week?", state, CONFIG, [])

        print(f"\nCalendly call args: {received}")

        assert len(received) == 1, f"Expected 1 Calendly call, got {len(received)}"

        call = received[0]
        assert call["event_type_uri"] == prefetched_uri, (
            f"Wrong event_type_uri: expected {prefetched_uri!r}, got {call['event_type_uri']!r}"
        )
        assert call["search_start"] == expected_start, (
            f"Wrong search_start: expected {expected_start}, got {call['search_start']}"
        )
        assert call["search_end"] == expected_end, (
            f"Wrong search_end: expected {expected_end}, got {call['search_end']}"
        )


# ── Scenario 5: two sequential date rejections ───────────────────────────────

class TestTwoSequentialDateRejections:
    """
    Full multi-turn flow with two successive date-change requests:

      Turn 1 (awaiting_choice): caller says "how about next week?"
             → LLM returns new_date with 7-day window
             → Calendly call #1 with that range
             → New slots stored in state

      Turn 2 (awaiting_choice): caller says "actually, how about next Thursday?"
             → LLM returns new_date with single-day range for Thursday
             → Calendly call #2 with that (different) range
             → Slots updated again

    Validates:
      - Each Calendly call receives the date range extracted from THAT turn's LLM response
      - The event_type_uri is preserved across both calls (not reset between turns)
      - available_slots is updated after each call
    """

    def test_two_sequential_new_date_requests(self):
        agent = SchedulingAgent()
        prefetched_uri = "https://api.calendly.com/event_types/discovered-123"

        state = {
            "stage": "awaiting_choice",
            "available_slots": INITIAL_SLOTS,
            "retry_count": 0,
            "matched_event_type_uri": prefetched_uri,
            "llm_history": [],
        }

        # Expected date ranges from the LLM for each turn
        # Turn 1: "next week" → 7-day window starting Monday Apr 13
        next_week_start = "2026-04-13T00:00:00Z"
        next_week_end   = "2026-04-20T00:00:00Z"

        # Turn 2: "next Thursday" → single-day window for Apr 16
        thursday_start = "2026-04-16T00:00:00Z"
        thursday_end   = "2026-04-16T23:59:00Z"

        next_week_slots  = _make_time_slot_objects(_make_slot_dicts(3, base_day_offset=7))
        thursday_slots   = _make_time_slot_objects(_make_slot_dicts(2, base_day_offset=10))

        calendly_calls: list[dict] = []

        def fake_calendly(purpose, config, event_type_uri, search_start=None, search_end=None):
            calendly_calls.append({
                "event_type_uri": event_type_uri,
                "search_start": search_start,
                "search_end": search_end,
            })
            # Return next_week_slots on first call, thursday_slots on second
            return next_week_slots if len(calendly_calls) == 1 else thursday_slots

        # ── Turn 1: "how about next week?" ────────────────────────────────────
        with patch("app.agents.scheduling.list_available_slots", side_effect=fake_calendly), \
             patch("app.agents.scheduling.llm_structured_call") as mock_llm, \
             patch("app.agents.scheduling.llm_text_call", return_value="Next week I have Monday, Tuesday, Wednesday."):

            mock_llm.side_effect = [
                _NeedsAnswerSignal(needs_answer=False),
                _SlotAction(action="new_date", start_time=next_week_start, end_time=next_week_end),
            ]
            resp1 = agent.process("How about next week?", state, CONFIG, [])
            state = resp1.internal_state

        print(f"\nTurn 1 (next week) — status={resp1.status.value} speak={resp1.speak!r}")
        print(f"  Calendly call 1: uri={calendly_calls[0]['event_type_uri'].rsplit('/', 1)[-1]!r}"
              f"  start={calendly_calls[0]['search_start']}  end={calendly_calls[0]['search_end']}")

        assert len(calendly_calls) == 1, f"Expected 1 Calendly call after turn 1, got {len(calendly_calls)}"
        assert calendly_calls[0]["event_type_uri"] == prefetched_uri
        assert calendly_calls[0]["search_start"] == datetime(2026, 4, 13, tzinfo=timezone.utc)
        assert calendly_calls[0]["search_end"]   == datetime(2026, 4, 20, tzinfo=timezone.utc)
        assert resp1.status == AgentStatus.IN_PROGRESS
        # State now holds next-week slots
        assert len(state["available_slots"]) == len(next_week_slots)
        # URI must still be preserved after the date-change fetch
        assert state["matched_event_type_uri"] == prefetched_uri, (
            f"matched_event_type_uri should be preserved after new_date fetch, "
            f"got {state['matched_event_type_uri']!r}"
        )

        # ── Turn 2: "actually, how about next Thursday?" ──────────────────────
        with patch("app.agents.scheduling.list_available_slots", side_effect=fake_calendly), \
             patch("app.agents.scheduling.llm_structured_call") as mock_llm, \
             patch("app.agents.scheduling.llm_text_call", return_value="Thursday I have 9 AM and 10 AM."):

            mock_llm.side_effect = [
                _NeedsAnswerSignal(needs_answer=False),
                _SlotAction(action="new_date", start_time=thursday_start, end_time=thursday_end),
            ]
            resp2 = agent.process("Actually, how about next Thursday?", state, CONFIG, [])
            state = resp2.internal_state

        print(f"Turn 2 (Thursday)  — status={resp2.status.value} speak={resp2.speak!r}")
        print(f"  Calendly call 2: uri={calendly_calls[1]['event_type_uri'].rsplit('/', 1)[-1]!r}"
              f"  start={calendly_calls[1]['search_start']}  end={calendly_calls[1]['search_end']}")

        assert len(calendly_calls) == 2, f"Expected 2 total Calendly calls after turn 2, got {len(calendly_calls)}"

        # Call 2 must use the Thursday range — not next_week range repeated
        assert calendly_calls[1]["event_type_uri"] == prefetched_uri
        assert calendly_calls[1]["search_start"] == datetime(2026, 4, 16, tzinfo=timezone.utc), (
            f"Turn 2 search_start should be Thursday Apr 16, got {calendly_calls[1]['search_start']}"
        )
        assert calendly_calls[1]["search_end"] == datetime(2026, 4, 16, 23, 59, tzinfo=timezone.utc), (
            f"Turn 2 search_end should be Apr 16 23:59, got {calendly_calls[1]['search_end']}"
        )
        assert resp2.status == AgentStatus.IN_PROGRESS
        # State now holds Thursday slots
        assert len(state["available_slots"]) == len(thursday_slots), (
            f"available_slots should be updated to Thursday slots after second fetch, "
            f"got {len(state['available_slots'])}"
        )
