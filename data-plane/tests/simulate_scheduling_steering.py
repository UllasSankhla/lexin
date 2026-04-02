"""
Simulation: steering handlers in the scheduling agent.

Tests the two steering guards:

  1. _slot_choice_bounds_guard
     — LLM returns an out-of-bounds slot index (e.g. slot 5 when only 3 exist)
     — Handler fires, retries with correction, agent recovers gracefully

  2. _booking_preflight
     — Slot data is incomplete (missing slot_id or start_time)
     — Slot description mismatch (chosen_slot_id points to different slot
       than what was presented for confirmation)
     — Handler blocks book_time_slot, agent re-presents slots instead

Both handlers are pure Python — no LLM — and protect irreversible or
structurally-wrong operations before they fire.

Run:
    cd data-plane
    PYTHONPATH=. python3 -m pytest tests/simulate_scheduling_steering.py -v -s
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock

import pytest

from app.agents.scheduling import SchedulingAgent
from app.agents.base import AgentStatus

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(name)s — %(message)s")

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_slots(n: int = 3) -> list[dict]:
    base = datetime(2026, 4, 7, 9, 0, tzinfo=timezone.utc)
    return [
        {
            "slot_id": f"slot-{i}",
            "description": f"{'Monday' if i==0 else 'Tuesday' if i==1 else 'Wednesday'} at {9+i} AM",
            "start_time": (base + timedelta(days=i, hours=i)).isoformat(),
            "end_time":   (base + timedelta(days=i, hours=i+1)).isoformat(),
            "event_type_uri": f"https://api.calendly.com/event_types/fake-{i}",
        }
        for i in range(n)
    ]


CONFIG = {
    "assistant": {"persona_name": "Aria"},
    "parameters": [],
    "calendly": {"api_token": "fake-token", "user_uri": "fake-uri"},
    "_collected": {"full_name": "Test Caller", "email_address": "test@example.com"},
}


# ── _slot_choice_bounds_guard ─────────────────────────────────────────────────

class TestSlotChoiceBoundsGuard:

    def test_returns_none_for_valid_index(self):
        """Guard passes silently for valid indices."""
        agent = SchedulingAgent()
        slots = _make_slots(3)

        assert agent._slot_choice_bounds_guard(0, slots) is None
        assert agent._slot_choice_bounds_guard(1, slots) is None
        assert agent._slot_choice_bounds_guard(2, slots) is None

    def test_fires_for_out_of_bounds_high(self):
        """Guard returns correction string when index exceeds list length."""
        agent = SchedulingAgent()
        slots = _make_slots(3)

        result = agent._slot_choice_bounds_guard(4, slots)  # 1-based: slot 5
        assert result is not None, "Expected correction string for idx=4 with 3 slots"
        assert "5" in result or "out of range" in result.lower()
        print(f"\n  bounds guard (idx=4, len=3): {result!r}")

    def test_fires_for_negative_index(self):
        """Guard catches negative indices."""
        agent = SchedulingAgent()
        slots = _make_slots(3)

        result = agent._slot_choice_bounds_guard(-1, slots)
        assert result is not None
        print(f"\n  bounds guard (idx=-1): {result!r}")

    def test_fires_for_empty_slot_list(self):
        """Guard catches attempt to pick from an empty list."""
        agent = SchedulingAgent()

        result = agent._slot_choice_bounds_guard(0, [])
        assert result is not None
        assert "empty" in result.lower() or "no slots" in result.lower()
        print(f"\n  bounds guard (empty list): {result!r}")

    def test_agent_recovers_after_bounds_violation(self):
        """
        Full agent turn: LLM initially returns out-of-bounds slot index.
        Steering fires, retries, agent should either pick a valid slot
        or re-present the available slots (not crash or book wrong slot).
        """
        agent = SchedulingAgent()
        slots = _make_slots(3)
        state = {
            "stage": "awaiting_choice",
            "available_slots": slots,
            "retry_count": 0,
            "llm_history": [],
        }

        # First SlotChoice call returns slot=5 (out of bounds for 3 slots)
        # Second call (steered retry) returns slot=1 (valid)
        call_count = {"n": 0}
        original_llm = __import__(
            "app.agents.llm_utils", fromlist=["llm_structured_call"]
        ).llm_structured_call

        def fake_llm(system, user_msg, model, **kwargs):
            from app.agents.agent_schemas import SlotChoice, DateRangePreference
            if model is SlotChoice:
                call_count["n"] += 1
                if call_count["n"] == 1:
                    # First call: out of bounds
                    return SlotChoice(slot=5)
                else:
                    # Steered retry: valid slot
                    return SlotChoice(slot=1)
            if model is DateRangePreference:
                return DateRangePreference(found=False)
            return original_llm(system, user_msg, model, **kwargs)

        with patch("app.agents.scheduling.llm_structured_call", side_effect=fake_llm), \
             patch("app.agents.scheduling.llm_text_call", return_value="I'll book you for Monday at 9 AM. Shall I confirm?"):

            resp = agent.process("The first one", state, CONFIG, [])

        print(f"\n  Recovery response: {resp.speak!r}  status={resp.status.value}")
        print(f"  SlotChoice calls made: {call_count['n']}")

        # Agent should not crash and should have moved to awaiting_confirm
        # OR re-presented slots if steered retry also failed
        assert resp.status in (AgentStatus.WAITING_CONFIRM, AgentStatus.IN_PROGRESS), (
            f"Expected WAITING_CONFIRM or IN_PROGRESS after bounds recovery, got {resp.status.value}"
        )
        # Must not have chosen out-of-bounds slot
        chosen_idx = resp.internal_state.get("chosen_slot_id")
        if chosen_idx is not None:
            assert 0 <= chosen_idx < len(slots), (
                f"chosen_slot_id={chosen_idx} is out of bounds for {len(slots)} slots"
            )


# ── _booking_preflight ────────────────────────────────────────────────────────

class TestBookingPreflight:

    def test_returns_none_for_valid_slot(self):
        """Preflight passes for a complete, consistent slot."""
        agent = SchedulingAgent()
        slots = _make_slots(1)
        state = {
            "available_slots": slots,
            "chosen_slot_id": 0,
            "pending_confirmation": {"slot": slots[0]["description"]},
        }
        result = agent._booking_preflight(slots[0], state)
        assert result is None, f"Expected None for valid slot, got: {result!r}"

    def test_fires_for_none_slot(self):
        """Preflight blocks when no slot is selected."""
        agent = SchedulingAgent()
        result = agent._booking_preflight(None, {})
        assert result is not None
        assert "no slot" in result.lower() or "nothing" in result.lower()
        print(f"\n  preflight (None slot): {result!r}")

    def test_fires_for_missing_slot_id(self):
        """Preflight blocks when slot_id is absent."""
        agent = SchedulingAgent()
        bad_slot = {
            "slot_id": "",  # empty
            "description": "Monday at 9 AM",
            "start_time": "2026-04-07T09:00:00+00:00",
            "end_time":   "2026-04-07T10:00:00+00:00",
        }
        result = agent._booking_preflight(bad_slot, {})
        assert result is not None
        assert "slot_id" in result
        print(f"\n  preflight (missing slot_id): {result!r}")

    def test_fires_for_missing_start_time(self):
        """Preflight blocks when start_time is absent."""
        agent = SchedulingAgent()
        bad_slot = {
            "slot_id": "valid-id",
            "description": "Monday at 9 AM",
            "start_time": "",   # missing
            "end_time":   "2026-04-07T10:00:00+00:00",
        }
        result = agent._booking_preflight(bad_slot, {})
        assert result is not None
        assert "start_time" in result
        print(f"\n  preflight (missing start_time): {result!r}")

    def test_fires_for_unparseable_start_time(self):
        """Preflight blocks when start_time is not a valid ISO datetime."""
        agent = SchedulingAgent()
        bad_slot = {
            "slot_id": "valid-id",
            "description": "Monday at 9 AM",
            "start_time": "not-a-date",
            "end_time":   "2026-04-07T10:00:00+00:00",
        }
        result = agent._booking_preflight(bad_slot, {})
        assert result is not None
        assert "not a valid" in result.lower() or "iso" in result.lower()
        print(f"\n  preflight (bad start_time): {result!r}")

    def test_fires_for_description_mismatch(self):
        """
        Preflight blocks when the slot being booked doesn't match
        what was presented to the caller in pending_confirmation.
        This catches LLM mis-indexing the slot list.
        """
        agent = SchedulingAgent()
        slots = _make_slots(3)
        chosen = slots[2]  # "Wednesday at 11 AM"
        state = {
            "pending_confirmation": {"slot": "Monday at 9 AM"},  # caller confirmed slot 0
        }
        result = agent._booking_preflight(chosen, state)
        assert result is not None, (
            "Expected preflight to block mismatch between confirmed and chosen slot"
        )
        assert "mismatch" in result.lower() or "monday" in result.lower() or "wednesday" in result.lower()
        print(f"\n  preflight (description mismatch): {result!r}")

    def test_agent_blocks_booking_on_preflight_failure(self):
        """
        Full agent turn: caller confirms a slot but state has a description
        mismatch (LLM mis-indexed). Preflight fires, agent re-presents slots
        instead of calling book_time_slot.
        """
        agent = SchedulingAgent()
        slots = _make_slots(3)
        state = {
            "stage": "awaiting_confirm",
            "available_slots": slots,
            "chosen_slot_id": 2,   # points to Wednesday at 11 AM
            "retry_count": 0,
            "llm_history": [],
            # But caller was presented and confirmed Monday at 9 AM — MISMATCH
            "pending_confirmation": {"slot": "Monday at 9 AM"},
        }

        book_called = {"called": False}

        def fake_book(*args, **kwargs):
            book_called["called"] = True
            return {"booking_id": "should-not-happen"}

        with patch("app.agents.scheduling._detect_confirmation", return_value="confirm"), \
             patch("app.agents.scheduling.book_time_slot", side_effect=fake_book), \
             patch("app.agents.scheduling.llm_text_call", return_value="Here are the available times again."):

            resp = agent.process("Yes, that works", state, CONFIG, [])

        print(f"\n  Preflight-blocked response: {resp.speak!r}  status={resp.status.value}")

        assert not book_called["called"], (
            "book_time_slot must NOT be called when preflight detects a slot mismatch"
        )
        assert resp.status == AgentStatus.IN_PROGRESS, (
            f"Expected IN_PROGRESS (re-presenting slots), got {resp.status.value}"
        )
        assert resp.internal_state.get("stage") == "awaiting_choice", (
            f"Stage should reset to awaiting_choice, got {resp.internal_state.get('stage')!r}"
        )

    def test_booking_proceeds_when_preflight_passes(self):
        """
        Full agent turn: valid slot, matching description, caller confirms.
        Preflight passes, book_time_slot is called, agent completes.
        """
        agent = SchedulingAgent()
        slots = _make_slots(1)
        state = {
            "stage": "awaiting_confirm",
            "available_slots": slots,
            "chosen_slot_id": 0,
            "retry_count": 0,
            "llm_history": [],
            "pending_confirmation": {"slot": slots[0]["description"]},
        }

        book_called = {"called": False}

        def fake_book(*args, **kwargs):
            book_called["called"] = True
            return {"booking_id": "confirmed-123"}

        with patch("app.agents.scheduling._detect_confirmation", return_value="confirm"), \
             patch("app.agents.scheduling.book_time_slot", side_effect=fake_book):

            resp = agent.process("Yes, please book it", state, CONFIG, [])

        print(f"\n  Successful booking response: {resp.speak!r}  status={resp.status.value}")

        assert book_called["called"], "book_time_slot must be called when preflight passes"
        assert resp.status == AgentStatus.COMPLETED, (
            f"Expected COMPLETED after successful booking, got {resp.status.value}"
        )
