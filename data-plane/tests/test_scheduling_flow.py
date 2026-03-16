"""
Unit tests for the multi-round scheduling conversation flow.

Covers:
  - Caller picks a slot directly from the initial list
  - Caller declines initial slots and asks for a different time (natural language)
  - LLM translates natural-language preference to a date range
  - New slots are fetched and presented for the preferred range
  - Caller picks from the new list and confirms
  - Booking is created with all collected data (name, email, description)
  - Confirm message includes cancel/reschedule email note
  - Edge cases: no slots in initial window, no slots in preferred range,
    repeated preference changes, max-retry fallback to first slot

All external dependencies (Calendly API, Cerebras LLM) are replaced with
lightweight fakes so the tests run without network access or API keys.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Optional
from unittest.mock import MagicMock, patch, call

import pytest

from app.pipeline.booking_workflow import BookingWorkflow, WorkflowStage, ScheduleStage
from app.pipeline.parameter_collector import CollectionState, ParameterSpec
from app.services.calendar_service import TimeSlot


# ── Fixtures & helpers ────────────────────────────────────────────────────────

def _make_slot(n: int, offset_days: int = 1) -> TimeSlot:
    """Create a deterministic TimeSlot for test use."""
    base = datetime.now(timezone.utc).replace(
        hour=10, minute=0, second=0, microsecond=0
    ) + timedelta(days=offset_days + n - 1)
    return TimeSlot(
        slot_id=f"slot-{n}",
        start=base,
        end=base + timedelta(hours=1),
        description=f"Option {n}: {base.strftime('%A, %B %-d at 10:00 AM')}",
        event_type_uri="https://api.calendly.com/event_types/TEST",
        spots_remaining=1,
    )


def _make_collected(**kwargs) -> dict:
    """Shorthand for building the collected-params dict."""
    defaults = {
        "client_full_name": "Jane Smith",
        "email": "jane@example.com",
        "case_description": "Contract review",
    }
    defaults.update(kwargs)
    return defaults


def _make_collection_state(collected: dict | None = None) -> CollectionState:
    """CollectionState with no pending fields (all already collected)."""
    cs = CollectionState(parameters=[])
    cs.collected = collected or _make_collected()
    return cs


def _make_booking_result(slot: TimeSlot) -> dict:
    return {
        "booking_id": "BK-TEST-001",
        "invitee_uri": "https://api.calendly.com/scheduled_events/abc/invitees/xyz",
        "status": "active",
        "event_uri": "https://api.calendly.com/scheduled_events/abc",
        "cancel_url": "https://calendly.com/cancellations/xyz",
        "reschedule_url": "https://calendly.com/reschedulings/xyz",
        "confirmed_at": datetime.now(timezone.utc).isoformat(),
    }


def _make_workflow(
    slots_by_call: list[list[TimeSlot]] | None = None,
    collected: dict | None = None,
    config: dict | None = None,
) -> tuple[BookingWorkflow, MagicMock, MagicMock]:
    """
    Build a BookingWorkflow with:
      - a fake LLMToolkit (mock)
      - list_available_slots patched to return successive slot lists per call
      - book_time_slot patched

    Returns (workflow, llm_mock, book_mock).
    """
    initial_slots = [_make_slot(1), _make_slot(2), _make_slot(3)]
    if slots_by_call is None:
        slots_by_call = [initial_slots]

    llm = MagicMock()
    # Sensible defaults for LLM methods
    llm.match_event_type.return_value = None
    llm.present_slots.side_effect = lambda slots: (
        "Here are the available times: " + ", ".join(s.description for s in slots)
    )
    llm.extract_slot_choice.return_value = None  # default: no match
    llm.extract_date_preference.return_value = None  # default: no date pref
    llm.confirm_slot.side_effect = lambda slot: f"Shall I book {slot.description}?"
    llm.confirm_booking.return_value = (
        "Your appointment is confirmed! A confirmation email is on its way. Goodbye!"
    )

    cfg = config or {"assistant": {"persona_description": "consultation"}}
    cs = _make_collection_state(collected)

    wf = BookingWorkflow(
        collection_state=cs,
        llm_toolkit=llm,
        config=cfg,
    )

    # Patch list_available_slots to pop from slots_by_call on each invocation
    call_iter = iter(slots_by_call)

    def _fake_list_slots(purpose, config, event_type_uri=None, search_start=None, search_end=None):
        try:
            return next(call_iter)
        except StopIteration:
            return []

    book_mock = MagicMock(side_effect=lambda slot, collected, config: _make_booking_result(slot))

    return wf, llm, book_mock, _fake_list_slots


# ── 1. Caller picks a slot directly from initial list ─────────────────────────

def test_direct_slot_pick_and_confirm():
    """
    Happy path: caller picks the second slot from the initial list without
    asking for alternative dates.
    """
    slots = [_make_slot(1), _make_slot(2), _make_slot(3)]
    wf, llm, book_mock, fake_list = _make_workflow(slots_by_call=[slots])
    llm.extract_slot_choice.return_value = slots[1]  # caller picks slot 2

    with patch("app.pipeline.booking_workflow.list_available_slots", fake_list), \
         patch("app.pipeline.booking_workflow.book_time_slot", book_mock):

        opening = wf.get_opening()
        assert "Option 1" in opening or "Option 2" in opening  # slots presented
        assert wf.stage == WorkflowStage.SCHEDULING
        assert wf.schedule_stage == ScheduleStage.AWAITING_CHOICE

        # Caller chooses slot 2
        result = wf.process_utterance("I'll take the second one")
        assert wf.schedule_stage == ScheduleStage.AWAITING_CONFIRM
        assert slots[1].description in result.speak
        assert result.booking_details is None  # not booked yet

        # Caller confirms
        result = wf.process_utterance("Yes, that works")
        assert result.call_complete is True
        assert result.booking_details is not None
        assert result.booking_details["booking_id"] == "BK-TEST-001"
        book_mock.assert_called_once_with(slots[1], wf.collection_state.collected, wf.config)


# ── 2. Caller declines initial slots and requests different dates ──────────────

def test_caller_requests_different_dates():
    """
    Caller says 'none of these work, what about next Tuesday morning'.
    Workflow fetches a new window and presents those slots.
    Caller then picks from the new list.
    """
    initial_slots = [_make_slot(1), _make_slot(2)]
    next_week_slots = [_make_slot(10, offset_days=8), _make_slot(11, offset_days=8)]

    date_pref = {
        "start_time": "2026-03-17T08:00:00Z",
        "end_time":   "2026-03-17T12:00:00Z",
    }

    wf, llm, book_mock, fake_list = _make_workflow(
        slots_by_call=[initial_slots, next_week_slots]
    )
    llm.extract_slot_choice.return_value = None  # first utterance: no slot matched
    llm.extract_date_preference.return_value = date_pref

    with patch("app.pipeline.booking_workflow.list_available_slots", fake_list), \
         patch("app.pipeline.booking_workflow.book_time_slot", book_mock):

        wf.get_opening()  # presents initial_slots

        # Caller asks for a different time
        result = wf.process_utterance("None of those work, how about next Tuesday morning?")
        assert wf.schedule_stage == ScheduleStage.AWAITING_CHOICE  # still choosing
        assert result.booking_details is None
        # New slots presented
        assert "Option 10" in result.speak or "Option 11" in result.speak

        # Verify LLM was asked to extract a date preference
        llm.extract_date_preference.assert_called_once()

        # Now caller picks from new list
        llm.extract_slot_choice.return_value = next_week_slots[0]
        llm.extract_date_preference.return_value = None
        result = wf.process_utterance("The first one please")
        assert wf.schedule_stage == ScheduleStage.AWAITING_CONFIRM

        # Confirm
        result = wf.process_utterance("Yes")
        assert result.call_complete is True
        book_mock.assert_called_once_with(next_week_slots[0], wf.collection_state.collected, wf.config)


# ── 3. extract_date_preference produces correct UTC range ─────────────────────

def test_extract_date_preference_parses_json(monkeypatch):
    """
    LLMToolkit.extract_date_preference must parse the LLM's JSON response and
    return a dict with start_time / end_time.
    """
    from app.pipeline.llm import LLMToolkit, LLMClient

    fake_client = MagicMock(spec=LLMClient)
    fake_client.complete.return_value = (
        '{"start_time": "2026-03-17T08:00:00Z", "end_time": "2026-03-17T12:00:00Z"}',
        20,
        50.0,
    )

    toolkit = LLMToolkit(fake_client)
    result = toolkit.extract_date_preference("next Tuesday morning", "2026-03-13")

    assert result is not None
    assert result["start_time"] == "2026-03-17T08:00:00Z"
    assert result["end_time"] == "2026-03-17T12:00:00Z"


def test_extract_date_preference_returns_none_for_none_response():
    from app.pipeline.llm import LLMToolkit, LLMClient

    fake_client = MagicMock(spec=LLMClient)
    fake_client.complete.return_value = ("NONE", 5, 30.0)

    toolkit = LLMToolkit(fake_client)
    result = toolkit.extract_date_preference("I don't know", "2026-03-13")
    assert result is None


def test_extract_date_preference_handles_markdown_fencing():
    """LLM sometimes wraps JSON in ```json ... ``` — strip it."""
    from app.pipeline.llm import LLMToolkit, LLMClient

    fake_client = MagicMock(spec=LLMClient)
    fake_client.complete.return_value = (
        '```json\n{"start_time": "2026-03-20T00:00:00Z", "end_time": "2026-03-20T23:59:00Z"}\n```',
        25,
        40.0,
    )
    toolkit = LLMToolkit(fake_client)
    result = toolkit.extract_date_preference("next Thursday", "2026-03-13")
    assert result is not None
    assert result["start_time"] == "2026-03-20T00:00:00Z"


# ── 4. No slots in initial window — immediate preference prompt ───────────────

def test_no_initial_slots_asks_preference():
    """
    When the next 7 days have no openings, the workflow should immediately
    ask the caller for their preferred time rather than saying 'no slots'.
    """
    wf, llm, book_mock, fake_list = _make_workflow(slots_by_call=[[]])  # empty initial

    with patch("app.pipeline.booking_workflow.list_available_slots", fake_list), \
         patch("app.pipeline.booking_workflow.book_time_slot", book_mock):

        opening = wf.get_opening()
        assert wf.stage == WorkflowStage.SCHEDULING
        assert wf.schedule_stage == ScheduleStage.AWAITING_CHOICE
        # Should ask for preference, not say "no slots available"
        opening_lower = opening.lower()
        assert any(word in opening_lower for word in ("when", "prefer", "work", "time"))
        # Should NOT have called present_slots (no slots to present)
        llm.present_slots.assert_not_called()


# ── 5. No slots in preferred range — asks for another preference ──────────────

def test_no_slots_in_preferred_range_asks_again():
    """
    Caller asks for 'next Monday' but there are no openings.
    Workflow should say so and prompt for another range — NOT end the call.
    """
    date_pref = {"start_time": "2026-03-16T00:00:00Z", "end_time": "2026-03-16T23:59:00Z"}

    wf, llm, book_mock, fake_list = _make_workflow(
        slots_by_call=[[_make_slot(1)], []]  # initial has 1 slot; preferred range is empty
    )
    llm.extract_slot_choice.return_value = None
    llm.extract_date_preference.return_value = date_pref

    with patch("app.pipeline.booking_workflow.list_available_slots", fake_list), \
         patch("app.pipeline.booking_workflow.book_time_slot", book_mock):

        wf.get_opening()

        result = wf.process_utterance("How about next Monday?")
        assert result.call_complete is False
        assert wf.schedule_stage == ScheduleStage.AWAITING_CHOICE
        speak_lower = result.speak.lower()
        assert any(w in speak_lower for w in ("sorry", "no", "openings", "different", "try"))
        book_mock.assert_not_called()


# ── 6. Multiple preference changes before booking ─────────────────────────────

def test_multiple_preference_changes():
    """
    Caller asks for 'next week', then 'the week after', then picks a slot.
    Each change should trigger a new Calendly query and slot presentation.
    """
    slots_wk1 = [_make_slot(8, offset_days=7)]
    slots_wk2 = [_make_slot(15, offset_days=14), _make_slot(16, offset_days=14)]

    pref1 = {"start_time": "2026-03-16T00:00:00Z", "end_time": "2026-03-22T23:59:00Z"}
    pref2 = {"start_time": "2026-03-23T00:00:00Z", "end_time": "2026-03-29T23:59:00Z"}

    wf, llm, book_mock, fake_list = _make_workflow(
        slots_by_call=[[_make_slot(1)], slots_wk1, slots_wk2]
    )

    call_sequence: list[Optional[dict]] = [pref1, pref2, None]
    pref_iter = iter(call_sequence)
    slot_iter = iter([None, None, slots_wk2[0]])

    llm.extract_date_preference.side_effect = lambda *a, **kw: next(pref_iter)
    llm.extract_slot_choice.side_effect = lambda *a, **kw: next(slot_iter)

    with patch("app.pipeline.booking_workflow.list_available_slots", fake_list), \
         patch("app.pipeline.booking_workflow.book_time_slot", book_mock):

        wf.get_opening()  # presents slot 1 (initial)

        result = wf.process_utterance("Can I see next week instead?")
        assert "Option 8" in result.speak
        assert wf.schedule_stage == ScheduleStage.AWAITING_CHOICE

        result = wf.process_utterance("Actually the week after would be better")
        assert "Option 15" in result.speak or "Option 16" in result.speak
        assert wf.schedule_stage == ScheduleStage.AWAITING_CHOICE

        result = wf.process_utterance("The first one")
        assert wf.schedule_stage == ScheduleStage.AWAITING_CONFIRM

        result = wf.process_utterance("yes")
        assert result.call_complete is True
        book_mock.assert_called_once_with(slots_wk2[0], wf.collection_state.collected, wf.config)


# ── 7. Max-retry fallback to first slot ───────────────────────────────────────

def test_max_retry_defaults_to_first_slot():
    """
    After MAX_RETRIES unclear utterances (no slot match, no date preference),
    the workflow should default to the first available slot and ask to confirm.
    """
    slots = [_make_slot(1), _make_slot(2)]
    wf, llm, book_mock, fake_list = _make_workflow(slots_by_call=[slots])
    llm.extract_slot_choice.return_value = None
    llm.extract_date_preference.return_value = None

    with patch("app.pipeline.booking_workflow.list_available_slots", fake_list), \
         patch("app.pipeline.booking_workflow.book_time_slot", book_mock):

        wf.get_opening()

        for _ in range(BookingWorkflow.MAX_RETRIES - 1):
            result = wf.process_utterance("um, I'm not sure")
            assert wf.schedule_stage == ScheduleStage.AWAITING_CHOICE

        # Final retry → defaults to first slot
        result = wf.process_utterance("um, I'm not sure")
        assert wf.schedule_stage == ScheduleStage.AWAITING_CONFIRM
        assert slots[0].description in result.speak

        result = wf.process_utterance("yes")
        assert result.call_complete is True
        book_mock.assert_called_once_with(slots[0], wf.collection_state.collected, wf.config)


# ── 8. Booking includes all collected caller data ─────────────────────────────

def test_booking_includes_all_collected_data():
    """
    book_time_slot must receive the full collected dict including name, email,
    and any other fields captured during the COLLECTING stage.
    """
    collected = {
        "client_full_name": "Alice Nguyen",
        "email": "alice@lawfirm.com",
        "phone": "5551234567",
        "case_description": "Employment dispute",
    }
    slots = [_make_slot(1)]
    wf, llm, book_mock, fake_list = _make_workflow(
        slots_by_call=[slots], collected=collected
    )
    llm.extract_slot_choice.return_value = slots[0]

    with patch("app.pipeline.booking_workflow.list_available_slots", fake_list), \
         patch("app.pipeline.booking_workflow.book_time_slot", book_mock):

        wf.get_opening()
        wf.process_utterance("I'll take option one")  # pick
        wf.process_utterance("yes please")            # confirm

        _, call_kwargs = book_mock.call_args_list[0][0], book_mock.call_args_list[0]
        passed_collected = book_mock.call_args[0][1]   # second positional arg
        assert passed_collected["client_full_name"] == "Alice Nguyen"
        assert passed_collected["email"] == "alice@lawfirm.com"
        assert passed_collected["case_description"] == "Employment dispute"


# ── 9. Confirmation message mentions cancel/reschedule email ─────────────────

def test_confirm_booking_mentions_email_links():
    """
    confirm_booking LLM call should note that a confirmation email with
    cancel/reschedule links has been sent when those URLs are present.
    """
    from app.pipeline.llm import LLMToolkit, LLMClient

    fake_client = MagicMock(spec=LLMClient)
    fake_client.complete.return_value = (
        "Great, Alice! Your appointment is confirmed for Monday, March 16 at 10:00 AM. "
        "Your booking reference is BK-TEST-001. A confirmation email with cancel and "
        "reschedule links has been sent. Have a wonderful day!",
        60,
        80.0,
    )
    toolkit = LLMToolkit(fake_client)
    slot = _make_slot(1)
    booking = _make_booking_result(slot)
    collected = _make_collected()

    result = toolkit.confirm_booking(slot, booking, collected)

    # Verify the prompt included a note about the email links
    prompt_sent = fake_client.complete.call_args[0][0][0]["content"]
    assert "cancel" in prompt_sent.lower()
    assert "reschedule" in prompt_sent.lower()
    assert "email" in prompt_sent.lower()


def test_confirm_booking_no_email_note_when_urls_absent():
    """
    When booking has no cancel/reschedule URLs, the prompt should NOT mention
    a confirmation email.
    """
    from app.pipeline.llm import LLMToolkit, LLMClient

    fake_client = MagicMock(spec=LLMClient)
    fake_client.complete.return_value = ("Confirmed!", 10, 20.0)
    toolkit = LLMToolkit(fake_client)
    slot = _make_slot(1)
    booking = {"booking_id": "BK-DUMMY", "status": "confirmed"}  # no URLs
    collected = _make_collected()

    toolkit.confirm_booking(slot, booking, collected)

    prompt_sent = fake_client.complete.call_args[0][0][0]["content"]
    # The word "email" may still appear as a field key in the caller-details
    # summary, but the explicit cancel/reschedule instruction must be absent.
    assert "cancel" not in prompt_sent.lower()
    assert "reschedule" not in prompt_sent.lower()


# ── 10. calendar_service respects search_start / search_end ──────────────────

def test_list_available_slots_passes_search_window():
    """
    list_available_slots must forward search_start/search_end to
    _list_calendly_slots, which should use them instead of the default
    lookahead window.
    """
    from app.services import calendar_service

    search_start = datetime(2026, 3, 17, 8, 0, 0, tzinfo=timezone.utc)
    search_end   = datetime(2026, 3, 17, 12, 0, 0, tzinfo=timezone.utc)

    captured: dict = {}

    def _fake_calendly_slots(cfg, event_type_uri, ss=None, se=None):
        captured["search_start"] = ss
        captured["search_end"]   = se
        return [_make_slot(99)]

    cfg = {"calendly": {"api_token": "tok", "lookahead_days": 30, "timezone": "UTC"}}

    with patch.object(calendar_service, "_list_calendly_slots", _fake_calendly_slots):
        result = calendar_service.list_available_slots(
            "consultation", cfg,
            event_type_uri="https://api.calendly.com/event_types/TEST",
            search_start=search_start,
            search_end=search_end,
        )

    assert result[0].slot_id == "slot-99"
    assert captured["search_start"] == search_start
    assert captured["search_end"] == search_end


# ── 11. Slot rejection cycles back to choice stage ───────────────────────────

def test_rejecting_slot_returns_to_choice():
    """
    If the caller says 'no' to the confirm-slot question, the workflow
    re-presents available slots and goes back to AWAITING_CHOICE.
    """
    slots = [_make_slot(1), _make_slot(2)]
    wf, llm, book_mock, fake_list = _make_workflow(slots_by_call=[slots])
    llm.extract_slot_choice.return_value = slots[0]

    with patch("app.pipeline.booking_workflow.list_available_slots", fake_list), \
         patch("app.pipeline.booking_workflow.book_time_slot", book_mock):

        wf.get_opening()
        wf.process_utterance("option one")           # pick slot 0
        assert wf.schedule_stage == ScheduleStage.AWAITING_CONFIRM

        result = wf.process_utterance("no, actually")  # reject
        assert wf.schedule_stage == ScheduleStage.AWAITING_CHOICE
        assert result.call_complete is False
        book_mock.assert_not_called()
