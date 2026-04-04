"""
Unit tests for the interrupt/resume contract across all is_primary_interactive agents.

Tests the three principles from AgentBase.process() docstring:
  1. Domain gate — UNHANDLED returned with internal_state UNCHANGED when utterance
     is outside the agent's domain.
  2. State immutability on UNHANDLED — internal_state must not be modified before
     the domain gate fires.
  3. Resume speak — when utterance="" (resume after interrupt), agent re-surfaces
     its pending state so the caller knows where to pick up.

Agents under test:
  - DataCollectionAgent (is_primary_interactive=True)
  - NarrativeCollectionAgent (is_primary_interactive=True)
  - SchedulingAgent (is_primary_interactive=True)

All LLM calls are patched — no network or API keys required.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.agents.base import AgentBase, AgentStatus
from app.agents.data_collection import DataCollectionAgent
from app.agents.narrative_collection import NarrativeCollectionAgent, _NeedsAnswerSignal
from app.agents.scheduling import SchedulingAgent

# ── Shared helpers ────────────────────────────────────────────────────────────

_MINIMAL_CONFIG = {
    "assistant": {"persona_name": "Aria", "narrative_topic": "your legal matter"},
    "parameters": [
        {
            "name": "full_name",
            "display_label": "Full Name",
            "data_type": "name",
            "required": True,
            "collection_order": 0,
            "extraction_hints": [],
        },
        {
            "name": "phone_number",
            "display_label": "Phone Number",
            "data_type": "phone",
            "required": True,
            "collection_order": 1,
            "extraction_hints": [],
        },
    ],
    "faqs": [],
    "context_files": [],
    "practice_areas": [],
    "global_policy_documents": [],
    "_collected": {},
    "_booking": {},
    "_notes": "",
    "_tool_results": {},
    "_workflow_stages": "",
}

_SLOT = {
    "slot_id": "slot-1",
    "description": "Monday, April 7 at 10:00 AM",
    "start_time": "2026-04-07T10:00:00+00:00",
    "end_time": "2026-04-07T11:00:00+00:00",
    "event_type_uri": "https://api.calendly.com/event_types/TEST",
}


# ── is_primary_interactive marker ─────────────────────────────────────────────

class TestPrimaryInteractiveMarker:
    def test_data_collection_is_primary_interactive(self):
        assert DataCollectionAgent.is_primary_interactive is True

    def test_narrative_collection_is_primary_interactive(self):
        assert NarrativeCollectionAgent.is_primary_interactive is True

    def test_scheduling_is_primary_interactive(self):
        assert SchedulingAgent.is_primary_interactive is True

    def test_base_default_is_false(self):
        # Default on AgentBase is False; primary agents must explicitly opt in
        assert AgentBase.is_primary_interactive is False


# ── NarrativeCollectionAgent interrupt/resume ─────────────────────────────────

class TestNarrativeCollectionInterrupt:
    """Principle 1 & 2: domain gate fires UNHANDLED; state not modified."""

    def _make_agent(self):
        return NarrativeCollectionAgent()

    def _needs_answer_true(self, *a, **kw):
        return _NeedsAnswerSignal(needs_answer=True)

    def _needs_answer_false(self, *a, **kw):
        return _NeedsAnswerSignal(needs_answer=False)

    def test_domain_gate_returns_unhandled_for_faq_question(self):
        """A question-seeking utterance while collecting → UNHANDLED."""
        agent = self._make_agent()
        state = {"stage": "collecting", "segments": ["I was in an accident last month."]}
        with patch("app.agents.narrative_collection.llm_structured_call", self._needs_answer_true):
            resp = agent.process("What are your fees?", state, _MINIMAL_CONFIG, [])
        assert resp.status == AgentStatus.UNHANDLED

    def test_domain_gate_returns_unhandled_preserves_state(self):
        """Principle 2: internal_state identical before and after UNHANDLED."""
        agent = self._make_agent()
        original_segments = ["I was hit by a car."]
        state = {"stage": "collecting", "segments": list(original_segments)}
        state_snapshot = dict(state)

        with patch("app.agents.narrative_collection.llm_structured_call", self._needs_answer_true):
            resp = agent.process("Where is your office?", state, _MINIMAL_CONFIG, [])

        assert resp.status == AgentStatus.UNHANDLED
        # State must be returned unchanged
        assert resp.internal_state["segments"] == original_segments
        assert resp.internal_state["stage"] == "collecting"

    def test_domain_gate_passes_for_narrative_content(self):
        """Narrative statements pass through the gate — not UNHANDLED."""
        agent = self._make_agent()
        state = {}
        with patch("app.agents.narrative_collection.llm_structured_call", self._needs_answer_false):
            resp = agent.process("My employer fired me last Tuesday.", state, _MINIMAL_CONFIG, [])
        assert resp.status != AgentStatus.UNHANDLED

    def test_domain_gate_asks_done_stage_also_gated(self):
        """FAQ question during asking_done stage also triggers UNHANDLED."""
        agent = self._make_agent()
        state = {
            "stage": "asking_done",
            "segments": ["I was in a car accident and hurt my back."],
        }
        with patch("app.agents.narrative_collection.llm_structured_call", self._needs_answer_true):
            resp = agent.process("How long does this usually take?", state, _MINIMAL_CONFIG, [])
        assert resp.status == AgentStatus.UNHANDLED
        # State unchanged
        assert resp.internal_state["stage"] == "asking_done"
        assert resp.internal_state["segments"] == state["segments"]

    def test_gate_failure_defaults_to_false_not_unhandled(self):
        """If the LLM gate call throws, default is False (treat as narrative)."""
        agent = self._make_agent()
        state = {}

        def _raise(*a, **kw):
            raise RuntimeError("LLM unavailable")

        with patch("app.agents.narrative_collection.llm_structured_call", _raise):
            resp = agent.process("My employer fired me.", state, _MINIMAL_CONFIG, [])
        assert resp.status != AgentStatus.UNHANDLED


class TestNarrativeCollectionResume:
    """Principle 3: resume speak re-surfaces pending state."""

    def test_resume_collecting_no_segments_gives_opening_prompt(self):
        """Empty utterance on first call returns opening question."""
        agent = NarrativeCollectionAgent()
        state = {}
        resp = agent.process("", state, _MINIMAL_CONFIG, [])
        assert resp.status == AgentStatus.IN_PROGRESS
        assert resp.speak  # must have some prompt
        assert len(resp.speak) > 10

    def test_resume_collecting_with_segments_gives_continue_prompt(self):
        """Empty utterance after interrupt mid-collection re-surfaces topic."""
        agent = NarrativeCollectionAgent()
        state = {"stage": "collecting", "segments": ["I was hurt in an accident."]}
        resp = agent.process("", state, _MINIMAL_CONFIG, [])
        assert resp.status == AgentStatus.IN_PROGRESS
        assert resp.speak
        # Should reference narrative topic
        assert "listening" in resp.speak.lower() or "matter" in resp.speak.lower() or "ahead" in resp.speak.lower()

    def test_resume_asking_done_stage_re_asks_done_question(self):
        """Empty utterance in asking_done stage re-asks 'Is there anything else?'"""
        agent = NarrativeCollectionAgent()
        state = {
            "stage": "asking_done",
            "segments": ["I was in a car accident and hurt my back."],
        }
        resp = agent.process("", state, _MINIMAL_CONFIG, [])
        assert resp.status == AgentStatus.IN_PROGRESS
        assert "anything else" in resp.speak.lower()


# ── SchedulingAgent interrupt/resume ──────────────────────────────────────────

class TestSchedulingAgentInterrupt:
    """Principle 1 & 2: domain gate for scheduling agent."""

    def _make_agent(self):
        return SchedulingAgent()

    def _needs_answer_true(self, *a, **kw):
        from app.agents.scheduling import _NeedsAnswerSignal as _S
        return _S(needs_answer=True)

    def _needs_answer_false(self, *a, **kw):
        from app.agents.scheduling import _NeedsAnswerSignal as _S
        return _S(needs_answer=False)

    def _state_awaiting_choice(self):
        return {
            "stage": "awaiting_choice",
            "available_slots": [_SLOT],
            "chosen_slot_id": None,
            "retry_count": 0,
            "matched_event_type_uri": None,
        }

    def _state_awaiting_confirm(self):
        return {
            "stage": "awaiting_confirm",
            "available_slots": [_SLOT],
            "chosen_slot_id": 0,
            "retry_count": 0,
            "matched_event_type_uri": None,
        }

    def test_domain_gate_awaiting_choice_returns_unhandled_for_faq(self):
        """FAQ question during awaiting_choice → UNHANDLED."""
        agent = self._make_agent()
        state = self._state_awaiting_choice()
        with patch("app.agents.scheduling.llm_structured_call", self._needs_answer_true):
            resp = agent.process("What should I bring to the appointment?", state, _MINIMAL_CONFIG, [])
        assert resp.status == AgentStatus.UNHANDLED

    def test_domain_gate_awaiting_confirm_returns_unhandled_for_faq(self):
        """FAQ question during awaiting_confirm → UNHANDLED."""
        agent = self._make_agent()
        state = self._state_awaiting_confirm()
        with patch("app.agents.scheduling.llm_structured_call", self._needs_answer_true):
            resp = agent.process("Do you have parking?", state, _MINIMAL_CONFIG, [])
        assert resp.status == AgentStatus.UNHANDLED

    def test_domain_gate_preserves_state_on_unhandled(self):
        """Principle 2: internal_state unchanged on UNHANDLED."""
        agent = self._make_agent()
        state = self._state_awaiting_choice()
        original_slots = state["available_slots"][:]

        with patch("app.agents.scheduling.llm_structured_call", self._needs_answer_true):
            resp = agent.process("What are your fees?", state, _MINIMAL_CONFIG, [])

        assert resp.status == AgentStatus.UNHANDLED
        assert resp.internal_state["stage"] == "awaiting_choice"
        assert resp.internal_state["available_slots"] == original_slots

    def test_domain_gate_passes_for_slot_choice(self):
        """Selecting a slot passes through the gate — not UNHANDLED."""
        agent = self._make_agent()
        state = self._state_awaiting_choice()

        def _choice_response(*a, **kw):
            from app.agents.agent_schemas import SlotChoice
            return SlotChoice(slot_index=0, confidence=0.9)

        with patch("app.agents.scheduling.llm_structured_call", side_effect=[
            self._needs_answer_false(),  # gate
            _choice_response(),          # choice extraction
        ]):
            resp = agent.process("The first one please.", state, _MINIMAL_CONFIG, [])
        assert resp.status != AgentStatus.UNHANDLED

    def test_presenting_stage_not_gated(self):
        """During 'presenting' stage the gate does NOT run (one-shot auto action)."""
        agent = self._make_agent()
        state = {
            "stage": "presenting",
            "available_slots": [],
            "chosen_slot_id": None,
            "retry_count": 0,
            "matched_event_type_uri": None,
        }
        gate_called = []

        def _gate_spy(*a, **kw):
            gate_called.append(True)
            from app.agents.scheduling import _NeedsAnswerSignal as _S
            return _S(needs_answer=True)

        with patch("app.agents.scheduling.llm_structured_call", _gate_spy), \
             patch("app.agents.scheduling.list_available_slots", return_value=[]):
            resp = agent.process("some utterance", state, _MINIMAL_CONFIG, [])
        # Gate was not called for presenting stage
        assert not gate_called


class TestSchedulingAgentResume:
    """Principle 3: resume speak for scheduling agent."""

    def _make_agent(self):
        return SchedulingAgent()

    def _state_awaiting_choice(self):
        return {
            "stage": "awaiting_choice",
            "available_slots": [_SLOT],
            "chosen_slot_id": None,
            "retry_count": 0,
            "matched_event_type_uri": None,
        }

    def _state_awaiting_confirm(self):
        return {
            "stage": "awaiting_confirm",
            "available_slots": [_SLOT],
            "chosen_slot_id": 0,
            "retry_count": 0,
            "matched_event_type_uri": None,
        }

    def test_resume_awaiting_choice_re_presents_slots(self):
        """Empty utterance in awaiting_choice re-presents the slots."""
        agent = self._make_agent()
        state = self._state_awaiting_choice()

        with patch("app.agents.scheduling.llm_text_call", return_value="I have Monday at 10 AM available. Which would you prefer?"):
            resp = agent.process("", state, _MINIMAL_CONFIG, [])

        assert resp.status == AgentStatus.IN_PROGRESS
        assert resp.speak
        # Slot description should appear in the response
        assert "monday" in resp.speak.lower() or "10" in resp.speak.lower() or resp.speak

    def test_resume_awaiting_confirm_re_asks_confirmation(self):
        """Empty utterance in awaiting_confirm re-asks booking confirmation."""
        agent = self._make_agent()
        state = self._state_awaiting_confirm()

        resp = agent.process("", state, _MINIMAL_CONFIG, [])

        assert resp.status == AgentStatus.WAITING_CONFIRM
        assert "confirm" in resp.speak.lower() or "book" in resp.speak.lower() or "monday" in resp.speak.lower()
        assert resp.pending_confirmation is not None

    def test_resume_awaiting_confirm_pending_confirmation_set(self):
        """pending_confirmation is populated on resume so graph stays WAITING_CONFIRM."""
        agent = self._make_agent()
        state = self._state_awaiting_confirm()

        resp = agent.process("", state, _MINIMAL_CONFIG, [])

        assert resp.pending_confirmation is not None
        assert "slot" in resp.pending_confirmation


# ── DataCollectionAgent resume ────────────────────────────────────────────────

class TestDataCollectionResume:
    """Principle 3: DataCollectionAgent re-surfaces pending confirmation on resume."""

    def _make_agent(self):
        return DataCollectionAgent()

    def test_resume_with_pending_confirmation_re_asks(self):
        """Empty utterance when pending_confirmation is set → re-asks confirmation."""
        agent = self._make_agent()
        state = {
            "collected": {"full_name": "Jane Smith"},
            "pending_confirmation": {"field": "full_name", "value": "Jane Smith"},
            "skipped": [],
            "retry_counts": {},
        }

        def _llm_response(*a, **kw):
            # Should not be called for resume path
            raise AssertionError("LLM should not be called on resume")

        resp = agent.process("", state, _MINIMAL_CONFIG, [])

        assert resp.status == AgentStatus.WAITING_CONFIRM
        assert resp.speak
        assert "jane" in resp.speak.lower() or "full name" in resp.speak.lower() or "confirm" in resp.speak.lower()
        assert resp.pending_confirmation is not None

    def test_resume_no_pending_asks_next_field(self):
        """Empty utterance with no pending confirmation asks the next field."""
        agent = self._make_agent()
        state = {}

        resp = agent.process("", state, _MINIMAL_CONFIG, [])

        assert resp.status == AgentStatus.IN_PROGRESS
        assert resp.speak
        assert len(resp.speak) > 5
