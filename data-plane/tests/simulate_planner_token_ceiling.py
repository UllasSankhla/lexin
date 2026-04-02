"""
Simulation: Planner LLM token ceiling causes empty-content failure.

Reproduces the bug observed in the Sanjay Sankhla transcript where:
  - The planner is called with max_tokens=1024 (before the fix)
  - On turn 7 ("Uh, no. That's the one I gave you."), the LLM hit the ceiling
    and Cerebras returned empty content → fallback plan fired
  - data_collection was invoked without pending_confirmation context
  - LLM generated "Perfect, I have what I need" speak but email was never stored

Tests:
  1. Planner falls back gracefully when llm_structured_call raises empty-content
     ValueError — the fallback plan must invoke data_collection and not crash.
  2. After the fix (max_tokens=2048), verify planner.py uses 2048 not 1024.
  3. Split-brain guard fires when LLM returns status=completed but required
     fields are missing from Python collected state.

Run:
    cd data-plane
    PYTHONPATH=. python3 -m pytest tests/simulate_planner_token_ceiling.py -v -s
"""
from __future__ import annotations

import copy
import logging
from unittest.mock import patch, MagicMock

import pytest

from app.agents.base import AgentStatus
from app.agents.planner import Planner
from app.agents.workflow import WorkflowGraph
from app.agents.data_collection import DataCollectionAgent
from app.agents.data_collection_schema import DataCollectionLLMResponse

logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname)-8s %(name)s — %(message)s",
)

# ── Shared config (name + email, like the Sanjay scenario) ────────────────────

CONFIG = {
    "assistant": {"persona_name": "Aria"},
    "parameters": [
        {
            "name": "first_name",
            "display_label": "First Name",
            "data_type": "name",
            "required": True,
            "collection_order": 1,
            "extraction_hints": [],
        },
        {
            "name": "last_name",
            "display_label": "Last Name",
            "data_type": "name",
            "required": True,
            "collection_order": 2,
            "extraction_hints": [],
        },
        {
            "name": "email_address",
            "display_label": "Email Address",
            "data_type": "email",
            "required": True,
            "collection_order": 3,
            "extraction_hints": [],
        },
    ],
}

# Graph seeded: data_collection WAITING_CONFIRM for email, first/last confirmed
def _seeded_graph():
    from app.agents.graph_config import APPOINTMENT_BOOKING
    graph = WorkflowGraph(APPOINTMENT_BOOKING)
    dc_state = graph.states["data_collection"]
    dc_state.status = AgentStatus.WAITING_CONFIRM
    dc_state.internal_state = {
        "collected": {"first_name": "Sanjay", "last_name": "Sankhla"},
        "pending_confirmation": {"field": "email_address", "value": "u.s@c.com"},
        "extraction_queue": [],
        "retry_count": 0,
        "history": [
            {"role": "assistant", "content": "I have your email as u dot s at c dot com — is that correct?"},
        ],
    }
    return graph


# ── History for planner context ───────────────────────────────────────────────

_HISTORY = [
    {"role": "assistant", "content": "Could I get your First Name, please?"},
    {"role": "user",      "content": "Yeah. My first name is Sanjay."},
    {"role": "assistant", "content": "I have Sanjay — is that correct?"},
    {"role": "user",      "content": "Yeah. That's correct."},
    {"role": "assistant", "content": "Great, and could I get your Last Name, please?"},
    {"role": "user",      "content": "My last name is Sankhla."},
    {"role": "assistant", "content": "I have Sankhla — is that correct?"},
    {"role": "user",      "content": "Yes. And my email address is u dot s at c dot com."},
    {"role": "assistant", "content": "I have your email as u dot s at c dot com — is that correct?"},
    {"role": "user",      "content": "No. You still"},   # false reject turn
    {"role": "assistant", "content": "No problem. Could you give me your Email Address again?"},
    {"role": "user",      "content": "Uh, no. That's the one I gave you."},  # ← the failing turn
]


# ── Test 1: Planner fallback on empty-content ValueError ─────────────────────

class TestPlannerFallbackOnEmptyContent:

    def test_fallback_plan_invokes_data_collection(self):
        """
        When llm_structured_call raises ValueError('LLM returned empty content'),
        the planner must fall back to a single-step plan targeting data_collection,
        not crash or return an empty plan.
        """
        graph = _seeded_graph()
        planner = Planner(graph)

        with patch(
            "app.agents.planner.llm_structured_call",
            side_effect=ValueError("LLM returned empty content"),
        ):
            steps = planner.plan("Uh, no. That's the one I gave you.", _HISTORY)

        print(f"\n  Fallback steps: {[(s.action, s.agent_id) for s in steps]}")

        invoke_steps = [s for s in steps if s.action == "invoke"]
        assert invoke_steps, "Expected at least one invoke step from fallback plan"
        assert invoke_steps[0].agent_id == "data_collection", (
            f"Expected fallback to target data_collection, got {invoke_steps[0].agent_id}"
        )

    def test_fallback_does_not_crash_with_no_pending_in_state(self):
        """
        Even if pending_confirmation was already cleared from dc_state before
        the planner is called (as in the Sanjay bug), the fallback must still
        produce a valid plan.
        """
        graph = _seeded_graph()
        # Simulate that pending was cleared by the false reject
        graph.states["data_collection"].internal_state["pending_confirmation"] = None
        graph.states["data_collection"].status = AgentStatus.IN_PROGRESS
        planner = Planner(graph)

        with patch(
            "app.agents.planner.llm_structured_call",
            side_effect=ValueError("LLM returned empty content"),
        ):
            steps = planner.plan("Uh, no. That's the one I gave you.", _HISTORY)

        assert steps, "Fallback must return a non-empty plan"
        print(f"\n  Fallback (no pending) steps: {[(s.action, s.agent_id) for s in steps]}")


# ── Test 2: Verify planner uses max_tokens=2048 after the fix ────────────────

class TestPlannerMaxTokens:

    def test_planner_calls_llm_with_2048_tokens(self):
        """
        Verify that the planner passes max_tokens=2048 to llm_structured_call.
        This locks in the fix for the token-ceiling bug.
        """
        graph = _seeded_graph()
        planner = Planner(graph)
        captured = {}

        def _capture_llm(system, user_msg, model, **kwargs):
            captured["max_tokens"] = kwargs.get("max_tokens")
            # Return a minimal valid response so the planner completes
            from app.agents.agent_schemas import MultiIntentLLMResponse, IntentItem
            return MultiIntentLLMResponse(
                thinking="caller is confirming the email",
                intents=[IntentItem(type="CONFIRMATION", field=None, reason="yes/no pending")],
            )

        with patch("app.agents.planner.llm_structured_call", side_effect=_capture_llm):
            planner.plan("Uh, no. That's the one I gave you.", _HISTORY)

        assert captured.get("max_tokens") == 2048, (
            f"Planner must use max_tokens=2048, got {captured.get('max_tokens')}. "
            f"This was the root cause of the Sanjay Sankhla empty-content bug."
        )
        print(f"\n  Planner max_tokens: {captured.get('max_tokens')} ✓")


# ── Test 3: Split-brain guard fires in data_collection ────────────────────────

class TestSplitBrainGuard:

    def test_guard_fires_when_llm_says_completed_but_email_missing(self):
        """
        DataCollectionAgent must NOT return COMPLETED when the LLM says
        status=completed but email_address is missing from Python collected state.

        The split-brain guard in Step E must:
        - Log ERROR
        - Return IN_PROGRESS with a re-ask speak
        """
        agent = DataCollectionAgent()
        state = {
            "collected": {"first_name": "Sanjay", "last_name": "Sankhla"},
            # email NOT in collected — this is the split-brain state
            "pending_confirmation": None,
            "extraction_queue": [],
            "retry_count": 0,
            "history": list(_HISTORY),
        }
        history = list(_HISTORY)

        # LLM returns status=completed even though email is missing
        fake_result = DataCollectionLLMResponse(
            intent="answer",
            extracted=[],
            correction_value=None,
            speak="Perfect, I have what I need for now.",
            status="completed",
            pending_confirmation=None,
            incomplete_utterance=False,
            cannot_process=False,
            cannot_process_reason=None,
        )

        with patch(
            "app.agents.data_collection.llm_structured_call",
            return_value=fake_result,
        ):
            resp = agent.process(
                "one I gave you is the right one.",
                state,
                CONFIG,
                history,
            )

        print(f"\n  Guard result: status={resp.status.value}  speak={resp.speak!r}")

        assert resp.status != AgentStatus.COMPLETED, (
            "BUG: Agent returned COMPLETED even though email_address is missing "
            "from Python collected state. Split-brain guard did not fire."
        )
        assert resp.status == AgentStatus.IN_PROGRESS, (
            f"Expected IN_PROGRESS after guard fires, got {resp.status.value}"
        )
        # Speak must ask for the missing field, not say "I have everything"
        speak_lower = resp.speak.lower()
        completion_phrases = ("have everything", "have what i need", "have all", "have the information")
        assert not any(p in speak_lower for p in completion_phrases), (
            f"BUG: Agent spoke a completion phrase after guard fired: {resp.speak!r}"
        )

    def test_guard_does_not_fire_when_all_fields_collected(self):
        """
        Guard must NOT fire when all required fields are genuinely in collected.
        Confirms no false positives.
        """
        agent = DataCollectionAgent()
        state = {
            "collected": {
                "first_name": "Sanjay",
                "last_name": "Sankhla",
                "email_address": "u.s@c.com",
            },
            "pending_confirmation": None,
            "extraction_queue": [],
            "retry_count": 0,
            "history": [],
        }

        fake_result = DataCollectionLLMResponse(
            intent="confirm_yes",
            extracted=[],
            correction_value=None,
            speak="Perfect, I have what I need for now.",
            status="completed",
            pending_confirmation=None,
            incomplete_utterance=False,
            cannot_process=False,
            cannot_process_reason=None,
        )

        with patch(
            "app.agents.data_collection.llm_structured_call",
            return_value=fake_result,
        ):
            resp = agent.process("yes that's right", state, CONFIG, [])

        print(f"\n  No-guard result: status={resp.status.value}  speak={resp.speak!r}")

        assert resp.status == AgentStatus.COMPLETED, (
            f"Expected COMPLETED when all fields collected, got {resp.status.value}"
        )
