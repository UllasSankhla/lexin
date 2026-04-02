"""
Simulation: Sanjay Sankhla — full pipeline (planner + data_collection).

Reproduces the production transcript bug and verifies the fix:
  - "No. You still" (truncated) triggers a false email reject via classifier
  - Caller clarifies over 2 more turns: "Uh, no. That's the one I gave you." /
    "one I gave you is the right one."
  - With max_tokens=2048 in the planner, turn 7 is correctly routed as
    CONFIRMATION so the mega-prompt can recover the email from transcript context
  - email_address ends up confirmed in collected exactly once
  - No double-COMPLETED

Seeded at the exact bug moment: first_name + last_name confirmed, email pending.

Run:
    cd data-plane
    PYTHONPATH=. python3 -m pytest tests/simulate_sanjay_planner.py -v -s
"""
from __future__ import annotations

import copy
import logging

import pytest

from app.agents.base import AgentStatus, SubagentResponse
from app.agents.data_collection import DataCollectionAgent
from app.agents.narrative_collection import NarrativeCollectionAgent
from app.agents.intake_qualification import IntakeQualificationAgent
from app.agents.faq import FAQAgent
from app.agents.context_docs import ContextDocsAgent
from app.agents.fallback import FallbackAgent
from app.agents.farewell import FarewellAgent
from app.agents.empathy import EmpathyAgent
from app.agents.scheduling import SchedulingAgent
from app.agents.graph_config import APPOINTMENT_BOOKING
from app.agents.workflow import WorkflowGraph
from app.agents.planner import Planner

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(name)s — %(message)s")

# ── Config ────────────────────────────────────────────────────────────────────

CONFIG = {
    "assistant": {"persona_name": "Aria", "narrative_topic": "your legal matter"},
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
    "faqs": [],
    "context_files": [],
    "practice_areas": [],
    "global_policy_documents": [],
    "_workflow_stages": "",
}

# ── Seeded history at the exact bug moment ────────────────────────────────────

_SEEDED_HISTORY = [
    {"role": "assistant", "content": "Could I get your First Name, please?"},
    {"role": "user",      "content": "Yeah. My first name is Sanjay."},
    {"role": "assistant", "content": "I have Sanjay — is that correct?"},
    {"role": "user",      "content": "Yeah. That's correct."},
    {"role": "assistant", "content": "Great, and could I get your Last Name, please?"},
    {"role": "user",      "content": "My last name is Sankhla."},
    {"role": "assistant", "content": "I have Sankhla — is that correct?"},
    {"role": "user",      "content": "Yes. And my email address is u dot s at c dot com."},
    {"role": "assistant", "content": "I have your email as u dot s at c dot com — is that correct?"},
]

_SEEDED_DC_STATE = {
    "collected": {
        "first_name": "Sanjay",
        "last_name": "Sankhla",
    },
    "pending_confirmation": {"field": "email_address", "value": "u.s@c.com"},
    "extraction_queue": [],
    "retry_count": 0,
    "history": list(_SEEDED_HISTORY),
}


# ── Pipeline helpers ──────────────────────────────────────────────────────────

def _build_registry() -> dict:
    return {
        "empathy":              EmpathyAgent(),
        "farewell":             FarewellAgent(),
        "faq":                  FAQAgent(),
        "context_docs":         ContextDocsAgent(),
        "fallback":             FallbackAgent(),
        "data_collection":      DataCollectionAgent(),
        "narrative_collection": NarrativeCollectionAgent(),
        "intake_qualification": IntakeQualificationAgent(),
        "scheduling":           SchedulingAgent(),
    }


def _enrich(config: dict, graph: WorkflowGraph, collected: dict) -> dict:
    return {
        **config,
        "_collected": dict(collected),
        "_booking": {},
        "_notes": "",
        "_tool_results": {},
        "_workflow_stages": graph.primary_goal_summary(),
    }


def _invoke(agent_id, utterance, turn_idx, graph, registry, planner,
            config, collected, call_history, chain_depth=0, max_chain=4):
    enriched = _enrich(config, graph, collected)
    agent = registry[agent_id]
    agent_state = graph.states[agent_id]

    response: SubagentResponse = agent.process(
        utterance, dict(agent_state.internal_state), enriched, call_history,
    )
    graph.update(agent_id, response, turn_idx)

    if response.collected:
        collected.update(response.collected)

    if response.status == AgentStatus.UNHANDLED:
        if chain_depth < max_chain:
            re_steps = planner.plan(utterance, call_history)
            re_agent = next(
                (s.agent_id for s in re_steps
                 if s.action == "invoke" and s.agent_id != agent_id), None,
            )
            if re_agent:
                return _invoke(re_agent, utterance, turn_idx, graph, registry,
                               planner, config, collected, call_history, chain_depth + 1)
        return "", None

    edge = graph.get_edge(agent_id, response.status)
    speak = response.speak or ""

    if edge.target in ("decider", "end"):
        return speak, edge.reason if edge.target == "end" else None
    if chain_depth >= max_chain:
        return speak, None
    if edge.target == "resume":
        resume_id = planner.pop_resume()
        if resume_id:
            rs, fr = _invoke(resume_id, "", turn_idx, graph, registry, planner,
                             config, collected, call_history, chain_depth + 1)
            return (speak + " " + rs).strip(), fr
        return speak, None

    chain_speak, finalize = _invoke(edge.target, "", turn_idx, graph, registry, planner,
                                    config, collected, call_history, chain_depth + 1)
    return (speak + " " + chain_speak).strip(), finalize


class SanjayPipelineRunner:
    """Pipeline runner seeded at the Sanjay bug moment."""

    def __init__(self) -> None:
        self.graph = WorkflowGraph(APPOINTMENT_BOOKING)
        self.planner = Planner(self.graph)
        self.registry = _build_registry()
        self.collected: dict = {"first_name": "Sanjay", "last_name": "Sankhla"}
        self.call_history: list[dict] = list(_SEEDED_HISTORY)
        self.turn_idx = 0
        self.statuses: list[AgentStatus] = []

        # Seed data_collection state at the bug moment
        dc_state = self.graph.states["data_collection"]
        dc_state.status = AgentStatus.WAITING_CONFIRM
        dc_state.internal_state = copy.deepcopy(_SEEDED_DC_STATE)

        print(f"\n{'=' * 70}")
        print("SIM: Sanjay Sankhla — full pipeline")
        print(f"{'=' * 70}")
        print(f"[seeded] pending=email_address='u.s@c.com'  collected={list(self.collected)}")

    def turn(self, utterance: str) -> str:
        self.turn_idx += 1
        self.call_history.append({"role": "user", "content": utterance})
        print(f"\n[T{self.turn_idx:02d}] Caller: {utterance!r}")

        steps = self.planner.plan(utterance, self.call_history)
        speaks = []

        for step in steps:
            if step.action == "reset_fields" and step.fields:
                self.planner.reset_fields(step.fields, self.collected)
            elif step.action == "invoke" and step.agent_id:
                speak, finalize = _invoke(
                    step.agent_id,
                    "" if step.use_empty_utterance else utterance,
                    self.turn_idx,
                    self.graph, self.registry, self.planner,
                    CONFIG, self.collected, self.call_history,
                )
                agent_state = self.graph.states.get(step.agent_id)
                if agent_state:
                    self.statuses.append(agent_state.status)
                if speak:
                    speaks.append((speak, step.agent_id, 1.0))
                if finalize:
                    break

        speak_text = self.planner.combine_speaks(speaks) or "(no speak)"
        self.call_history.append({"role": "assistant", "content": speak_text})

        dc_state = self.graph.states["data_collection"]
        pending = dc_state.internal_state.get("pending_confirmation")
        print(f"        AI:  {speak_text!r}")
        print(f"        dc_status={dc_state.status.value}  pending={pending}  collected={list(self.collected)}")
        return speak_text


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_email_confirmed_by_end_of_clarification():
    """
    Full clarification flow: "No. You still" triggers re-ask (acceptable),
    then caller clarifies across two more turns and email gets confirmed.

    With planner max_tokens=2048 the planner correctly routes turns 2 and 3
    as CONFIRMATION/CONTINUATION so the mega-prompt can recover the email.

    Final assertion: email_address is in collected with value u.s@c.com.
    """
    runner = SanjayPipelineRunner()

    runner.turn("No. You still")           # T1: false reject → re-ask email
    runner.turn("Uh, no. That's the one I gave you.")  # T2: clarification
    runner.turn("one I gave you is the right one.")    # T3: confirmation

    email = runner.collected.get("email_address")
    print(f"\n  Final collected: {runner.collected}")

    assert email, (
        f"BUG: email_address not confirmed after 3-turn clarification.\n"
        f"  collected={runner.collected}"
    )
    assert email == "u.s@c.com", (
        f"BUG: email_address confirmed with wrong value {email!r}, expected 'u.s@c.com'."
    )


def test_no_double_completed():
    """
    DataCollectionAgent must reach COMPLETED status exactly once —
    not on T2 and T3 both (double-speak bug from original transcript).
    """
    runner = SanjayPipelineRunner()

    runner.turn("No. You still")
    runner.turn("Uh, no. That's the one I gave you.")
    runner.turn("one I gave you is the right one.")

    completed_count = runner.statuses.count(AgentStatus.COMPLETED)
    print(f"\n  COMPLETED count across all turns: {completed_count}")
    print(f"  All statuses: {[s.value for s in runner.statuses]}")

    assert completed_count <= 1, (
        f"BUG: DataCollection reached COMPLETED {completed_count} times — "
        f"expected at most 1. Double-COMPLETED causes double-speak in TTS pipeline."
    )


def test_split_brain_guard_prevents_false_completion_speak():
    """
    On turn 2 ("Uh, no. That's the one I gave you."), the LLM may try to say
    'Perfect, I have what I need' while email is still missing.
    The split-brain guard must intercept this and return IN_PROGRESS instead.

    Checks that no turn produces a completion speak before email is confirmed.
    """
    runner = SanjayPipelineRunner()

    # After T1 the email is NOT in collected (false reject cleared it)
    speak_t1 = runner.turn("No. You still")

    # After T1, email must not be in collected
    assert "email_address" not in runner.collected, (
        "email_address should not be in collected after false reject on T1"
    )

    # T2: planner should route correctly now; check speak doesn't falsely say "I have everything"
    speak_t2 = runner.turn("Uh, no. That's the one I gave you.")

    # If email still not confirmed after T2, the speak must NOT be a completion phrase
    if "email_address" not in runner.collected:
        completion_phrases = (
            "have everything", "have what i need", "have all",
            "have the information", "i have what i need",
        )
        assert not any(p in speak_t2.lower() for p in completion_phrases), (
            f"BUG: Agent spoke completion phrase on T2 before email was confirmed.\n"
            f"  speak={speak_t2!r}\n"
            f"  collected={runner.collected}"
        )
