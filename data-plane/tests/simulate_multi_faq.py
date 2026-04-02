"""
Simulation: caller asks multiple questions in a single utterance.

The FAQAgent processes one question at a time — a single faq_match call
against the whole utterance. When a caller asks two distinct questions in
one breath, the agent may only answer the best-matched one and drop the rest.

This sim exposes that limitation and establishes a baseline for
what "all questions answered" would look like.

Scenarios:
  1. Two unrelated questions in one utterance (practice area + fees)
  2. Three questions in one utterance
  3. Multi-question during data collection (interrupt + resume)
  4. Two questions where one is a legal deflect + one is a real FAQ

Run:
    cd data-plane
    PYTHONPATH=. python3 -m pytest tests/simulate_multi_faq.py -v -s
"""
from __future__ import annotations

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

_PARAMS = [
    {
        "name": "full_name", "display_label": "Full Name",
        "data_type": "name", "required": True,
        "collection_order": 0, "extraction_hints": [],
    },
    {
        "name": "phone_number", "display_label": "Phone Number",
        "data_type": "phone", "required": True,
        "collection_order": 1, "extraction_hints": [],
    },
    {
        "name": "email_address", "display_label": "Email Address",
        "data_type": "email", "required": True,
        "collection_order": 2, "extraction_hints": [],
    },
]

_FAQS = [
    {
        "question": "What areas of law do you practice?",
        "answer": (
            "We handle three main areas: Immigration Law — including H1B visas, "
            "green cards, and work authorisation; Employment Law — including wrongful "
            "termination, discrimination, and unpaid wages; and Family Law — including "
            "divorce, custody, and domestic violence orders."
        ),
    },
    {
        "question": "How much does a consultation cost?",
        "answer": (
            "Initial consultations are $150 for a 30-minute session. "
            "If you retain us, that fee is applied toward your case."
        ),
    },
    {
        "question": "Do you offer free consultations?",
        "answer": (
            "We don't offer free consultations, but our initial consultation is $150 "
            "for 30 minutes, and that amount is credited toward your case if you retain us."
        ),
    },
    {
        "question": "Where are you located?",
        "answer": (
            "Our office is located at 123 Main Street, Suite 400, San Francisco, CA 94105. "
            "We're on the 4th floor of the Wells Fargo building, accessible by BART."
        ),
    },
    {
        "question": "What are your office hours?",
        "answer": (
            "We're open Monday through Friday, 9am to 6pm Pacific time. "
            "We also offer evening appointments on Tuesdays by request."
        ),
    },
]

_PRACTICE_AREAS = [
    {
        "name": "Immigration Law",
        "description": "H1B, green cards, visas, citizenship.",
        "qualification_criteria": "H1B transfers, visa renewals, green card filings.",
        "disqualification_signals": "Criminal immigration violations.",
        "ambiguous_signals": "Asylum claims.",
        "referral_suggestion": None,
    },
]

CONFIG = {
    "assistant": {"persona_name": "Aria", "narrative_topic": "your legal matter"},
    "parameters": _PARAMS,
    "faqs": _FAQS,
    "context_files": [],
    "practice_areas": _PRACTICE_AREAS,
    "global_policy_documents": [],
    "_workflow_stages": "",
}


# ── Pipeline runner (mirrored from simulate_faq_interrupt.py) ─────────────────

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
                re_speak, re_fr = _invoke(re_agent, utterance, turn_idx, graph, registry,
                                          planner, config, collected, call_history,
                                          chain_depth + 1)
                if not re_fr and agent_id == "data_collection":
                    dc_state = graph.states.get("data_collection")
                    if dc_state and dc_state.internal_state.get("pending_confirmation"):
                        dc_speak, dc_fr = _invoke(
                            "data_collection", "", turn_idx, graph, registry,
                            planner, config, collected, call_history, chain_depth + 1,
                        )
                        return (re_speak + " " + dc_speak).strip(), dc_fr
                return re_speak, re_fr
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


class PipelineRunner:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.graph = WorkflowGraph(APPOINTMENT_BOOKING)
        self.planner = Planner(self.graph)
        self.registry = _build_registry()
        self.collected: dict = {}
        self.call_history: list[dict] = []
        self.turn_idx = 0
        self.finalize_reason: str | None = None

        self.graph.states["data_collection"].status = AgentStatus.IN_PROGRESS
        opening_speak, _ = _invoke(
            "data_collection", "", 0,
            self.graph, self.registry, self.planner,
            self.config, self.collected, self.call_history,
        )
        if opening_speak:
            self.call_history.append({"role": "assistant", "content": opening_speak})
        self.last_speak = opening_speak
        print(f"\n{'=' * 70}")
        print(f"[open] AI: {opening_speak}")

    def turn(self, utterance: str) -> str:
        self.turn_idx += 1
        self.call_history.append({"role": "user", "content": utterance})

        steps = self.planner.plan(utterance, self.call_history)
        speaks: list = []
        fr: str | None = None
        waiting_speak: str | None = None

        for step in steps:
            if step.action == "reset_fields" and step.fields:
                self.planner.reset_fields(step.fields, self.collected)
            elif step.action == "invoke" and step.agent_id:
                agent_id = step.agent_id
                node = self.graph.nodes.get(agent_id)
                if node and node.interrupt_eligible:
                    primary_id = self.planner.active_primary_for_resume()
                    if primary_id:
                        self.planner.push_resume(primary_id)

                step_speak, step_fr = _invoke(
                    agent_id, utterance, self.turn_idx,
                    self.graph, self.registry, self.planner,
                    self.config, self.collected, self.call_history,
                )

                agent_state = self.graph.states.get(agent_id)
                if agent_state and agent_state.status == AgentStatus.WAITING_CONFIRM:
                    waiting_speak = step_speak
                    if step_fr is not None:
                        fr = step_fr
                    break

                if step_speak:
                    speaks.append((step_speak, agent_id, 1.0))
                if step_fr is not None:
                    fr = step_fr
                    break

        speak_text = (waiting_speak if waiting_speak is not None
                      else self.planner.combine_speaks(speaks))
        if not speak_text or not speak_text.strip():
            speak_text = "I'm sorry, I didn't quite catch that. Could you please say that again?"

        self.call_history.append({"role": "assistant", "content": speak_text})
        self.last_speak = speak_text
        if fr is not None:
            self.finalize_reason = fr

        dc = self.graph.states["data_collection"]
        dc_collected = dc.internal_state.get("collected", {})
        dc_pending = dc.internal_state.get("pending_confirmation")
        print(
            f"\n[T{self.turn_idx}] Caller: {utterance!r}\n"
            f"         AI: {speak_text}\n"
            f"         dc_pending={dc_pending}  collected={list(dc_collected.keys())}"
        )
        return speak_text

    def dc_collected(self) -> dict:
        return dict(self.graph.states["data_collection"].internal_state.get("collected") or {})

    def dc_pending(self) -> dict | None:
        return self.graph.states["data_collection"].internal_state.get("pending_confirmation")


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_two_faq_questions_in_one_utterance():
    """
    Caller asks two distinct FAQ questions in a single utterance:
      "What areas of law do you practice? And how much does a consultation cost?"

    Expected: both questions answered in the response.
    Currently the FAQAgent matches only one — this test documents the gap.
    """
    runner = PipelineRunner(CONFIG)

    speak = runner.turn(
        "What areas of law do you practice? And how much does a consultation cost?"
    )

    speak_lower = speak.lower()
    print(f"\n  Checking coverage of both questions in: {speak!r}")

    has_practice_area = any(kw in speak_lower for kw in ("immigration", "employment", "family", "law"))
    has_fee = any(kw in speak_lower for kw in ("150", "consultation", "fee", "cost", "charge"))

    print(f"  practice area answered: {has_practice_area}")
    print(f"  fee answered:           {has_fee}")

    assert has_practice_area, (
        f"Practice area question not answered in: {speak!r}"
    )
    assert has_fee, (
        f"Fee question not answered in: {speak!r}"
    )


def test_two_faq_questions_different_topics():
    """
    Caller asks two clearly distinct questions:
      "Where are you located and what are your office hours?"

    Expected: both location and hours addressed.
    """
    runner = PipelineRunner(CONFIG)

    speak = runner.turn(
        "Where are you located and what are your office hours?"
    )

    speak_lower = speak.lower()
    print(f"\n  Response: {speak!r}")

    has_location = any(kw in speak_lower for kw in (
        "located", "address", "street", "san francisco", "office", "floor", "bart"
    ))
    has_hours = any(kw in speak_lower for kw in (
        "monday", "friday", "9am", "6pm", "hours", "open", "tuesday", "evening"
    ))

    print(f"  location answered: {has_location}")
    print(f"  hours answered:    {has_hours}")

    assert has_location, f"Location question not answered in: {speak!r}"
    assert has_hours, f"Office hours question not answered in: {speak!r}"


def test_three_faq_questions_in_one_utterance():
    """
    Caller asks three questions in one breath:
      "What areas of law do you practice, where are you located,
       and do you offer free consultations?"

    All three topics should be addressed.
    """
    runner = PipelineRunner(CONFIG)

    speak = runner.turn(
        "What areas of law do you practice, where are you located, "
        "and do you offer free consultations?"
    )

    speak_lower = speak.lower()
    print(f"\n  Response: {speak!r}")

    has_practice = any(kw in speak_lower for kw in ("immigration", "employment", "family", "law"))
    has_location = any(kw in speak_lower for kw in ("located", "address", "san francisco", "office"))
    has_free    = any(kw in speak_lower for kw in ("free", "150", "fee", "consultation", "cost"))

    print(f"  practice area: {has_practice}")
    print(f"  location:      {has_location}")
    print(f"  free consult:  {has_free}")

    assert has_practice, f"Practice area not answered: {speak!r}"
    assert has_location, f"Location not answered: {speak!r}"
    assert has_free,     f"Free consultation not answered: {speak!r}"


def test_multi_faq_mid_collection_resumes():
    """
    Caller asks two FAQ questions mid-collection, then resumes providing data.

    Asserts:
    - Both questions addressed (or at minimum the response is substantive)
    - Data collection resumes after the FAQ interrupt
    - Already-confirmed fields are not lost or re-asked
    """
    runner = PipelineRunner(CONFIG)

    # T1: provide name → WAITING_CONFIRM
    runner.turn("My name is Priya Nair")
    assert runner.dc_pending() is not None, "Name should be pending confirmation"

    # T2: confirm name
    runner.turn("Yes")
    assert runner.dc_collected().get("full_name"), "full_name should be confirmed"
    print(f"  ✓ full_name confirmed: {runner.dc_collected()['full_name']!r}")

    # T3: two FAQ questions while collection is in progress
    speak = runner.turn(
        "Before we continue — what areas of law do you handle, "
        "and how much does a consultation cost?"
    )
    speak_lower = speak.lower()

    has_practice = any(kw in speak_lower for kw in ("immigration", "employment", "family", "law"))
    has_fee      = any(kw in speak_lower for kw in ("150", "consultation", "fee", "cost"))

    print(f"  Multi-FAQ mid-collection response: {speak!r}")
    print(f"  practice area: {has_practice}")
    print(f"  fee:           {has_fee}")

    # full_name must still be in collected — not dropped by FAQ interrupt
    assert runner.dc_collected().get("full_name"), (
        f"full_name was lost during FAQ interrupt! collected={runner.dc_collected()}"
    )
    print(f"  ✓ full_name preserved: {runner.dc_collected()['full_name']!r}")

    assert has_practice, f"Practice area question not answered: {speak!r}"
    assert has_fee,      f"Fee question not answered: {speak!r}"

    # T4-T5: resume — provide phone
    runner.turn("six five zero five five five one two three four")
    runner.turn("Yes")
    assert runner.dc_collected().get("phone_number"), (
        f"phone_number should be confirmed after resuming. collected={runner.dc_collected()}"
    )
    print(f"  ✓ phone_number confirmed after FAQ interrupt + resume")
