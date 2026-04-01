"""
Regression test: fallback mid-collection does not produce a truncated reply.

Transcript being reproduced:
  [016.24s] CALLER: Hey. What information do you need to book me an appointment?
  [029.69s] CALLER: Okay. You got cut off there. Can you say that again?
  [044.66s] CALLER: Hey Arya, can you tell me something about yours?
  [046.15s] ARIA (BUG): "Sure thing — I don't have that information on hand,
             but I've noted your question and a team member"   ← truncated, drops
             the rest of the fallback sentence AND the data-collection resume.

Root cause: apply_empathy_filter called the Cerebras LLM with max_tokens=256,
which was too small to reproduce the combined speak text (fallback answer +
data_collection resume question).  Fixed by raising max_tokens to 512.

This simulation asserts:
  1. When a caller asks an unanswerable off-topic question mid-data-collection,
     the reply contains BOTH:
       a. a fallback acknowledgement ("noted" / "team member" / "follow up")
       b. the data-collection resume question ("first name" / "name")
  2. The combined speak ends with a question mark — i.e., it is not truncated
     mid-sentence before the data-collection question is appended.

Run:
    cd data-plane
    PYTHONPATH=. python3 -m pytest tests/simulate_fallback_mid_collection_truncation.py -v -s
"""
from __future__ import annotations

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
from app.agents.empathy_filter import apply_empathy_filter


# ── Config ────────────────────────────────────────────────────────────────────

_PARAMS = [
    {
        "name": "full_name",
        "display_label": "First Name",
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
    {
        "name": "email_address",
        "display_label": "Email Address",
        "data_type": "email",
        "required": True,
        "collection_order": 2,
        "extraction_hints": [],
    },
]

CONFIG = {
    "assistant": {
        "persona_name": "Aria",
        "narrative_topic": "your legal matter",
    },
    "parameters": _PARAMS,
    "faqs": [],          # no FAQs — forces fallback for any unknown question
    "context_files": [],
    "practice_areas": [],
    "global_policy_documents": [],
    "_workflow_stages": "",
}


# ── Pipeline driver (mirrors simulate_faq_interrupt.py) ───────────────────────

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


def _invoke(
    agent_id: str,
    utterance: str,
    turn_idx: int,
    graph: WorkflowGraph,
    registry: dict,
    planner: Planner,
    config: dict,
    collected: dict,
    call_history: list[dict],
    chain_depth: int = 0,
    max_chain: int = 4,
) -> tuple[str, str | None]:
    enriched = _enrich(config, graph, collected)
    agent = registry[agent_id]
    agent_state = graph.states[agent_id]

    response: SubagentResponse = agent.process(
        utterance, dict(agent_state.internal_state), enriched, call_history,
    )
    graph.update(agent_id, response, turn_idx)

    if response.collected:
        collected.update(response.collected)
    if response.hidden_collected:
        for k, v in response.hidden_collected.items():
            if k not in collected:
                collected[k] = v

    if response.status == AgentStatus.UNHANDLED:
        if chain_depth < max_chain:
            re_steps = planner.plan(utterance, call_history)
            re_agent = next(
                (s.agent_id for s in re_steps if s.action == "invoke" and s.agent_id != agent_id),
                None,
            )
            if re_agent:
                re_speak, re_fr = _invoke(
                    re_agent, utterance, turn_idx, graph, registry,
                    planner, config, collected, call_history, chain_depth + 1,
                )
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
            rs, fr = _invoke(
                resume_id, "", turn_idx, graph, registry, planner,
                config, collected, call_history, chain_depth + 1,
            )
            return (speak + " " + rs).strip(), fr
        return speak, None

    chain_speak, finalize = _invoke(
        edge.target, "", turn_idx, graph, registry, planner,
        config, collected, call_history, chain_depth + 1,
    )
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
        print(f"\n{'=' * 60}")
        print(f"[open] AI: {opening_speak}")

    def turn(self, utterance: str) -> str:
        self.turn_idx += 1
        self.call_history.append({"role": "user", "content": utterance})

        steps = self.planner.plan(utterance, self.call_history)
        speaks: list[tuple[str, str, float]] = []
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

        speak_text = waiting_speak if waiting_speak is not None else self.planner.combine_speaks(speaks)
        if not speak_text or not speak_text.strip():
            speak_text = "I'm sorry, I didn't quite catch that. Could you please say that again?"

        self.call_history.append({"role": "assistant", "content": speak_text})
        self.last_speak = speak_text
        if fr is not None:
            self.finalize_reason = fr

        print(f"[T{self.turn_idx}] Caller: {utterance!r}")
        print(f"         AI:     {speak_text}")
        return speak_text


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_fallback_mid_collection_includes_resume_question():
    """
    Caller asks an unanswerable off-topic question before providing their name.
    The fallback agent should handle it and the pipeline should append the
    data_collection resume question in the same reply.

    Reproduces: CALLER asks "Hey Arya, can you tell me something about yours?"
    and ARIA was truncated to "...a team member" without completing the sentence
    or appending the data-collection question.

    Asserts:
      - Reply contains the fallback acknowledgement (noted/team member/follow up)
      - Reply contains the data-collection resume question (name/first name)
      - Reply ends with '?' — i.e., it is not truncated before the question
    """
    runner = PipelineRunner(CONFIG)

    # Transcript T1: caller asks what info is needed (FAQ/general question)
    runner.turn("Hey. What information do you need to, um, book me an appointment?")

    # Transcript T2: caller says they got cut off — should repeat
    runner.turn("Okay. You got cut off there. Can you say that again?")

    # Transcript T3: the bug turn — caller asks something unanswerable mid-collection
    speak = runner.turn("Hey, Arya. Can you tell me something about yours?")

    print(f"\n--- Asserting T3 reply ---")
    print(f"Reply: {speak!r}")

    speak_lower = speak.lower()

    # 1. Fallback acknowledgement must be present
    fallback_keywords = ("noted", "team member", "follow up", "don't have", "information on hand")
    assert any(kw in speak_lower for kw in fallback_keywords), (
        f"Reply must contain a fallback acknowledgement. Got: {speak!r}"
    )
    print("  ✓ Fallback acknowledgement present")

    # 2. Data-collection resume question must be present
    dc_keywords = ("first name", "your name", "name, please", "name?")
    assert any(kw in speak_lower for kw in dc_keywords), (
        f"Reply must contain the data-collection resume question. Got: {speak!r}\n"
        f"Expected to see one of: {dc_keywords}"
    )
    print("  ✓ Data-collection resume question present")

    # 3. Reply must not be truncated — ends with a question mark
    assert speak.strip().endswith("?"), (
        f"Reply must end with '?' (data-collection question). Got: {speak!r}\n"
        "This indicates the combined speak was truncated before the resume question."
    )
    print("  ✓ Reply ends with '?' — not truncated")


def test_empathy_filter_does_not_truncate_composite_speak():
    """
    Unit test for apply_empathy_filter with a composite speak text matching the
    exact pattern from the bug transcript (fallback answer + data_collection question).

    Verifies that with max_tokens=512 the LLM reproduces the full combined text
    and does not drop the data-collection question.
    """
    composite_speak = (
        "I don't have that information on hand, but I've noted your question "
        "and a team member will make sure to follow up with you on that. "
        "Could I get your First Name, please?"
    )

    # Simulate a transcript where the caller described their situation
    transcript_turns = [
        {"role": "user", "content": "Hey Arya, can you tell me something about yours?"},
    ]
    collected = {}  # no name yet — empathy filter will use situation context

    result = apply_empathy_filter(
        composite_speak,
        collected,
        transcript_turns,
        agents_ran=frozenset({"faq"}),
    )

    print(f"\n--- Empathy filter unit test ---")
    print(f"Input:  {composite_speak!r}")
    print(f"Output: {result!r}")

    result_lower = result.lower()

    # The fallback sentence must survive
    assert any(kw in result_lower for kw in ("noted", "team member", "follow up")), (
        f"Empathy filter dropped the fallback acknowledgement. Output: {result!r}"
    )
    print("  ✓ Fallback sentence preserved")

    # The data-collection question must survive
    assert any(kw in result_lower for kw in ("first name", "your name", "name, please", "name?")), (
        f"Empathy filter dropped the data-collection question. Output: {result!r}\n"
        "This is the truncation bug — max_tokens was too low."
    )
    print("  ✓ Data-collection question preserved")

    # Must end with a question mark
    assert result.strip().endswith("?"), (
        f"Output must end with '?' to confirm both sentences present. Got: {result!r}"
    )
    print("  ✓ Output ends with '?' — not truncated")
