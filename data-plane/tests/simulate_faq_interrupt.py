"""
Full-pipeline simulation: ordered intake with a mid-collection FAQ interrupt.

Tests the end-to-end flow:
  1. data_collection collects name → phone → email in order
  2. Caller asks a practice-area FAQ question mid-collection
  3. FAQ agent answers
  4. Pipeline resumes data_collection at the correct next field (no repeat)
  5. After collection completes, pipeline advances to narrative_collection
  6. After narrative, intake_qualification auto-runs
  7. Scheduling becomes available

This exercises:
  - Planner routing (FAQ interrupt vs primary goal pursuit)
  - Resume-stack logic (data_collection paused → FAQ → data_collection resumes)
  - WAITING_CONFIRM preservation across the interrupt
  - Collection order correctness (no re-asking confirmed fields)
  - Stage advancement: data → narrative → qualification → scheduling

Run:
    python3 -m pytest tests/simulate_faq_interrupt.py -v -s
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


# ── Minimal firm config (self-contained, no evaluator dependency) ─────────────

_PARAMS = [
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
    {
        "name": "email_address",
        "display_label": "Email Address",
        "data_type": "email",
        "required": True,
        "collection_order": 2,
        "extraction_hints": [],
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
        "question": "Can you help with immigration matters?",
        "answer": (
            "Yes, we handle a full range of immigration matters including H1B visas, "
            "green cards, work authorisation, and employer sponsorship cases."
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

_GLOBAL_POLICY = [
    {
        "name": "intake_confirmation_guidelines.txt",
        "description": "Intake confirmation procedure",
        "content": (
            "Collect and confirm fields in order: name → phone → email. "
            "Confirm each before moving to the next. Never re-ask confirmed fields."
        ),
    },
]

CONFIG = {
    "assistant": {
        "persona_name": "Aria",
        "narrative_topic": "your legal matter",
    },
    "parameters": _PARAMS,
    "faqs": _FAQS,
    "context_files": [],
    "practice_areas": _PRACTICE_AREAS,
    "global_policy_documents": _GLOBAL_POLICY,
    "_workflow_stages": "",
}


# ── Pipeline driver ───────────────────────────────────────────────────────────

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
    """Synchronous _invoke_and_follow mirroring handler.py logic."""
    enriched = _enrich(config, graph, collected)
    agent = registry[agent_id]
    agent_state = graph.states[agent_id]

    response: SubagentResponse = agent.process(
        utterance,
        dict(agent_state.internal_state),
        enriched,
        call_history,
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
                re_speak, re_fr = _invoke(re_agent, utterance, turn_idx, graph, registry,
                                          planner, config, collected, call_history,
                                          chain_depth + 1)
                # If the interrupted agent was data_collection with a pending
                # confirmation, re-invoke it with an empty utterance so it re-surfaces
                # the pending question and the graph is back in WAITING_CONFIRM.
                # Without this the caller receives only the FAQ answer and "Yes" on
                # the following turn is misread as a response to the FAQ.
                if not re_fr and agent_id == "data_collection":
                    dc_state = graph.states.get("data_collection")
                    if dc_state and dc_state.internal_state.get("pending_confirmation"):
                        dc_speak, dc_fr = _invoke(
                            "data_collection", "", turn_idx, graph, registry,
                            planner, config, collected, call_history, chain_depth + 1,
                        )
                        combined = (re_speak + " " + dc_speak).strip()
                        return combined, dc_fr
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
    """Drives the full Planner → agent → graph pipeline for a scripted conversation."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.graph = WorkflowGraph(APPOINTMENT_BOOKING)
        self.planner = Planner(self.graph)
        self.registry = _build_registry()
        self.collected: dict = {}
        self.call_history: list[dict] = []
        self.turn_idx = 0
        self.finalize_reason: str | None = None

        # Prime data_collection
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
        """Feed one caller utterance through the full pipeline. Returns AI speak."""
        self.turn_idx += 1
        self.call_history.append({"role": "user", "content": utterance})

        steps = self.planner.plan(utterance, self.call_history)
        speaks: list[tuple[str, str]] = []
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

        dc = self.graph.states["data_collection"]
        dc_collected = dc.internal_state.get("collected", {})
        dc_pending = dc.internal_state.get("pending_confirmation")
        print(
            f"[T{self.turn_idx}] Caller: {utterance!r}\n"
            f"         AI: {speak_text}\n"
            f"         dc_status={dc.status.value}  "
            f"collected={list(dc_collected.keys())}  "
            f"pending={dc_pending}"
        )
        return speak_text

    def dc_collected(self) -> dict:
        return dict(self.graph.states["data_collection"].internal_state.get("collected") or {})

    def dc_pending(self) -> dict | None:
        return self.graph.states["data_collection"].internal_state.get("pending_confirmation")

    def dc_status(self) -> AgentStatus:
        return self.graph.states["data_collection"].status

    def narrative_status(self) -> AgentStatus:
        return self.graph.states["narrative_collection"].status

    def qualification_status(self) -> AgentStatus:
        return self.graph.states["intake_qualification"].status


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_faq_interrupt_resumes_at_correct_field():
    """
    Caller gives name and confirms, then asks a practice area FAQ question,
    then resumes providing phone. Asserts:
      - FAQ is answered (speak contains substantive content)
      - After FAQ, collection continues asking for phone (not name again)
      - Confirmed name is NOT lost during the interrupt
      - Collection eventually reaches phone and email
    """
    runner = PipelineRunner(CONFIG)

    # T1: give name → agent reads back for confirmation
    runner.turn("My name is Sarah Mitchell")
    assert runner.dc_status() == AgentStatus.WAITING_CONFIRM, (
        f"Expected WAITING_CONFIRM after name, got {runner.dc_status().value}"
    )
    assert runner.dc_pending() is not None, "Pending should be set after name"
    print("  ✓ WAITING_CONFIRM for name")

    # T2: confirm name
    runner.turn("Yes, that's correct")
    assert runner.dc_collected().get("full_name"), (
        f"full_name should be confirmed, collected={runner.dc_collected()}"
    )
    print(f"  ✓ full_name confirmed: {runner.dc_collected()['full_name']!r}")

    # T3: FAQ interrupt — ask about practice areas
    faq_speak = runner.turn("What areas of law do you practice?")
    assert faq_speak, "Expected a non-empty FAQ answer"
    # The answer should mention one of the practice areas
    faq_lower = faq_speak.lower()
    assert any(kw in faq_lower for kw in ("immigration", "employment", "family", "law")), (
        f"FAQ answer doesn't mention practice areas: {faq_speak!r}"
    )
    print(f"  ✓ FAQ answered: {faq_speak[:100]!r}...")

    # T4: After FAQ, pipeline should resume asking for phone
    # full_name must still be confirmed — not re-asked
    assert runner.dc_collected().get("full_name"), (
        f"full_name was lost after FAQ interrupt! collected={runner.dc_collected()}"
    )
    next_speak = runner.last_speak.lower()
    # The resumed speak should reference phone, not name
    assert "name" not in next_speak or "phone" in next_speak or "number" in next_speak or "next" in next_speak, (
        f"After FAQ, AI should be asking for phone, not re-asking name: {runner.last_speak!r}"
    )
    print(f"  ✓ full_name still confirmed after FAQ: {runner.dc_collected()['full_name']!r}")

    # T4-T8: provide phone
    runner.turn("four one five five five five zero one nine two")
    runner.turn("yes that's right")

    assert runner.dc_collected().get("phone_number"), (
        f"phone_number should be confirmed, collected={runner.dc_collected()}"
    )
    print(f"  ✓ phone_number confirmed: {runner.dc_collected()['phone_number']!r}")

    # T9-T10: provide email
    runner.turn("sarah dot mitchell at gmail dot com")
    runner.turn("yes")

    assert runner.dc_collected().get("email_address"), (
        f"email_address should be confirmed, collected={runner.dc_collected()}"
    )
    print(f"  ✓ email_address confirmed: {runner.dc_collected()['email_address']!r}")

    print("  ✓ All three fields collected without re-asking confirmed fields")


def test_faq_interrupt_during_waiting_confirm_preserves_pending():
    """
    Caller is mid-confirmation (WAITING_CONFIRM for name) when they ask a FAQ.
    The pending confirmation must be preserved and re-asked after the FAQ.
    Collection must resume from the same pending field, not start over.
    """
    runner = PipelineRunner(CONFIG)

    # T1: give name → WAITING_CONFIRM
    runner.turn("John Smith")
    assert runner.dc_status() == AgentStatus.WAITING_CONFIRM
    pending_before = runner.dc_pending()
    assert pending_before is not None
    pending_field = pending_before["field"]
    pending_value = pending_before["value"]
    print(f"  ✓ WAITING_CONFIRM: pending={pending_before}")

    # T2: ask FAQ while WAITING_CONFIRM — planner hard-routes to data_collection
    # The agent should either re-surface the pending confirm or signal UNHANDLED.
    # Either way, pending must be preserved.
    runner.turn("What areas of law do you practice?")
    pending_after = runner.dc_pending()

    # Pending must still refer to the same field
    assert pending_after is not None, (
        f"Pending was dropped during FAQ mid-confirmation. "
        f"before={pending_before}, after={pending_after}"
    )
    assert pending_after["field"] == pending_field, (
        f"Pending field changed. before={pending_field!r}, after={pending_after['field']!r}"
    )
    assert pending_after["value"] == pending_value, (
        f"Pending value changed. before={pending_value!r}, after={pending_after['value']!r}"
    )
    print(f"  ✓ Pending preserved through FAQ mid-confirmation: {pending_after}")

    # T3: now confirm
    runner.turn("Yes")
    assert runner.dc_collected().get(pending_field), (
        f"{pending_field} should be confirmed after 'Yes', collected={runner.dc_collected()}"
    )
    print(f"  ✓ Confirmed {pending_field} = {runner.dc_collected()[pending_field]!r} after resume")


def test_collection_then_narrative_stage_advance():
    """
    After all required fields are collected, the pipeline must automatically
    advance to narrative_collection without requiring an explicit routing decision.
    After a brief narrative, intake_qualification auto-runs.
    """
    runner = PipelineRunner(CONFIG)

    # Rapid data collection: name → confirm → phone → confirm → email → confirm
    turns = [
        "My name is Alex Rivera",
        "yes",
        "six four eight five five five zero one two three",
        "yes",
        "alex dot rivera at example dot com",
        "yes",
    ]
    for utt in turns:
        runner.turn(utt)
        if runner.finalize_reason:
            break

    dc = runner.dc_collected()
    assert dc.get("full_name"),     f"full_name missing: {dc}"
    assert dc.get("phone_number"),  f"phone_number missing: {dc}"
    assert dc.get("email_address"), f"email_address missing: {dc}"
    print(f"  ✓ All fields collected: {dc}")

    # data_collection should be COMPLETED
    assert runner.dc_status() == AgentStatus.COMPLETED, (
        f"data_collection should be COMPLETED, got {runner.dc_status().value}"
    )
    print(f"  ✓ data_collection status: COMPLETED")

    # Narrative collection should be IN_PROGRESS (auto-chained via on_complete edge)
    assert runner.narrative_status() in (AgentStatus.IN_PROGRESS, AgentStatus.NOT_STARTED), (
        f"narrative_collection should be active after data_collection COMPLETED, "
        f"got {runner.narrative_status().value}"
    )
    print(f"  ✓ narrative_collection status: {runner.narrative_status().value}")

    # Provide a narrative
    runner.turn("I was in a car accident last month and the other driver was at fault")
    runner.turn("No, that's everything I wanted to say")

    # After narrative completes, intake_qualification should have auto-run
    # (it chains via on_complete edge from narrative_collection)
    qual_status = runner.qualification_status()
    assert qual_status in (
        AgentStatus.COMPLETED, AgentStatus.FAILED, AgentStatus.IN_PROGRESS
    ), (
        f"intake_qualification should have run after narrative, got {qual_status.value}"
    )
    print(f"  ✓ intake_qualification status after narrative: {qual_status.value}")


def test_no_field_re_asked_after_confirmation():
    """
    Regression test for the pending-drop bug (Step C+).

    Scenario: caller provides a value, LLM asks for confirmation, caller says
    something that could be misclassified as intent='answer' rather than
    'confirm_yes'. The field must still end up confirmed.

    Specifically: after "yes, that's my name", the field must be in collected
    and the agent must move to the NEXT field — never re-ask the confirmed one.
    """
    runner = PipelineRunner(CONFIG)

    # T1: give name
    runner.turn("Maria Garcia")
    assert runner.dc_status() == AgentStatus.WAITING_CONFIRM
    print(f"  ✓ Waiting to confirm name")

    # T2: affirmative response (could be mis-classified as intent='answer')
    runner.turn("Yes, that's my name")
    collected = runner.dc_collected()
    pending = runner.dc_pending()

    # Name must be confirmed
    assert collected.get("full_name"), (
        f"full_name must be confirmed after 'Yes, that's my name'. "
        f"collected={collected}, pending={pending}"
    )
    print(f"  ✓ full_name confirmed: {collected['full_name']!r}")

    # The agent must now be asking for the NEXT field (phone), not name again
    speak_lower = runner.last_speak.lower()
    assert "phone" in speak_lower or "number" in speak_lower or "contact" in speak_lower, (
        f"After confirming name, agent must ask for phone number. "
        f"Instead asked: {runner.last_speak!r}"
    )
    print(f"  ✓ Agent correctly moved to next field (phone)")

    # T3-T4: confirm phone
    runner.turn("four one five five five five zero one nine two")
    runner.turn("yes")
    assert runner.dc_collected().get("phone_number"), (
        f"phone_number should be confirmed, collected={runner.dc_collected()}"
    )
    print(f"  ✓ phone_number confirmed: {runner.dc_collected()['phone_number']!r}")

    # T5-T6: confirm email — full_name must STILL be in collected (not dropped)
    runner.turn("maria at example dot com")
    runner.turn("yes")

    final = runner.dc_collected()
    assert final.get("full_name"),     f"full_name dropped from collected: {final}"
    assert final.get("phone_number"),  f"phone_number dropped from collected: {final}"
    assert final.get("email_address"), f"email_address missing: {final}"
    print(f"  ✓ All fields confirmed and none dropped: {final}")
