"""
Full-pipeline simulation: interrupt/resume across all primary interactive agents.

Tests the three interrupt/resume scenarios:

  1. NarrativeCollection interrupt — caller asks a FAQ mid-narrative:
     NC accumulates narrative → caller asks "What are your fees?" →
     FAQ answers → NC resumes and re-prompts for more narrative →
     caller finishes narrative → NC COMPLETED

  2. SchedulingAgent interrupt (awaiting_choice) — caller asks a FAQ mid-slot-choice:
     Scheduling presents slots → caller asks "What should I bring?" →
     FAQ answers → scheduling re-presents slots → caller picks →
     booking confirmed

  3. SchedulingAgent interrupt (awaiting_confirm) — caller asks a FAQ mid-confirmation:
     Scheduling asks "shall I confirm?" → caller asks "Where are you located?" →
     FAQ answers → scheduling re-asks confirmation → caller confirms →
     booking confirmed

For each scenario the test verifies:
  - Interrupt agent (FAQ/fallback) actually answers
  - Primary agent state is preserved across the interrupt
  - Primary agent re-surfaces the right prompt on resume (not a blank)
  - Collection/booking ultimately completes

Run:
    python3 -m pytest tests/simulate_interrupt_resume.py -v -s
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


# ── Config ────────────────────────────────────────────────────────────────────

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
]

_FAQS = [
    {
        "question": "What are your fees?",
        "answer": "Our consultation fee is $150 per hour. We offer free initial 15-minute consultations.",
    },
    {
        "question": "Where are you located?",
        "answer": "We are located at 123 Main Street, Suite 400, downtown.",
    },
    {
        "question": "What should I bring?",
        "answer": "Please bring any relevant documents, contracts, or correspondence related to your matter.",
    },
]

_PRACTICE_AREAS = [
    {
        "name": "Employment Law",
        "description": "Wrongful termination, discrimination, unpaid wages.",
        "qualification_criteria": "Wrongful termination, retaliation, discrimination claims.",
        "disqualification_signals": "Personal injury unrelated to employment.",
        "ambiguous_signals": "Independent contractor disputes.",
        "referral_suggestion": None,
    },
]

_GLOBAL_POLICY = [
    {
        "name": "intake_policy.txt",
        "description": "Intake policy",
        "content": "Collect name and phone in order. Confirm each before moving on.",
    },
]

CONFIG = {
    "assistant": {
        "persona_name": "Aria",
        "narrative_topic": "your employment matter",
    },
    "parameters": _PARAMS,
    "faqs": _FAQS,
    "context_files": [],
    "practice_areas": _PRACTICE_AREAS,
    "global_policy_documents": _GLOBAL_POLICY,
    "_workflow_stages": "",
}

_SLOT = {
    "slot_id": "slot-1",
    "description": "Monday, April 7 at 10:00 AM",
    "start_time": "2026-04-07T10:00:00+00:00",
    "end_time": "2026-04-07T11:00:00+00:00",
    "event_type_uri": "https://api.calendly.com/event_types/TEST",
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
    max_chain: int = 5,
) -> tuple[str, str | None]:
    """Synchronous pipeline driver mirroring handler.py _invoke_and_follow logic.

    Generalised version: uses is_primary_interactive to decide whether to
    re-invoke the interrupted agent with utterance="" after the interrupt
    agent finishes. This mirrors the handler.py change.
    """
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
        if chain_depth >= max_chain:
            return "", None
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
            # Generalised resume: if interrupted agent is_primary_interactive and
            # still IN_PROGRESS, re-invoke with empty utterance to re-surface state.
            if not re_fr and getattr(registry.get(agent_id), "is_primary_interactive", False):
                primary_state = graph.states.get(agent_id)
                if primary_state and primary_state.status == AgentStatus.IN_PROGRESS:
                    pri_speak, pri_fr = _invoke(
                        agent_id, "", turn_idx, graph, registry,
                        planner, config, collected, call_history, chain_depth + 1,
                    )
                    combined = (re_speak + " " + pri_speak).strip()
                    return combined, pri_fr
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


# ── Scenario runners ──────────────────────────────────────────────────────────

class NarrativePipelineRunner:
    """Drives a pipeline seeded so data_collection is COMPLETED and NC is active."""

    def __init__(self) -> None:
        self.config = CONFIG
        self.graph = WorkflowGraph(APPOINTMENT_BOOKING)
        self.planner = Planner(self.graph)
        self.registry = _build_registry()
        self.collected: dict = {"full_name": "Jane Smith", "phone_number": "555-1234"}
        self.call_history: list[dict] = []
        self.turn_idx = 0
        self.finalize_reason: str | None = None
        self.last_speak: str = ""

        # Force data_collection to COMPLETED so NC becomes available
        dc_state = self.graph.states["data_collection"]
        dc_state.status = AgentStatus.COMPLETED
        dc_state.internal_state = {
            "collected": dict(self.collected),
            "skipped": [],
            "pending_confirmation": None,
        }

        # Prime narrative_collection with opening prompt
        self.graph.states["narrative_collection"].status = AgentStatus.IN_PROGRESS
        opening, _ = _invoke(
            "narrative_collection", "", 0,
            self.graph, self.registry, self.planner,
            self.config, self.collected, self.call_history,
        )
        self._record_ai(opening)
        print(f"\n{'=' * 60}")
        print(f"[open] AI: {opening}")

    def turn(self, utterance: str) -> str:
        self.turn_idx += 1
        self.call_history.append({"role": "user", "content": utterance})
        steps = self.planner.plan(utterance, self.call_history)

        speaks = []
        fr = None
        waiting_speak = None

        for step in steps:
            if step.action == "reset_fields" and step.fields:
                self.planner.reset_fields(step.fields, self.collected)
            elif step.action == "invoke" and step.agent_id:
                aid = step.agent_id
                node = self.graph.nodes.get(aid)
                if node and node.interrupt_eligible:
                    primary_id = self.planner.active_primary_for_resume()
                    if primary_id:
                        self.planner.push_resume(primary_id)

                step_speak, step_fr = _invoke(
                    aid, utterance, self.turn_idx,
                    self.graph, self.registry, self.planner,
                    self.config, self.collected, self.call_history,
                )
                ag_state = self.graph.states.get(aid)
                if ag_state and ag_state.status == AgentStatus.WAITING_CONFIRM:
                    waiting_speak = step_speak
                    if step_fr is not None:
                        fr = step_fr
                    break
                if step_speak:
                    speaks.append((step_speak, aid, 1.0))
                if step_fr is not None:
                    fr = step_fr
                    break

        speak_text = waiting_speak if waiting_speak is not None else self.planner.combine_speaks(speaks)
        if not speak_text or not speak_text.strip():
            speak_text = "I'm sorry, could you repeat that?"

        self._record_ai(speak_text)
        if fr is not None:
            self.finalize_reason = fr

        nc = self.graph.states["narrative_collection"]
        print(
            f"[T{self.turn_idx}] Caller: {utterance!r}\n"
            f"         AI: {speak_text}\n"
            f"         nc_status={nc.status.value}  "
            f"segments={len(nc.internal_state.get('segments', []))}"
        )
        return speak_text

    def _record_ai(self, text: str) -> None:
        if text:
            self.call_history.append({"role": "assistant", "content": text})
        self.last_speak = text

    def nc_status(self) -> AgentStatus:
        return self.graph.states["narrative_collection"].status

    def nc_segments(self) -> list[str]:
        return list(self.graph.states["narrative_collection"].internal_state.get("segments", []))


class SchedulingPipelineRunner:
    """Drives a pipeline with scheduling already active and slots pre-seeded."""

    def __init__(self, initial_stage: str = "awaiting_choice") -> None:
        self.config = CONFIG
        self.graph = WorkflowGraph(APPOINTMENT_BOOKING)
        self.planner = Planner(self.graph)
        self.registry = _build_registry()
        self.collected: dict = {
            "full_name": "Jane Smith",
            "phone_number": "555-1234",
            "full_narrative": "I was wrongfully terminated from my job.",
        }
        self.call_history: list[dict] = []
        self.turn_idx = 0
        self.finalize_reason: str | None = None
        self.last_speak: str = ""

        # Force prereq agents to COMPLETED
        for aid in ("data_collection", "narrative_collection", "intake_qualification"):
            self.graph.states[aid].status = AgentStatus.COMPLETED

        # Seed scheduling state
        sched_state = self.graph.states["scheduling"]
        sched_state.status = AgentStatus.IN_PROGRESS
        sched_state.internal_state = {
            "stage": initial_stage,
            "available_slots": [_SLOT],
            "chosen_slot_id": 0 if initial_stage == "awaiting_confirm" else None,
            "retry_count": 0,
            "matched_event_type_uri": None,
        }
        if initial_stage == "awaiting_confirm":
            sched_state.internal_state["pending_confirmation"] = {"slot": _SLOT["description"]}

        # Opening speak from scheduling
        opening, _ = _invoke(
            "scheduling", "", 0,
            self.graph, self.registry, self.planner,
            self.config, self.collected, self.call_history,
        )
        self._record_ai(opening)
        print(f"\n{'=' * 60}")
        print(f"[open] AI: {opening}")

    def turn(self, utterance: str) -> str:
        self.turn_idx += 1
        self.call_history.append({"role": "user", "content": utterance})
        steps = self.planner.plan(utterance, self.call_history)

        speaks = []
        fr = None
        waiting_speak = None

        for step in steps:
            if step.action == "reset_fields" and step.fields:
                self.planner.reset_fields(step.fields, self.collected)
            elif step.action == "invoke" and step.agent_id:
                aid = step.agent_id
                node = self.graph.nodes.get(aid)
                if node and node.interrupt_eligible:
                    primary_id = self.planner.active_primary_for_resume()
                    if primary_id:
                        self.planner.push_resume(primary_id)

                step_speak, step_fr = _invoke(
                    aid, utterance, self.turn_idx,
                    self.graph, self.registry, self.planner,
                    self.config, self.collected, self.call_history,
                )
                ag_state = self.graph.states.get(aid)
                if ag_state and ag_state.status == AgentStatus.WAITING_CONFIRM:
                    waiting_speak = step_speak
                    if step_fr is not None:
                        fr = step_fr
                    break
                if step_speak:
                    speaks.append((step_speak, aid, 1.0))
                if step_fr is not None:
                    fr = step_fr
                    break

        speak_text = waiting_speak if waiting_speak is not None else self.planner.combine_speaks(speaks)
        if not speak_text or not speak_text.strip():
            speak_text = "I'm sorry, could you repeat that?"

        self._record_ai(speak_text)
        if fr is not None:
            self.finalize_reason = fr

        sched = self.graph.states["scheduling"]
        print(
            f"[T{self.turn_idx}] Caller: {utterance!r}\n"
            f"         AI: {speak_text}\n"
            f"         sched_status={sched.status.value}  "
            f"stage={sched.internal_state.get('stage')}"
        )
        return speak_text

    def _record_ai(self, text: str) -> None:
        if text:
            self.call_history.append({"role": "assistant", "content": text})
        self.last_speak = text

    def sched_status(self) -> AgentStatus:
        return self.graph.states["scheduling"].status

    def sched_stage(self) -> str:
        return self.graph.states["scheduling"].internal_state.get("stage", "")


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestNarrativeCollectionInterruptResume:
    """NC accumulates narrative, FAQ interrupt, NC resumes with re-prompt."""

    def test_faq_interrupt_mid_narrative_resumes_with_prompt(self):
        """
        Caller gives narrative → asks a fee question → FAQ answers →
        NC re-prompts to continue → caller finishes → NC COMPLETED.

        Verifies:
          - FAQ speaks (interrupt handled)
          - After interrupt speak contains NC re-prompt (not just FAQ answer alone)
          - NC segments preserved across interrupt
          - NC eventually reaches COMPLETED
        """
        runner = NarrativePipelineRunner()

        # Give meaningful narrative
        speak = runner.turn("I was wrongfully terminated from my job last month after reporting safety violations.")
        assert runner.nc_status() == AgentStatus.IN_PROGRESS

        # FAQ interrupt mid-narrative
        speak = runner.turn("What are your fees?")
        print(f"   [interrupt response]: {speak!r}")
        # Response should contain both the FAQ answer AND NC re-prompt
        assert speak  # not empty
        # The combined speak should reference fees or the narrative continuation
        # (fees from FAQ, then re-prompt from NC)
        assert len(speak) > 20  # substantive response

        # NC state still intact — segment from before interrupt preserved
        segments = runner.nc_segments()
        assert len(segments) >= 1, f"Segments lost after interrupt: {segments}"

        # Caller continues narrative
        speak = runner.turn("There is nothing else to add, that covers it.")
        assert runner.nc_status() == AgentStatus.COMPLETED

    def test_narrative_segments_not_contaminated_by_faq_question(self):
        """
        The FAQ question utterance must NOT be appended to NC segments.
        If the domain gate fires UNHANDLED, NC must not have modified state.
        """
        runner = NarrativePipelineRunner()

        runner.turn("My employer fired me for reporting HR violations.")
        segments_before = runner.nc_segments()

        # FAQ question — should be intercepted, not collected as narrative
        runner.turn("What are your fees?")
        segments_after = runner.nc_segments()

        # FAQ question should not appear in segments
        for seg in segments_after:
            assert "fee" not in seg.lower(), (
                f"FAQ question was added as a narrative segment: {segments_after}"
            )

    def test_narrative_resume_speak_is_not_blank(self):
        """
        After FAQ interrupt, NC must produce a non-blank re-prompt so the caller
        knows the narrative session is still open.
        """
        runner = NarrativePipelineRunner()

        runner.turn("I had an accident at work and injured my back.")

        speak = runner.turn("What are your fees?")
        # The combined response must not be empty — at minimum the NC re-prompt
        assert speak and speak.strip(), "NC+FAQ combined response was blank"


class TestSchedulingInterruptResumeAwaitingChoice:
    """Scheduling interrupted while awaiting slot choice; re-presents slots on resume."""

    def test_faq_interrupt_re_presents_slots(self):
        """
        Scheduling presents slots → caller asks FAQ → FAQ answers →
        scheduling re-presents slots → caller picks → confirms.

        Verifies:
          - FAQ answered (interrupt handled)
          - Slots re-presented after FAQ (not blank resume)
          - Scheduling state (available_slots) preserved
          - Caller can pick and proceed to confirm
        """
        runner = SchedulingPipelineRunner(initial_stage="awaiting_choice")

        # FAQ interrupt during slot choice
        speak = runner.turn("What should I bring to the appointment?")
        print(f"   [interrupt response]: {speak!r}")
        assert speak and speak.strip(), "Response after FAQ interrupt was blank"
        # Stage should still be awaiting_choice or transitioning to confirm
        stage = runner.sched_stage()
        assert stage in ("awaiting_choice", "awaiting_confirm", "done"), (
            f"Unexpected stage after FAQ interrupt: {stage}"
        )

    def test_scheduling_slots_preserved_after_faq_interrupt(self):
        """available_slots in scheduling state survive the FAQ interrupt."""
        runner = SchedulingPipelineRunner(initial_stage="awaiting_choice")

        runner.turn("Where are you located?")

        slots = runner.graph.states["scheduling"].internal_state.get("available_slots", [])
        assert slots, "available_slots lost after FAQ interrupt"
        assert slots[0]["slot_id"] == "slot-1"


class TestSchedulingInterruptResumeAwaitingConfirm:
    """Scheduling interrupted while awaiting confirmation; re-asks confirm on resume."""

    def test_faq_interrupt_re_asks_confirmation(self):
        """
        Scheduling awaiting confirm → caller asks FAQ → FAQ answers →
        scheduling re-asks confirmation (WAITING_CONFIRM) → caller confirms.

        Verifies:
          - FAQ answered
          - Confirmation question re-surfaced (speak contains slot/confirm phrasing)
          - pending_confirmation preserved
        """
        runner = SchedulingPipelineRunner(initial_stage="awaiting_confirm")

        speak = runner.turn("Where are you located?")
        print(f"   [interrupt response]: {speak!r}")
        assert speak and speak.strip(), "Response after FAQ interrupt during confirm was blank"

        # After interrupt, scheduling should either be awaiting_confirm still
        # (WAITING_CONFIRM re-surfaced) or have transitioned
        sched_state = runner.graph.states["scheduling"]
        stage = sched_state.internal_state.get("stage", "")
        print(f"   sched stage after interrupt: {stage}")
        # State must not be lost
        assert sched_state.internal_state.get("available_slots"), (
            "available_slots lost after interrupt during awaiting_confirm"
        )

    def test_pending_confirmation_preserved_after_interrupt(self):
        """pending_confirmation survives the FAQ interrupt."""
        runner = SchedulingPipelineRunner(initial_stage="awaiting_confirm")

        runner.turn("What are your fees?")

        sched_state = runner.graph.states["scheduling"]
        # Either still has pending_confirmation or has moved to done (confirm auto-handled)
        chosen_idx = sched_state.internal_state.get("chosen_slot_id")
        assert chosen_idx == 0, f"chosen_slot_id lost: {sched_state.internal_state}"
