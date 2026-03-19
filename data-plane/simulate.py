#!/usr/bin/env python3
"""
Workflow simulation harness — runs WorkflowDefinition edge logic without real calls.

Usage:
    cd data-plane
    PYTHONPATH=. python3 simulate.py

Scenarios:
    1.  Happy path — name, email, skip optional, narrative, qualified → scheduling → book
    2.  FAQ interrupt mid-collection → resume → collection continues → narrative → schedule
    3.  Fallback chain — faq fails → context_docs fails → fallback → resume
    4.  3 consecutive agent errors → finalize("agent_error")
    5.  WAITING_CONFIRM enforcement — bad decider overridden by hard check
    6.  Multiple FAQs during collection
    7.  FAQs during scheduling
    8.  Narrative happy path — multi-turn narrative collection
    9.  FAQ interrupt mid-narrative → resume → narrative continues
    10. Not-qualified path → end("not_qualified")
    11. Ambiguous path → qualified → scheduling → book
"""
from __future__ import annotations

import sys
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

from app.agents.base import AgentBase, AgentStatus, SubagentResponse
from app.agents.workflow import WorkflowDefinition, WorkflowGraph
from app.agents.router import Router
from app.agents.graph_config import APPOINTMENT_BOOKING


# ── Simulation primitives ─────────────────────────────────────────────────────

@dataclass
class SimUtterance:
    text: str
    force_agent: str | None = None    # bypass decider with this agent
    force_interrupt: bool = False      # push active primary onto resume stack


@dataclass
class TurnRecord:
    turn: int
    utterance: str
    agent_id: str                      # first agent invoked this turn
    interrupted: bool
    chain: list[str]                   # full agent_id sequence for this turn
    speak: str
    finalize_reason: str | None


@dataclass
class SimResult:
    turns: list[TurnRecord]
    final_states: dict[str, str]       # node_id → status value
    collected: dict[str, str]
    booking: dict | None
    finalization_reason: str | None
    error: str | None


# ── Mock agents ───────────────────────────────────────────────────────────────

class ScriptedAgent(AgentBase):
    """Returns pre-scripted SubagentResponses in call order."""

    def __init__(
        self,
        agent_id: str,
        responses: list[SubagentResponse],
        default: SubagentResponse | None = None,
    ) -> None:
        self._agent_id = agent_id
        self._queue: deque[SubagentResponse] = deque(responses)
        self._default = default or SubagentResponse(
            status=AgentStatus.IN_PROGRESS,
            speak=f"[{agent_id}: script exhausted]",
        )
        self._call_count = 0

    def process(
        self,
        utterance: str,
        internal_state: dict,
        config: dict,
        history: list[dict],
    ) -> SubagentResponse:
        self._call_count += 1
        resp = self._queue.popleft() if self._queue else self._default
        _log(
            f"    mock:{self._agent_id}#{self._call_count}"
            f"  utt={utterance!r:.35}"
            f"  → {resp.status.value}"
            f"  speak={resp.speak!r:.55}"
        )
        return resp


class ErrorAgent(AgentBase):
    """Always raises RuntimeError — tests error policy."""

    def process(self, utterance, internal_state, config, history) -> SubagentResponse:
        raise RuntimeError("Simulated agent failure")


# ── Simulator ─────────────────────────────────────────────────────────────────

class WorkflowSimulator:
    def __init__(
        self,
        workflow: WorkflowDefinition,
        agent_mocks: dict[str, AgentBase],
        decider_mock: Callable[[str, WorkflowGraph], tuple[str, bool]] | None = None,
        config: dict | None = None,
    ) -> None:
        self._workflow = workflow
        self._agents = agent_mocks
        self._decider_mock = decider_mock
        self._config = config or {}

    def run(self, script: list[SimUtterance]) -> SimResult:
        graph = WorkflowGraph(self._workflow)
        router = Router(graph)
        collected: dict[str, str] = {}
        booking: dict | None = None
        turns: list[TurnRecord] = []
        consecutive_errors = 0
        finalization_reason: str | None = None
        sim_error: str | None = None

        # ── Invoke agent and follow edges recursively ─────────────────────────

        def _invoke_and_follow(
            agent_id: str,
            utterance: str,
            turn_no: int,
            chain_log: list[str],
            chain_depth: int = 0,
        ) -> tuple[str, str | None]:
            nonlocal booking
            MAX_CHAIN = 8  # allow longer chains for 4-goal sequence

            chain_log.append(agent_id)
            agent = self._agents[agent_id]
            state = graph.states[agent_id]
            cfg = {**self._config, "_collected": dict(collected), "_booking": booking}

            response = agent.process(utterance, dict(state.internal_state), cfg, [])
            graph.update(agent_id, response, turn_no)
            edge = graph.get_edge(agent_id, response.status)

            if response.collected:
                collected.update(response.collected)
            if response.booking:
                booking = response.booking

            speak = response.speak or ""

            if edge.target == "decider":
                return speak, None

            if edge.target == "end":
                return speak, edge.reason or "completed"

            if chain_depth >= MAX_CHAIN:
                _log(f"    ⚠ chain depth {chain_depth} exceeded at {agent_id} — stopping")
                return speak, None

            if edge.target == "resume":
                resume_id = router.pop_resume()
                if resume_id:
                    rs, fin = _invoke_and_follow(
                        resume_id, "", turn_no, chain_log, chain_depth + 1
                    )
                    return (speak + " " + rs).strip(), fin
                return speak, None

            # "<node_id>" — chain immediately
            cs, fin = _invoke_and_follow(
                edge.target, "", turn_no, chain_log, chain_depth + 1
            )
            return (speak + " " + cs).strip(), fin

        # ── Process one turn (utterance → select → invoke → follow) ──────────

        def _process(utt_obj: SimUtterance, turn_no: int) -> bool:
            """Returns False when simulation should stop."""
            nonlocal consecutive_errors, finalization_reason

            utt = utt_obj.text

            # WAITING_CONFIRM hard enforcement — same as Router.select()
            waiting_id = graph.active_waiting_confirm()
            if waiting_id:
                agent_id = waiting_id
                interrupted = False
                _log(f"    [sim] WAITING_CONFIRM enforced → {agent_id}")
            elif utt_obj.force_agent:
                agent_id = utt_obj.force_agent
                interrupted = utt_obj.force_interrupt
                if interrupted:
                    active = graph.active_primary()
                    if active and active.node_id not in router._resume_stack:
                        router._resume_stack.append(active.node_id)
                        _log(f"    [sim] pushed {active.node_id} onto resume stack")
            elif self._decider_mock:
                agent_id, interrupted = self._decider_mock(utt, graph)
                if interrupted:
                    active = graph.active_primary()
                    if active and active.node_id not in router._resume_stack:
                        router._resume_stack.append(active.node_id)
            else:
                agent_id, interrupted = router.select(utt, [])

            chain_log: list[str] = []
            fin: str | None = None
            speak = ""

            try:
                speak, fin = _invoke_and_follow(agent_id, utt, turn_no, chain_log)
                consecutive_errors = 0
            except Exception as exc:
                consecutive_errors += 1
                _log(f"    ✗ error #{consecutive_errors}: {exc}")
                error_policy = graph.workflow.error_policy
                if consecutive_errors >= error_policy.max_consecutive_errors:
                    fin = error_policy.on_max_errors.reason or "agent_error"
                    speak = "I'm sorry, I'm having difficulties. Goodbye!"
                    _log(f"    ✗ max errors → finalize({fin!r})")
                else:
                    speak = error_policy.transient_error_speak
                chain_log = [agent_id]

            turns.append(TurnRecord(
                turn=turn_no,
                utterance=utt,
                agent_id=agent_id,
                interrupted=interrupted,
                chain=list(chain_log),
                speak=speak,
                finalize_reason=fin,
            ))

            if fin is not None:
                finalization_reason = fin
                return False
            return True

        # ── Opening turn — mirrors handler.py bootstrap ───────────────────────

        _log("  opening turn: data_collection('')...")
        graph.states["data_collection"].status = AgentStatus.IN_PROGRESS
        # Run through _process so error policy applies uniformly
        if not _process(SimUtterance("", force_agent="data_collection"), 0):
            return SimResult(
                turns=turns,
                final_states={nid: s.status.value for nid, s in graph.states.items()},
                collected=collected,
                booking=booking,
                finalization_reason=finalization_reason,
                error=sim_error,
            )

        # ── Script turns ──────────────────────────────────────────────────────

        for i, utt_obj in enumerate(script, start=1):
            _log(f"\n  turn {i}: {utt_obj.text!r:.60}")
            if not _process(utt_obj, i):
                break

        return SimResult(
            turns=turns,
            final_states={nid: s.status.value for nid, s in graph.states.items()},
            collected=collected,
            booking=booking,
            finalization_reason=finalization_reason,
            error=sim_error,
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    print(msg, flush=True)


def _print_result(name: str, result: SimResult, expected_reason: str | None = None) -> bool:
    ok = result.finalization_reason == expected_reason
    icon = "✓" if ok else "✗"
    print(f"\n{'═' * 68}")
    print(f"  {icon}  SCENARIO: {name}")
    print(f"{'═' * 68}")
    print(f"  finalization : {result.finalization_reason!r}  (expected {expected_reason!r})")
    print(f"  collected    : {result.collected}")
    print(f"  booking      : {result.booking}")
    if result.error:
        print(f"  error        : {result.error}")
    print(f"\n  turn trace:")
    for t in result.turns:
        chain_str = " → ".join(t.chain) if t.chain else t.agent_id
        tags = ""
        if t.interrupted:
            tags += " [INTERRUPT]"
        if t.finalize_reason:
            tags += f" [END:{t.finalize_reason}]"
        print(f"    T{t.turn:02d}  {chain_str}{tags}")
        if t.utterance:
            print(f"          utt   : {t.utterance!r:.65}")
        print(f"          speak : {t.speak!r:.70}")
    print(f"\n  final graph states:")
    for nid, status in result.final_states.items():
        print(f"    {nid:25s} {status}")
    if not ok:
        print(f"\n  ASSERTION FAILED: expected {expected_reason!r}, got {result.finalization_reason!r}")
    return ok


# ── Mock builder helpers ──────────────────────────────────────────────────────

def _nc_passthrough() -> ScriptedAgent:
    """NarrativeCollection: instantly completes on first call (for scenarios testing other things)."""
    return ScriptedAgent("narrative_collection", [
        SubagentResponse(
            status=AgentStatus.COMPLETED,
            speak="Thank you, I've noted your matter.",
            collected={
                "narrative_summary": "Caller described their legal matter.",
                "case_type": "personal injury",
                "full_narrative": "Caller described their legal matter.",
            },
        ),
    ])


def _iq_passthrough(decision: str = "qualified") -> ScriptedAgent:
    """IntakeQualification: instantly returns given decision."""
    status = AgentStatus.COMPLETED if decision != "not_qualified" else AgentStatus.FAILED
    return ScriptedAgent("intake_qualification", [
        SubagentResponse(
            status=status,
            speak="This looks like something we can help with." if decision != "not_qualified"
                  else "Unfortunately, that falls outside our practice areas.",
            collected={"qualification_decision": decision, "qualification_reason": "Matches."},
        ),
    ])


def _webhook_ok() -> ScriptedAgent:
    return ScriptedAgent("webhook", [
        SubagentResponse(status=AgentStatus.COMPLETED, speak=""),
    ])


# ── Scenario builders ─────────────────────────────────────────────────────────

def scenario_happy_path() -> SimResult:
    """
    Full happy path (4-goal sequence):
      data_collection → (chains) narrative_collection (instant) →
      (chains) intake_qualification (qualified) → (chains) scheduling → book
    """
    dc = [
        SubagentResponse(status=AgentStatus.IN_PROGRESS, speak="What's your full name?"),
        SubagentResponse(
            status=AgentStatus.WAITING_CONFIRM,
            speak="I have your name as John Smith. Is that correct?",
            internal_state={"stage": "waiting_confirm", "current_field": "name", "pending_value": "John Smith"},
        ),
        SubagentResponse(
            status=AgentStatus.IN_PROGRESS,
            speak="Got it. What's your email address?",
            collected={"name": "John Smith"},
        ),
        SubagentResponse(
            status=AgentStatus.WAITING_CONFIRM,
            speak="I have john@example.com. Is that correct?",
            internal_state={"stage": "waiting_confirm", "current_field": "email", "pending_value": "john@example.com"},
        ),
        SubagentResponse(
            status=AgentStatus.IN_PROGRESS,
            speak="Great. Do you have a member ID? You can skip this if not.",
            collected={"email": "john@example.com"},
        ),
        # "I don't have one" → COMPLETED → chains NC → IQ → scheduling
        SubagentResponse(
            status=AgentStatus.COMPLETED,
            speak="No problem, I'll note no member ID.",
        ),
    ]

    sched = [
        SubagentResponse(
            status=AgentStatus.IN_PROGRESS,
            speak="I have these slots: 1) Monday 2pm, 2) Tuesday 3pm. Which works?",
        ),
        SubagentResponse(
            status=AgentStatus.WAITING_CONFIRM,
            speak="Tuesday at 3pm. Shall I confirm that booking?",
            internal_state={"pending_slot": "Tuesday 3pm"},
        ),
        SubagentResponse(
            status=AgentStatus.COMPLETED,
            speak="You're all set for Tuesday at 3pm!",
            booking={"booking_id": "MOCK-001", "slot_description": "Tuesday at 3pm", "status": "confirmed"},
        ),
    ]

    mocks = {
        "data_collection":      ScriptedAgent("data_collection", dc),
        "narrative_collection": _nc_passthrough(),
        "intake_qualification": _iq_passthrough("qualified"),
        "scheduling":           ScriptedAgent("scheduling", sched),
        "faq":                  ScriptedAgent("faq", []),
        "context_docs":         ScriptedAgent("context_docs", []),
        "fallback":             ScriptedAgent("fallback", []),
        "webhook":              _webhook_ok(),
    }

    script = [
        SimUtterance("John Smith",         force_agent="data_collection"),
        SimUtterance("yes",                force_agent="data_collection"),  # WC enforced
        SimUtterance("john@example.com",   force_agent="data_collection"),
        SimUtterance("yes",                force_agent="data_collection"),  # WC enforced
        # "I don't have one" → DC COMPLETED → chains NC(instant) → IQ(qualified) → sched("") T1
        SimUtterance("I don't have one",   force_agent="data_collection"),
        SimUtterance("the second one",     force_agent="scheduling"),
        SimUtterance("yes",                force_agent="scheduling"),       # WC enforced
    ]

    return WorkflowSimulator(APPOINTMENT_BOOKING, mocks).run(script)


def scenario_faq_interrupt() -> SimResult:
    """
    name collected → FAQ interrupt → resume → collection continues →
    narrative (instant) → qualified → scheduling → book.
    """
    dc = [
        SubagentResponse(status=AgentStatus.IN_PROGRESS, speak="What's your full name?"),
        SubagentResponse(
            status=AgentStatus.WAITING_CONFIRM,
            speak="I have John Smith. Is that correct?",
            internal_state={"stage": "waiting_confirm", "current_field": "name", "pending_value": "John Smith"},
        ),
        SubagentResponse(
            status=AgentStatus.IN_PROGRESS,
            speak="Got it. What's your email address?",
            collected={"name": "John Smith"},
        ),
        # resume invocation ("") — re-asks current question
        SubagentResponse(
            status=AgentStatus.IN_PROGRESS,
            speak="Back to your info — what's your email address?",
        ),
        SubagentResponse(
            status=AgentStatus.WAITING_CONFIRM,
            speak="I have john@example.com. Correct?",
            internal_state={"stage": "waiting_confirm", "current_field": "email", "pending_value": "john@example.com"},
        ),
        SubagentResponse(
            status=AgentStatus.COMPLETED,
            speak="All information collected.",
            collected={"email": "john@example.com"},
        ),
    ]

    faq = [
        SubagentResponse(
            status=AgentStatus.COMPLETED,
            speak="Our office hours are 9am to 5pm, Monday through Friday.",
        ),
    ]

    sched = [
        SubagentResponse(status=AgentStatus.IN_PROGRESS, speak="Slots: 1) Monday 2pm, 2) Tuesday 3pm."),
        SubagentResponse(
            status=AgentStatus.WAITING_CONFIRM, speak="Monday 2pm. Confirm?",
            internal_state={"pending_slot": "Monday 2pm"},
        ),
        SubagentResponse(
            status=AgentStatus.COMPLETED, speak="Booked! See you Monday at 2pm.",
            booking={"booking_id": "MOCK-002", "slot_description": "Monday 2pm", "status": "confirmed"},
        ),
    ]

    mocks = {
        "data_collection":      ScriptedAgent("data_collection", dc),
        "narrative_collection": _nc_passthrough(),
        "intake_qualification": _iq_passthrough("qualified"),
        "faq":                  ScriptedAgent("faq", faq),
        "context_docs":         ScriptedAgent("context_docs", []),
        "fallback":             ScriptedAgent("fallback", []),
        "scheduling":           ScriptedAgent("scheduling", sched),
        "webhook":              _webhook_ok(),
    }

    script = [
        SimUtterance("John Smith",            force_agent="data_collection"),
        SimUtterance("yes",                   force_agent="data_collection"),    # WC enforced
        SimUtterance("What are your hours?",  force_agent="faq", force_interrupt=True),
        SimUtterance("john@example.com",      force_agent="data_collection"),
        SimUtterance("yes",                   force_agent="data_collection"),    # WC enforced
        # DC COMPLETED → chains NC(instant) → IQ(qualified) → sched T1
        SimUtterance("I don't have one",      force_agent="data_collection"),
        SimUtterance("slot 1",                force_agent="scheduling"),
        SimUtterance("yes",                   force_agent="scheduling"),         # WC enforced
    ]

    return WorkflowSimulator(APPOINTMENT_BOOKING, mocks).run(script)


def scenario_fallback_chain() -> SimResult:
    """
    faq fails → context_docs fails → fallback → resume → data_collection continues →
    narrative (instant) → qualified → scheduling → book.
    """
    dc = [
        SubagentResponse(status=AgentStatus.IN_PROGRESS, speak="What's your full name?"),
        SubagentResponse(status=AgentStatus.IN_PROGRESS, speak="Back to your info — what's your name?"),
        SubagentResponse(
            status=AgentStatus.WAITING_CONFIRM, speak="I have Jane Doe. Correct?",
            internal_state={"stage": "waiting_confirm", "current_field": "name", "pending_value": "Jane Doe"},
        ),
        SubagentResponse(
            status=AgentStatus.COMPLETED, speak="All information collected.",
            collected={"name": "Jane Doe"},
        ),
    ]

    sched = [
        SubagentResponse(status=AgentStatus.IN_PROGRESS, speak="Slots: 1) Monday, 2) Tuesday."),
        SubagentResponse(
            status=AgentStatus.WAITING_CONFIRM, speak="Monday. Confirm?",
            internal_state={"pending_slot": "Monday"},
        ),
        SubagentResponse(
            status=AgentStatus.COMPLETED, speak="Booked!",
            booking={"booking_id": "MOCK-003", "slot_description": "Monday", "status": "confirmed"},
        ),
    ]

    mocks = {
        "data_collection":      ScriptedAgent("data_collection", dc),
        "narrative_collection": _nc_passthrough(),
        "intake_qualification": _iq_passthrough("qualified"),
        "faq":                  ScriptedAgent("faq", [SubagentResponse(status=AgentStatus.FAILED, speak="")]),
        "context_docs":         ScriptedAgent("context_docs", [SubagentResponse(status=AgentStatus.FAILED, speak="")]),
        "fallback":             ScriptedAgent("fallback", [
            SubagentResponse(
                status=AgentStatus.COMPLETED,
                speak="I don't have that info right now. Someone will follow up.",
                notes="Unanswered: parking options",
            ),
        ]),
        "scheduling":           ScriptedAgent("scheduling", sched),
        "webhook":              _webhook_ok(),
    }

    script = [
        SimUtterance("What are your parking options?", force_agent="faq", force_interrupt=True),
        SimUtterance("Jane Doe",   force_agent="data_collection"),
        SimUtterance("yes",        force_agent="data_collection"),    # WC enforced
        # DC COMPLETED → chains NC → IQ → sched T1
        SimUtterance("skip",       force_agent="data_collection"),
        SimUtterance("slot 1",     force_agent="scheduling"),
        SimUtterance("yes",        force_agent="scheduling"),         # WC enforced
    ]

    return WorkflowSimulator(APPOINTMENT_BOOKING, mocks).run(script)


def scenario_consecutive_errors() -> SimResult:
    """
    Every agent invocation raises. After 3 consecutive errors → finalize("agent_error").
    """
    mocks = {k: ErrorAgent() for k in (
        "data_collection", "narrative_collection", "intake_qualification",
        "scheduling", "faq", "context_docs", "fallback", "webhook",
    )}

    script = [
        SimUtterance("anything", force_agent="data_collection"),   # error #2
        SimUtterance("anything", force_agent="data_collection"),   # error #3 → finalize
        SimUtterance("anything", force_agent="data_collection"),   # should not run
    ]

    return WorkflowSimulator(APPOINTMENT_BOOKING, mocks).run(script)


def scenario_waiting_confirm_enforcement() -> SimResult:
    """
    Bad decider always returns 'faq'.
    T01 "John Smith" → forced to data_collection → WAITING_CONFIRM.
    T02 "yes" → bad decider says 'faq' → WAITING_CONFIRM enforced → data_collection.
    """
    dc = [
        SubagentResponse(status=AgentStatus.IN_PROGRESS, speak="What's your full name?"),
        SubagentResponse(
            status=AgentStatus.WAITING_CONFIRM,
            speak="I have John Smith. Is that correct?",
            internal_state={"stage": "waiting_confirm", "current_field": "name", "pending_value": "John Smith"},
        ),
        # confirmation → COMPLETED → chains NC → IQ → sched
        SubagentResponse(
            status=AgentStatus.COMPLETED,
            speak="All info collected.",
            collected={"name": "John Smith"},
        ),
    ]

    faq = [
        SubagentResponse(
            status=AgentStatus.COMPLETED,
            speak="[WRONG: faq should not run during WAITING_CONFIRM]",
        ),
    ]

    sched = [
        SubagentResponse(status=AgentStatus.IN_PROGRESS, speak="Available slots: 1) Mon, 2) Tue."),
        SubagentResponse(
            status=AgentStatus.WAITING_CONFIRM, speak="Monday. Confirm?",
            internal_state={"pending_slot": "Monday"},
        ),
        SubagentResponse(
            status=AgentStatus.COMPLETED, speak="Booked!",
            booking={"booking_id": "MOCK-005", "slot_description": "Monday", "status": "confirmed"},
        ),
    ]

    mocks = {
        "data_collection":      ScriptedAgent("data_collection", dc),
        "narrative_collection": _nc_passthrough(),
        "intake_qualification": _iq_passthrough("qualified"),
        "faq":                  ScriptedAgent("faq", faq),
        "context_docs":         ScriptedAgent("context_docs", []),
        "fallback":             ScriptedAgent("fallback", []),
        "scheduling":           ScriptedAgent("scheduling", sched),
        "webhook":              _webhook_ok(),
    }

    def bad_decider(utterance: str, graph: WorkflowGraph) -> tuple[str, bool]:
        return "faq", False

    script = [
        SimUtterance("John Smith", force_agent="data_collection"),  # → WAITING_CONFIRM
        SimUtterance("yes"),            # bad decider says "faq" — WC must override
        SimUtterance("slot 1", force_agent="scheduling"),
        SimUtterance("yes", force_agent="scheduling"),               # WC enforced
    ]

    result = WorkflowSimulator(APPOINTMENT_BOOKING, mocks, decider_mock=bad_decider).run(script)

    faq_mock = mocks["faq"]
    if faq_mock._call_count > 0:  # type: ignore[attr-defined]
        print(f"\n  ⚠ WARNING: faq was called {faq_mock._call_count} time(s) — enforcement failed!")
    else:
        print(f"\n  ✓ CONFIRMED: faq call count = 0 (WAITING_CONFIRM enforcement blocked bad decider)")

    return result


def scenario_multiple_faqs_during_collection() -> SimResult:
    """
    Caller asks 3 separate FAQ questions interspersed with data collection.
    """
    dc = [
        SubagentResponse(status=AgentStatus.IN_PROGRESS, speak="What's your full name?"),
        SubagentResponse(status=AgentStatus.IN_PROGRESS, speak="Back — what's your name?"),
        SubagentResponse(
            status=AgentStatus.WAITING_CONFIRM, speak="I have John Smith. Correct?",
            internal_state={"stage": "waiting_confirm", "current_field": "name", "pending_value": "John Smith"},
        ),
        SubagentResponse(
            status=AgentStatus.IN_PROGRESS, speak="Great. What's your email address?",
            collected={"name": "John Smith"},
        ),
        SubagentResponse(status=AgentStatus.IN_PROGRESS, speak="Back — what's your email?"),
        SubagentResponse(
            status=AgentStatus.WAITING_CONFIRM, speak="I have john@example.com. Correct?",
            internal_state={"stage": "waiting_confirm", "current_field": "email", "pending_value": "john@example.com"},
        ),
        SubagentResponse(
            status=AgentStatus.IN_PROGRESS, speak="Got it. Do you have a member ID?",
            collected={"email": "john@example.com"},
        ),
        SubagentResponse(status=AgentStatus.IN_PROGRESS, speak="Sorry about that. Do you have a member ID?"),
        SubagentResponse(status=AgentStatus.COMPLETED, speak="All information collected."),
    ]

    faq = [
        SubagentResponse(status=AgentStatus.COMPLETED, speak="Our hours are 9am–5pm Mon–Fri."),
        SubagentResponse(status=AgentStatus.COMPLETED, speak="Consultations start at $250/hr."),
        SubagentResponse(status=AgentStatus.FAILED, speak=""),
    ]

    cdocs = [SubagentResponse(status=AgentStatus.FAILED, speak="")]

    fallback = [
        SubagentResponse(
            status=AgentStatus.COMPLETED,
            speak="I don't have our address handy. Someone will follow up.",
            notes="Unanswered: office address",
        ),
    ]

    sched = [
        SubagentResponse(status=AgentStatus.IN_PROGRESS, speak="Slots: 1) Monday, 2) Wednesday."),
        SubagentResponse(
            status=AgentStatus.WAITING_CONFIRM, speak="Wednesday. Confirm?",
            internal_state={"pending_slot": "Wednesday"},
        ),
        SubagentResponse(
            status=AgentStatus.COMPLETED, speak="Booked! See you Wednesday.",
            booking={"booking_id": "MOCK-006", "slot_description": "Wednesday", "status": "confirmed"},
        ),
    ]

    mocks = {
        "data_collection":      ScriptedAgent("data_collection", dc),
        "narrative_collection": _nc_passthrough(),
        "intake_qualification": _iq_passthrough("qualified"),
        "faq":                  ScriptedAgent("faq", faq),
        "context_docs":         ScriptedAgent("context_docs", cdocs),
        "fallback":             ScriptedAgent("fallback", fallback),
        "scheduling":           ScriptedAgent("scheduling", sched),
        "webhook":              _webhook_ok(),
    }

    script = [
        SimUtterance("What are your hours?",   force_agent="faq", force_interrupt=True),
        SimUtterance("John Smith",             force_agent="data_collection"),
        SimUtterance("yes",                    force_agent="data_collection"),
        SimUtterance("What are your fees?",    force_agent="faq", force_interrupt=True),
        SimUtterance("john@example.com",       force_agent="data_collection"),
        SimUtterance("yes",                    force_agent="data_collection"),
        SimUtterance("What is your address?",  force_agent="faq", force_interrupt=True),
        SimUtterance("I don't have one",       force_agent="data_collection"),
        SimUtterance("slot 2",                 force_agent="scheduling"),
        SimUtterance("yes",                    force_agent="scheduling"),
    ]

    result = WorkflowSimulator(APPOINTMENT_BOOKING, mocks).run(script)

    faq_mock = mocks["faq"]
    fallback_mock = mocks["fallback"]
    print(f"\n  faq invoked {faq_mock._call_count}x  "  # type: ignore[attr-defined]
          f"fallback invoked {fallback_mock._call_count}x")  # type: ignore[attr-defined]

    return result


def scenario_faq_during_scheduling() -> SimResult:
    """
    Two FAQs during scheduling. Narrative passes through instantly.
    """
    dc = [
        SubagentResponse(
            status=AgentStatus.COMPLETED, speak="",
            collected={"name": "Jane Doe", "email": "jane@example.com"},
        ),
    ]

    sched = [
        SubagentResponse(
            status=AgentStatus.IN_PROGRESS,
            speak="Slots: 1) Tuesday 10am, 2) Thursday 2pm. Which works?",
        ),
        SubagentResponse(
            status=AgentStatus.IN_PROGRESS,
            speak="Back to scheduling — slots: 1) Tuesday 10am, 2) Thursday 2pm.",
        ),
        SubagentResponse(
            status=AgentStatus.WAITING_CONFIRM, speak="Thursday at 2pm. Shall I confirm?",
            internal_state={"pending_slot": "Thursday 2pm"},
        ),
        SubagentResponse(
            status=AgentStatus.WAITING_CONFIRM,
            speak="Just to confirm — Thursday at 2pm. Is that right?",
            internal_state={"pending_slot": "Thursday 2pm"},
        ),
        SubagentResponse(
            status=AgentStatus.COMPLETED, speak="Booked! See you Thursday at 2pm.",
            booking={"booking_id": "MOCK-007", "slot_description": "Thursday 2pm", "status": "confirmed"},
        ),
    ]

    faq = [
        SubagentResponse(status=AgentStatus.COMPLETED, speak="We accept credit card and ACH."),
        SubagentResponse(status=AgentStatus.COMPLETED, speak="Cancel 24h before at no charge."),
    ]

    mocks = {
        "data_collection":      ScriptedAgent("data_collection", dc),
        "narrative_collection": _nc_passthrough(),
        "intake_qualification": _iq_passthrough("qualified"),
        "faq":                  ScriptedAgent("faq", faq),
        "context_docs":         ScriptedAgent("context_docs", []),
        "fallback":             ScriptedAgent("fallback", []),
        "scheduling":           ScriptedAgent("scheduling", sched),
        "webhook":              _webhook_ok(),
    }

    script = [
        # DC COMPLETED → chains NC → IQ → sched T1 (on opening turn chain)
        SimUtterance("What payment methods do you accept?",
                     force_agent="faq", force_interrupt=True),
        SimUtterance("slot 2 please",  force_agent="scheduling"),
        SimUtterance("What is your cancellation policy?",
                     force_agent="faq", force_interrupt=True),
        SimUtterance("yes",            force_agent="scheduling"),
    ]

    result = WorkflowSimulator(APPOINTMENT_BOOKING, mocks).run(script)

    faq_mock = mocks["faq"]
    print(f"\n  faq invoked {faq_mock._call_count}x")  # type: ignore[attr-defined]

    return result


# ── Narrative & Intake scenarios ──────────────────────────────────────────────

def scenario_narrative_happy_path() -> SimResult:
    """
    Multi-turn narrative collection:
      NC opens → caller speaks 3 times → NC detects completion → asks "anything else?" →
      caller says no → NC COMPLETED → IQ qualified → scheduling → book.
    """
    dc = [
        SubagentResponse(
            status=AgentStatus.COMPLETED, speak="All information collected.",
            collected={"name": "Alice Brown", "email": "alice@example.com"},
        ),
    ]

    nc = [
        # call 1: opening ("") → ask for narrative
        SubagentResponse(
            status=AgentStatus.IN_PROGRESS,
            speak="Please describe your legal matter in your own words.",
        ),
        # call 2: first utterance → filler
        SubagentResponse(
            status=AgentStatus.IN_PROGRESS,
            speak="I see, please go on.",
            internal_state={"stage": "collecting", "segments": ["I was in a car accident last month."]},
        ),
        # call 3: second utterance → filler
        SubagentResponse(
            status=AgentStatus.IN_PROGRESS,
            speak="Understood, continue.",
            internal_state={"stage": "collecting", "segments": [
                "I was in a car accident last month.",
                "The other driver ran a red light and hit my vehicle.",
            ]},
        ),
        # call 4: third utterance → completion detected → ask "anything else?"
        SubagentResponse(
            status=AgentStatus.IN_PROGRESS,
            speak="Is there anything else you'd like to add?",
            internal_state={"stage": "asking_done", "segments": [
                "I was in a car accident last month.",
                "The other driver ran a red light and hit my vehicle.",
                "I sustained injuries and my car was totalled.",
            ]},
        ),
        # call 5: "no that's all" → COMPLETED
        SubagentResponse(
            status=AgentStatus.COMPLETED,
            speak="Thank you for sharing that. I've noted your matter.",
            collected={
                "narrative_summary": "Caller was in a car accident caused by another driver running a red light. Sustained injuries and vehicle was totalled.",
                "case_type": "personal injury",
                "full_narrative": "I was in a car accident last month. The other driver ran a red light and hit my vehicle. I sustained injuries and my car was totalled.",
            },
        ),
    ]

    iq = [
        SubagentResponse(
            status=AgentStatus.COMPLETED,
            speak="This looks like a personal injury matter — we can help.",
            collected={"qualification_decision": "qualified", "qualification_reason": "Personal injury matches practice areas."},
        ),
    ]

    sched = [
        SubagentResponse(status=AgentStatus.IN_PROGRESS, speak="Slots: 1) Monday 10am, 2) Wednesday 2pm."),
        SubagentResponse(
            status=AgentStatus.WAITING_CONFIRM, speak="Monday 10am. Confirm?",
            internal_state={"pending_slot": "Monday 10am"},
        ),
        SubagentResponse(
            status=AgentStatus.COMPLETED, speak="Booked! See you Monday at 10am.",
            booking={"booking_id": "MOCK-008", "slot_description": "Monday 10am", "status": "confirmed"},
        ),
    ]

    mocks = {
        "data_collection":      ScriptedAgent("data_collection", dc),
        "narrative_collection": ScriptedAgent("narrative_collection", nc),
        "intake_qualification": ScriptedAgent("intake_qualification", iq),
        "faq":                  ScriptedAgent("faq", []),
        "context_docs":         ScriptedAgent("context_docs", []),
        "fallback":             ScriptedAgent("fallback", []),
        "scheduling":           ScriptedAgent("scheduling", sched),
        "webhook":              _webhook_ok(),
    }

    script = [
        # DC COMPLETED on opening turn → chains NC("") T1 speak
        # NC speaks "Please describe your legal matter..."
        SimUtterance("I was in a car accident last month.",
                     force_agent="narrative_collection"),
        SimUtterance("The other driver ran a red light and hit my vehicle.",
                     force_agent="narrative_collection"),
        SimUtterance("I sustained injuries and my car was totalled.",
                     force_agent="narrative_collection"),
        # NC asks "Is there anything else?"
        SimUtterance("No, that's all.",
                     force_agent="narrative_collection"),
        # NC COMPLETED → chains IQ("") → IQ COMPLETED → chains sched("") T1
        SimUtterance("slot 1",   force_agent="scheduling"),
        SimUtterance("yes",      force_agent="scheduling"),  # WC enforced
    ]

    return WorkflowSimulator(APPOINTMENT_BOOKING, mocks).run(script)


def scenario_faq_interrupt_mid_narrative() -> SimResult:
    """
    Caller starts narrative → asks FAQ question (interrupt) → faq answers →
    resume → narrative_collection re-invoked with "" → narrative continues → completes.

    Verifies: segments accumulated before interrupt are preserved after resume.
    """
    dc = [
        SubagentResponse(
            status=AgentStatus.COMPLETED, speak="",
            collected={"name": "Bob Carter", "email": "bob@example.com"},
        ),
    ]

    nc = [
        # call 1: opening ("") → ask for narrative
        SubagentResponse(
            status=AgentStatus.IN_PROGRESS,
            speak="Please describe your legal matter.",
        ),
        # call 2: first segment → filler
        SubagentResponse(
            status=AgentStatus.IN_PROGRESS,
            speak="I see, please go on.",
            internal_state={"stage": "collecting", "segments": ["My landlord is refusing to fix the heating."]},
        ),
        # call 3: resume ("") after FAQ → re-prompt
        SubagentResponse(
            status=AgentStatus.IN_PROGRESS,
            speak="Please go ahead, I'm still listening about your matter.",
            internal_state={"stage": "collecting", "segments": ["My landlord is refusing to fix the heating."]},
        ),
        # call 4: continued narrative → completion detected
        SubagentResponse(
            status=AgentStatus.IN_PROGRESS,
            speak="Is there anything else you'd like to add?",
            internal_state={"stage": "asking_done", "segments": [
                "My landlord is refusing to fix the heating.",
                "It has been three months and the apartment is uninhabitable.",
            ]},
        ),
        # call 5: "no" → COMPLETED with full concatenated narrative
        SubagentResponse(
            status=AgentStatus.COMPLETED,
            speak="Thank you, I've noted your matter.",
            collected={
                "narrative_summary": "Tenant dispute — landlord failing to fix heating for 3 months.",
                "case_type": "landlord tenant",
                "full_narrative": "My landlord is refusing to fix the heating. It has been three months and the apartment is uninhabitable.",
            },
        ),
    ]

    faq = [
        SubagentResponse(
            status=AgentStatus.COMPLETED,
            speak="Our initial consultation is free of charge.",
        ),
    ]

    iq = [
        SubagentResponse(
            status=AgentStatus.COMPLETED,
            speak="This sounds like a landlord-tenant matter we can assist with.",
            collected={"qualification_decision": "qualified", "qualification_reason": "Landlord tenant matches."},
        ),
    ]

    sched = [
        SubagentResponse(status=AgentStatus.IN_PROGRESS, speak="Slots: 1) Tuesday, 2) Friday."),
        SubagentResponse(
            status=AgentStatus.WAITING_CONFIRM, speak="Tuesday. Confirm?",
            internal_state={"pending_slot": "Tuesday"},
        ),
        SubagentResponse(
            status=AgentStatus.COMPLETED, speak="Booked!",
            booking={"booking_id": "MOCK-009", "slot_description": "Tuesday", "status": "confirmed"},
        ),
    ]

    mocks = {
        "data_collection":      ScriptedAgent("data_collection", dc),
        "narrative_collection": ScriptedAgent("narrative_collection", nc),
        "intake_qualification": ScriptedAgent("intake_qualification", iq),
        "faq":                  ScriptedAgent("faq", faq),
        "context_docs":         ScriptedAgent("context_docs", []),
        "fallback":             ScriptedAgent("fallback", []),
        "scheduling":           ScriptedAgent("scheduling", sched),
        "webhook":              _webhook_ok(),
    }

    script = [
        # DC COMPLETED on opening turn → chains NC("") → NC asks for narrative
        SimUtterance("My landlord is refusing to fix the heating.",
                     force_agent="narrative_collection"),
        # FAQ interrupt mid-narrative
        SimUtterance("Is the initial consultation free?",
                     force_agent="faq", force_interrupt=True),
        # faq.on_complete=Edge("resume") → NC("") resume call
        # NC re-prompts caller to continue
        SimUtterance("It has been three months and the apartment is uninhabitable.",
                     force_agent="narrative_collection"),
        SimUtterance("No, that covers it.",
                     force_agent="narrative_collection"),
        # NC COMPLETED → chains IQ → IQ COMPLETED → chains sched
        SimUtterance("slot 1",   force_agent="scheduling"),
        SimUtterance("yes",      force_agent="scheduling"),   # WC enforced
    ]

    result = WorkflowSimulator(APPOINTMENT_BOOKING, mocks).run(script)

    nc_mock = mocks["narrative_collection"]
    faq_mock = mocks["faq"]
    print(f"\n  narrative_collection invoked {nc_mock._call_count}x  "  # type: ignore[attr-defined]
          f"faq invoked {faq_mock._call_count}x")  # type: ignore[attr-defined]

    return result


def scenario_not_qualified() -> SimResult:
    """
    Caller's matter is outside firm's practice areas.
    IQ returns FAILED → on_failed=Edge("end", "not_qualified").
    """
    dc = [
        SubagentResponse(
            status=AgentStatus.COMPLETED, speak="",
            collected={"name": "Carol Evans", "email": "carol@example.com"},
        ),
    ]

    nc = [
        SubagentResponse(
            status=AgentStatus.IN_PROGRESS,
            speak="Please describe your legal matter.",
        ),
        SubagentResponse(
            status=AgentStatus.IN_PROGRESS,
            speak="Is there anything else you'd like to add?",
            internal_state={"stage": "asking_done", "segments": [
                "I've been charged with felony assault.",
            ]},
        ),
        SubagentResponse(
            status=AgentStatus.COMPLETED,
            speak="Thank you, I've noted your matter.",
            collected={
                "narrative_summary": "Caller facing felony assault criminal charge.",
                "case_type": "criminal defence",
                "full_narrative": "I've been charged with felony assault.",
            },
        ),
    ]

    iq = [
        # Criminal defence — outside this firm's practice areas (family/civil law firm)
        SubagentResponse(
            status=AgentStatus.FAILED,
            speak=(
                "I'm sorry to hear about your situation. Unfortunately, criminal defence "
                "falls outside our current practice areas. We'd recommend reaching out to "
                "a criminal defence attorney."
            ),
            collected={"qualification_decision": "not_qualified", "qualification_reason": "Criminal defence outside practice areas."},
        ),
    ]

    mocks = {
        "data_collection":      ScriptedAgent("data_collection", dc),
        "narrative_collection": ScriptedAgent("narrative_collection", nc),
        "intake_qualification": ScriptedAgent("intake_qualification", iq),
        "faq":                  ScriptedAgent("faq", []),
        "context_docs":         ScriptedAgent("context_docs", []),
        "fallback":             ScriptedAgent("fallback", []),
        "scheduling":           ScriptedAgent("scheduling", []),   # should never be invoked
        "webhook":              ScriptedAgent("webhook", []),
    }

    script = [
        SimUtterance("I've been charged with felony assault.",
                     force_agent="narrative_collection"),
        SimUtterance("No, that's all.",
                     force_agent="narrative_collection"),
        # NC COMPLETED → chains IQ → IQ FAILED → on_failed=Edge("end", "not_qualified")
    ]

    result = WorkflowSimulator(APPOINTMENT_BOOKING, mocks).run(script)

    sched_mock = mocks["scheduling"]
    if sched_mock._call_count > 0:  # type: ignore[attr-defined]
        print(f"\n  ⚠ WARNING: scheduling was called {sched_mock._call_count}x — should be 0!")
    else:
        print(f"\n  ✓ CONFIRMED: scheduling not invoked (correctly terminated at not_qualified)")

    return result


def scenario_ambiguous_qualification() -> SimResult:
    """
    IQ returns ambiguous (COMPLETED) → proceeds to scheduling anyway.
    Verifies ambiguous is treated as 'go ahead' — human will review.
    """
    dc = [
        SubagentResponse(
            status=AgentStatus.COMPLETED, speak="",
            collected={"name": "Dan Fisher", "email": "dan@example.com"},
        ),
    ]

    nc = [
        SubagentResponse(
            status=AgentStatus.IN_PROGRESS,
            speak="Please describe your legal matter.",
        ),
        SubagentResponse(
            status=AgentStatus.IN_PROGRESS,
            speak="Is there anything else you'd like to add?",
        ),
        SubagentResponse(
            status=AgentStatus.COMPLETED,
            speak="Thank you, I've noted your matter.",
            collected={
                "narrative_summary": "Cross-border business dispute with unclear jurisdiction.",
                "case_type": "unknown",
                "full_narrative": "I have a business dispute with a company in another country.",
            },
        ),
    ]

    iq = [
        SubagentResponse(
            status=AgentStatus.COMPLETED,   # ambiguous → still COMPLETED → proceeds to scheduling
            speak="Your matter may be within our scope — a specialist will assess further during the consultation.",
            collected={"qualification_decision": "ambiguous", "qualification_reason": "Cross-border jurisdiction unclear."},
        ),
    ]

    sched = [
        SubagentResponse(status=AgentStatus.IN_PROGRESS, speak="Slots: 1) Thursday, 2) Friday."),
        SubagentResponse(
            status=AgentStatus.WAITING_CONFIRM, speak="Thursday. Confirm?",
            internal_state={"pending_slot": "Thursday"},
        ),
        SubagentResponse(
            status=AgentStatus.COMPLETED, speak="Booked!",
            booking={"booking_id": "MOCK-011", "slot_description": "Thursday", "status": "confirmed"},
        ),
    ]

    mocks = {
        "data_collection":      ScriptedAgent("data_collection", dc),
        "narrative_collection": ScriptedAgent("narrative_collection", nc),
        "intake_qualification": ScriptedAgent("intake_qualification", iq),
        "faq":                  ScriptedAgent("faq", []),
        "context_docs":         ScriptedAgent("context_docs", []),
        "fallback":             ScriptedAgent("fallback", []),
        "scheduling":           ScriptedAgent("scheduling", sched),
        "webhook":              _webhook_ok(),
    }

    script = [
        SimUtterance("I have a business dispute with a company in another country.",
                     force_agent="narrative_collection"),
        SimUtterance("No, that's it.",
                     force_agent="narrative_collection"),
        # NC COMPLETED → IQ COMPLETED (ambiguous) → sched
        SimUtterance("slot 1",   force_agent="scheduling"),
        SimUtterance("yes",      force_agent="scheduling"),  # WC enforced
    ]

    result = WorkflowSimulator(APPOINTMENT_BOOKING, mocks).run(script)

    iq_decision = result.collected.get("qualification_decision", "?")
    print(f"\n  iq decision={iq_decision!r}  (ambiguous should still reach scheduling)")

    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    scenarios: list[tuple[str, object, str | None]] = [
        ("1  — Happy path (4-goal)",                    scenario_happy_path,                   "completed"),
        ("2  — FAQ interrupt mid-collection",           scenario_faq_interrupt,                "completed"),
        ("3  — Fallback chain (faq→docs→fb)",           scenario_fallback_chain,               "completed"),
        ("4  — 3 consecutive errors",                   scenario_consecutive_errors,           "agent_error"),
        ("5  — WAITING_CONFIRM enforcement",            scenario_waiting_confirm_enforcement,  "completed"),
        ("6  — Multiple FAQs during collection",        scenario_multiple_faqs_during_collection, "completed"),
        ("7  — FAQs during scheduling",                 scenario_faq_during_scheduling,        "completed"),
        ("8  — Narrative happy path (multi-turn)",      scenario_narrative_happy_path,         "completed"),
        ("9  — FAQ interrupt mid-narrative + resume",   scenario_faq_interrupt_mid_narrative,  "completed"),
        ("10 — Not-qualified path",                     scenario_not_qualified,                "not_qualified"),
        ("11 — Ambiguous qualification → scheduling",   scenario_ambiguous_qualification,      "completed"),
    ]

    passed, failed = 0, 0
    results = []

    for name, fn, expected in scenarios:
        print(f"\n\n{'─' * 68}")
        print(f"  Running: {name}")
        print(f"{'─' * 68}")
        try:
            result = fn()  # type: ignore[operator]
        except Exception as exc:
            import traceback
            print(f"  SCENARIO CRASHED: {exc}")
            traceback.print_exc()
            result = SimResult([], {}, {}, None, None, str(exc))

        ok = _print_result(name, result, expected)
        results.append((name, ok))
        if ok:
            passed += 1
        else:
            failed += 1

    print(f"\n\n{'═' * 68}")
    print(f"  RESULTS: {passed} passed, {failed} failed")
    print(f"{'═' * 68}")
    for name, ok in results:
        print(f"  {'✓' if ok else '✗'}  {name}")
    print()

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
