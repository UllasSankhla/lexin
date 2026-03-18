#!/usr/bin/env python3
"""
Workflow simulation harness — runs WorkflowDefinition edge logic without real calls.

Usage:
    cd data-plane
    PYTHONPATH=. python3 simulate.py

Scenarios:
    1. Happy path — name, email, skip optional field, scheduling, booking
    2. FAQ interrupt mid-collection → resume → collection continues
    3. Fallback chain — faq fails → context_docs fails → fallback
    4. 3 consecutive agent errors → finalize("agent_error")
    5. WAITING_CONFIRM enforcement — bad decider overridden by hard check
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
            MAX_CHAIN = 5

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
        print(f"    {nid:20s} {status}")
    if not ok:
        print(f"\n  ASSERTION FAILED: expected {expected_reason!r}, got {result.finalization_reason!r}")
    return ok


# ── Scenario builders ─────────────────────────────────────────────────────────

def _dc(*args) -> SubagentResponse:
    """Shorthand: SubagentResponse for data_collection."""
    return SubagentResponse(*args)


def scenario_happy_path() -> SimResult:
    """
    Full happy path:
      name → WAITING_CONFIRM → confirm → email → WAITING_CONFIRM → confirm →
      skip optional member ID (COMPLETED) → chain to scheduling →
      pick slot → WAITING_CONFIRM → confirm → booked → end("completed")
    """
    dc = [
        # call 1: opening "" → ask first field
        SubagentResponse(status=AgentStatus.IN_PROGRESS, speak="What's your full name?"),
        # call 2: "John Smith" → spell back, wait for confirm
        SubagentResponse(
            status=AgentStatus.WAITING_CONFIRM,
            speak="I have your name as John Smith. Is that correct?",
            internal_state={"stage": "waiting_confirm", "current_field": "name", "pending_value": "John Smith"},
        ),
        # call 3: "yes" (WAITING_CONFIRM enforced) → collect name, ask email
        SubagentResponse(
            status=AgentStatus.IN_PROGRESS,
            speak="Got it. What's your email address?",
            collected={"name": "John Smith"},
        ),
        # call 4: "john@example.com" → spell back
        SubagentResponse(
            status=AgentStatus.WAITING_CONFIRM,
            speak="I have john@example.com. Is that correct?",
            internal_state={"stage": "waiting_confirm", "current_field": "email", "pending_value": "john@example.com"},
        ),
        # call 5: "yes" (WAITING_CONFIRM enforced) → collect email, ask optional
        SubagentResponse(
            status=AgentStatus.IN_PROGRESS,
            speak="Great. Do you have a member ID? You can skip this if not.",
            collected={"email": "john@example.com"},
        ),
        # call 6: "I don't have one" → skip optional, all fields done → COMPLETED
        SubagentResponse(
            status=AgentStatus.COMPLETED,
            speak="No problem, I'll note no member ID.",
        ),
    ]

    sched = [
        # call 1: chained from data_collection.on_complete ("")
        SubagentResponse(
            status=AgentStatus.IN_PROGRESS,
            speak="I have these slots: 1) Monday 2pm, 2) Tuesday 3pm. Which works?",
        ),
        # call 2: "the second one"
        SubagentResponse(
            status=AgentStatus.WAITING_CONFIRM,
            speak="Tuesday at 3pm. Shall I confirm that booking?",
            internal_state={"pending_slot": "Tuesday 3pm"},
        ),
        # call 3: "yes" (WAITING_CONFIRM enforced)
        SubagentResponse(
            status=AgentStatus.COMPLETED,
            speak="You're all set for Tuesday at 3pm!",
            booking={"booking_id": "MOCK-001", "slot_description": "Tuesday at 3pm", "status": "confirmed"},
        ),
    ]

    mocks = {
        "data_collection": ScriptedAgent("data_collection", dc),
        "scheduling":      ScriptedAgent("scheduling", sched),
        "faq":             ScriptedAgent("faq", []),
        "context_docs":    ScriptedAgent("context_docs", []),
        "fallback":        ScriptedAgent("fallback", []),
        "webhook":         ScriptedAgent("webhook", [
            SubagentResponse(status=AgentStatus.COMPLETED, speak=""),
        ]),
    }

    # force_agent is overridden by WAITING_CONFIRM enforcement on "yes" turns
    script = [
        SimUtterance("John Smith",         force_agent="data_collection"),
        SimUtterance("yes",                force_agent="data_collection"),  # WC enforced
        SimUtterance("john@example.com",   force_agent="data_collection"),
        SimUtterance("yes",                force_agent="data_collection"),  # WC enforced
        SimUtterance("I don't have one",   force_agent="data_collection"),  # → chains to scheduling
        SimUtterance("the second one",     force_agent="scheduling"),
        SimUtterance("yes",                force_agent="scheduling"),        # WC enforced
    ]

    return WorkflowSimulator(APPOINTMENT_BOOKING, mocks).run(script)


def scenario_faq_interrupt() -> SimResult:
    """
    name collected → caller asks FAQ question (force_interrupt) →
    faq answers → on_complete=Edge("resume") → data_collection re-invoked with "" →
    collection continues → schedules → books.
    """
    dc = [
        SubagentResponse(status=AgentStatus.IN_PROGRESS, speak="What's your full name?"),
        SubagentResponse(
            status=AgentStatus.WAITING_CONFIRM,
            speak="I have John Smith. Is that correct?",
            internal_state={"stage": "waiting_confirm", "current_field": "name", "pending_value": "John Smith"},
        ),
        # "yes" → collect name, ask email
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
        "data_collection": ScriptedAgent("data_collection", dc),
        "faq":             ScriptedAgent("faq", faq),
        "context_docs":    ScriptedAgent("context_docs", []),
        "fallback":        ScriptedAgent("fallback", []),
        "scheduling":      ScriptedAgent("scheduling", sched),
        "webhook":         ScriptedAgent("webhook", [
            SubagentResponse(status=AgentStatus.COMPLETED, speak=""),
        ]),
    }

    script = [
        SimUtterance("John Smith",            force_agent="data_collection"),
        SimUtterance("yes",                   force_agent="data_collection"),    # WC enforced
        SimUtterance("What are your hours?",  force_agent="faq", force_interrupt=True),
        # faq.on_complete = Edge("resume") → data_collection("") re-invoked within same turn
        SimUtterance("john@example.com",      force_agent="data_collection"),
        SimUtterance("yes",                   force_agent="data_collection"),    # WC enforced
        SimUtterance("I don't have one",      force_agent="data_collection"),    # → chains to sched
        SimUtterance("slot 1",                force_agent="scheduling"),
        SimUtterance("yes",                   force_agent="scheduling"),         # WC enforced
    ]

    return WorkflowSimulator(APPOINTMENT_BOOKING, mocks).run(script)


def scenario_fallback_chain() -> SimResult:
    """
    Caller asks something not in FAQ, not in docs → faq.on_failed→context_docs →
    context_docs.on_failed→fallback → fallback.on_complete→resume →
    data_collection continues.
    """
    dc = [
        SubagentResponse(status=AgentStatus.IN_PROGRESS, speak="What's your full name?"),
        # resume invocation after fallback chain
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

    faq = [
        SubagentResponse(status=AgentStatus.FAILED, speak=""),          # no match
    ]

    cdocs = [
        SubagentResponse(status=AgentStatus.FAILED, speak=""),          # no relevant docs
    ]

    fallback = [
        SubagentResponse(
            status=AgentStatus.COMPLETED,
            speak="I don't have that info right now. Someone will follow up.",
            notes="Unanswered: parking options",
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
        "data_collection": ScriptedAgent("data_collection", dc),
        "faq":             ScriptedAgent("faq", faq),
        "context_docs":    ScriptedAgent("context_docs", cdocs),
        "fallback":        ScriptedAgent("fallback", fallback),
        "scheduling":      ScriptedAgent("scheduling", sched),
        "webhook":         ScriptedAgent("webhook", [
            SubagentResponse(status=AgentStatus.COMPLETED, speak=""),
        ]),
    }

    script = [
        # faq interrupt → faq FAILED → chains to context_docs → FAILED → fallback → COMPLETED → resume
        SimUtterance("What are your parking options?", force_agent="faq", force_interrupt=True),
        SimUtterance("Jane Doe",   force_agent="data_collection"),
        SimUtterance("yes",        force_agent="data_collection"),    # WC enforced
        SimUtterance("skip",       force_agent="data_collection"),    # → COMPLETED → chains to sched
        SimUtterance("slot 1",     force_agent="scheduling"),
        SimUtterance("yes",        force_agent="scheduling"),         # WC enforced
    ]

    return WorkflowSimulator(APPOINTMENT_BOOKING, mocks).run(script)


def scenario_consecutive_errors() -> SimResult:
    """
    Every agent invocation raises. After 3 consecutive errors → finalize("agent_error").
    Opening turn counts as error #1; script provides 2 more.
    """
    mocks = {k: ErrorAgent() for k in
             ("data_collection", "scheduling", "faq", "context_docs", "fallback", "webhook")}

    script = [
        SimUtterance("anything", force_agent="data_collection"),   # error #2
        SimUtterance("anything", force_agent="data_collection"),   # error #3 → finalize
        SimUtterance("anything", force_agent="data_collection"),   # should not run
    ]

    return WorkflowSimulator(APPOINTMENT_BOOKING, mocks).run(script)


def scenario_waiting_confirm_enforcement() -> SimResult:
    """
    A bad decider always returns 'faq'.
    T01 "John Smith" → forced to data_collection → WAITING_CONFIRM.
    T02 "yes" → bad decider says 'faq' → WAITING_CONFIRM hard check overrides → data_collection.
    Verifies T02 routes to data_collection, not faq.
    """
    dc = [
        SubagentResponse(status=AgentStatus.IN_PROGRESS, speak="What's your full name?"),
        SubagentResponse(
            status=AgentStatus.WAITING_CONFIRM,
            speak="I have John Smith. Is that correct?",
            internal_state={"stage": "waiting_confirm", "current_field": "name", "pending_value": "John Smith"},
        ),
        # confirmation "yes" → COMPLETED → chains to scheduling
        SubagentResponse(
            status=AgentStatus.COMPLETED,
            speak="All info collected.",
            collected={"name": "John Smith"},
        ),
    ]

    faq = [
        # Should NOT be consumed during the "yes" confirmation turn
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
        "data_collection": ScriptedAgent("data_collection", dc),
        "faq":             ScriptedAgent("faq", faq),
        "context_docs":    ScriptedAgent("context_docs", []),
        "fallback":        ScriptedAgent("fallback", []),
        "scheduling":      ScriptedAgent("scheduling", sched),
        "webhook":         ScriptedAgent("webhook", [
            SubagentResponse(status=AgentStatus.COMPLETED, speak=""),
        ]),
    }

    def bad_decider(utterance: str, graph: WorkflowGraph) -> tuple[str, bool]:
        return "faq", False

    script = [
        SimUtterance("John Smith", force_agent="data_collection"),  # triggers WAITING_CONFIRM
        SimUtterance("yes"),            # bad decider says "faq" — WAITING_CONFIRM must override
        SimUtterance("slot 1", force_agent="scheduling"),
        SimUtterance("yes", force_agent="scheduling"),               # WC enforced
    ]

    result = WorkflowSimulator(APPOINTMENT_BOOKING, mocks, decider_mock=bad_decider).run(script)

    # Additional assertion: verify faq was never invoked
    faq_mock = mocks["faq"]
    if faq_mock._call_count > 0:  # type: ignore[attr-defined]
        print(f"\n  ⚠ WARNING: faq was called {faq_mock._call_count} time(s) — enforcement failed!")
    else:
        print(f"\n  ✓ CONFIRMED: faq call count = 0 (WAITING_CONFIRM enforcement blocked bad decider)")

    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    scenarios: list[tuple[str, object, str | None]] = [
        ("1 — Happy path",                   scenario_happy_path,                "completed"),
        ("2 — FAQ interrupt + resume",        scenario_faq_interrupt,             "completed"),
        ("3 — Fallback chain (faq→docs→fb)",  scenario_fallback_chain,            "completed"),
        ("4 — 3 consecutive errors",          scenario_consecutive_errors,        "agent_error"),
        ("5 — WAITING_CONFIRM enforcement",   scenario_waiting_confirm_enforcement,"completed"),
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
