"""
Simulation: Narrative collection with embedded questions.

Captures the routing bug where a caller embeds a rhetorical or social question
inside their narrative (e.g. "I want help with a divorce case, can you help me
with this?") and the router incorrectly routes to the FAQ chain instead of
treating the whole utterance as narrative content.

TWO DISTINCT CASES:
  A) Rhetorical / social embedded question — "can you help me?" is a
     conversational flourish, not a real question. The whole utterance is
     narrative. Expected: router → narrative_collection.

  B) Genuine off-topic question embedded in narrative — "I was in an accident,
     what are your fees?" asks a specific question about firm policies.
     Expected: router → faq (interrupt, then resume).

The simulation exercises the router directly with the graph state set to
"data_collection COMPLETED, narrative_collection IN_PROGRESS" — which is the
state the system is in when a caller starts describing their matter.

Requirements:
    CEREBRAS_API_KEY env var

Run:
    cd data-plane
    PYTHONPATH=. python3 tests/simulate_narrative_embedded_questions.py
    # or via pytest:
    PYTHONPATH=. python3 -m pytest tests/simulate_narrative_embedded_questions.py -v -s
"""
from __future__ import annotations

import sys
import textwrap

import pytest

from app.agents.graph_config import APPOINTMENT_BOOKING
from app.agents.workflow import WorkflowGraph
from app.agents.planner import Planner
from app.agents.registry import build_registry
from app.agents.base import AgentStatus


# ── Nexus Law config ──────────────────────────────────────────────────────────

NEXUS_CONFIG = {
    "assistant": {
        "persona_name": "Aria at Nexus Law",
        "narrative_topic": "your legal matter",
        "system_prompt": (
            "Nexus Law is a personal injury and employment law firm in Los Angeles."
        ),
    },
    "practice_areas": [
        {"name": "Personal Injury", "description": "Car accidents, slip and fall, premises liability."},
        {"name": "Employment Law", "description": "Wrongful termination, discrimination, harassment."},
    ],
    "parameters": [
        {"name": "caller_name", "display_label": "Full Name", "required": True, "field_type": "name"},
        {"name": "phone", "display_label": "Phone Number", "required": True, "field_type": "phone"},
    ],
    "faqs": [
        {"question": "What are your consultation fees?", "answer": "Initial consultations are free."},
        {"question": "Where are you located?", "answer": "We are in downtown Los Angeles."},
        {"question": "Do you work on contingency?", "answer": "Yes, personal injury cases are handled on contingency."},
    ],
    "_collected": {
        "caller_name": "John Smith",
        "phone": "3105551234",
    },
    "_booking": {},
    "_notes": "",
    "_tool_results": {},
}


def _build_graph_mid_narrative() -> WorkflowGraph:
    """
    Return a WorkflowGraph in the state the system is in when a caller begins
    describing their matter: data_collection COMPLETED, empathy COMPLETED,
    narrative_collection IN_PROGRESS.
    Empathy is marked COMPLETED because in a real call it fires on the first
    narrative utterance — before narrative_collection reaches IN_PROGRESS.
    """
    graph = WorkflowGraph(APPOINTMENT_BOOKING)
    graph.states["data_collection"].status = AgentStatus.COMPLETED
    graph.states["empathy"].status = AgentStatus.COMPLETED
    graph.states["narrative_collection"].status = AgentStatus.IN_PROGRESS
    graph.states["narrative_collection"].internal_state = {
        "stage": "collecting",
        "segments": [],
        "summary": None,
        "case_type": None,
    }
    return graph


def _router_decision(utterance: str, history: list[dict] | None = None) -> tuple[str, bool]:
    """
    Run the planner for a single utterance and return (agent_id, interrupt).
    A fresh graph is built for each call — tests are independent.
    interrupt is True when the first step invokes an interrupt-eligible agent.
    """
    graph = _build_graph_mid_narrative()
    planner = Planner(graph)
    history = history or [
        {"role": "assistant", "content": "I'd like to understand your matter. Please go ahead."},
    ]
    steps = planner.plan(utterance, history)
    first_invoke = next((s for s in steps if s.action == "invoke"), None)
    if not first_invoke:
        return "narrative_collection", False
    agent_id = first_invoke.agent_id
    node = graph.nodes.get(agent_id)
    interrupt = bool(node and node.interrupt_eligible)
    return agent_id, interrupt


def _run_agent(agent_id: str, utterance: str) -> tuple[AgentStatus, str]:
    """Invoke the named agent with the given utterance and return (status, speak)."""
    graph = _build_graph_mid_narrative()
    registry = build_registry(call_id="sim-001", transcript=[])
    agent = registry[agent_id]
    state = graph.states[agent_id]
    response = agent.process(
        utterance,
        dict(state.internal_state),
        NEXUS_CONFIG,
        history=[],
    )
    return response.status, response.speak or ""


# ── Test cases ────────────────────────────────────────────────────────────────

# Rhetorical questions: the question IS the narrative opener. The whole utterance
# should be treated as narrative content — NOT an FAQ interrupt.
RHETORICAL_CASES = [
    {
        "label": "Divorce case opener with 'can you help'",
        "utterance": "I want some help with a divorce case, can you help me with this?",
        "expected_agent": "narrative_collection",
        "expected_interrupt": False,
        "description": (
            "Classic bug case. The caller is describing their matter (divorce case) "
            "and appending a social/confirmation question. This is NOT a FAQ question — "
            "it is their narrative opening move."
        ),
    },
    {
        "label": "Wrongful termination with 'is this something you handle'",
        "utterance": "I was fired without any reason last month, is this something you can help with?",
        "expected_agent": "narrative_collection",
        "expected_interrupt": False,
        "description": (
            "Caller describing their employment matter and asking if we handle it. "
            "Should be absorbed as narrative — the practice area check happens at qualification."
        ),
    },
    {
        "label": "Slip and fall with 'can you take my case'",
        "utterance": "I slipped on a wet floor at a grocery store and hurt my back. Can you take my case?",
        "expected_agent": "narrative_collection",
        "expected_interrupt": False,
        "description": (
            "'Can you take my case?' is a social confirmation question embedded in a clear "
            "personal injury narrative. Should continue collecting the narrative."
        ),
    },
    {
        "label": "Mid-narrative pause filler 'does that make sense'",
        "utterance": (
            "My employer was making me work overtime without paying me for it, "
            "and eventually I just couldn't take it anymore and I quit. Does that make sense?"
        ),
        "expected_agent": "narrative_collection",
        "expected_interrupt": False,
        "description": (
            "'Does that make sense?' is a verbal filler with no answer expected. "
            "The entire utterance is rich narrative that should be collected."
        ),
    },
    {
        "label": "Narrative opener with 'am I in the right place'",
        "utterance": "I was in a car accident three weeks ago and I'm still in pain. Am I in the right place?",
        "expected_agent": "narrative_collection",
        "expected_interrupt": False,
        "description": (
            "'Am I in the right place?' is a social reassurance question, not a policy question. "
            "The caller has already opened with clear narrative content."
        ),
    },
]

# Genuine questions: utterances that are PRIMARILY a concrete policy/fee question
# with minimal or no narrative content. These should interrupt to FAQ.
#
# NOTE: Mixed utterances that contain BOTH narrative content AND a question
# (e.g. "I need help with a workplace injury. Where are you located?") are
# handled by the parallel FAQ mechanism: the router correctly routes to
# narrative_collection (narrative priority rule) AND the handler concurrently
# queries the FAQ agent so the caller still gets the answer. Both outcomes
# (faq interrupt OR narrative_collection + parallel FAQ) are valid.
GENUINE_QUESTION_CASES = [
    {
        "label": "Pure fee question with minimal narrative",
        "utterance": "I was in a car accident last Tuesday. What are your consultation fees?",
        "expected_agent": "faq",
        "expected_interrupt": True,
        "description": (
            "Thin narrative opener followed by a concrete fee question. "
            "The question is the clear primary intent — should interrupt to FAQ."
        ),
    },
    {
        "label": "Mixed: narrative + location question (parallel FAQ handles either route)",
        "utterance": "I need help with a workplace injury. Where are you located?",
        # Both faq (interrupt) and narrative_collection (parallel FAQ) are valid:
        # — faq: answers the question, resumes narrative next turn
        # — narrative_collection: collects the narrative segment AND runs parallel
        #   FAQ, so caller still hears "I've noted your matter. We are in downtown LA."
        "expected_agent": "any_interrupt_or_narrative",
        "expected_interrupt": True,
        "description": (
            "Mixed utterance: narrative content + concrete question. Either routing is "
            "acceptable — the parallel FAQ mechanism ensures the question is answered "
            "regardless of which agent the router selects."
        ),
    },
]

# Pure narrative: no question at all. Baseline — should already work correctly.
PURE_NARRATIVE_CASES = [
    {
        "label": "Pure personal injury narrative",
        "utterance": "I was rear-ended on the 405 freeway two weeks ago and I'm having neck and back pain.",
        "expected_agent": "narrative_collection",
        "expected_interrupt": False,
        "description": "Baseline: pure narrative with no embedded question. Should work today.",
    },
    {
        "label": "Pure employment narrative",
        "utterance": "My manager has been making inappropriate comments about my age for six months.",
        "expected_agent": "narrative_collection",
        "expected_interrupt": False,
        "description": "Baseline: pure employment narrative. Should work today.",
    },
]


# ── Simulation runner ─────────────────────────────────────────────────────────

def _print_header(title: str) -> None:
    print(f"\n{'═' * 72}")
    print(f"  {title}")
    print(f"{'═' * 72}")


def _print_case(
    label: str,
    utterance: str,
    description: str,
    actual_agent: str,
    actual_interrupt: bool,
    expected_agent: str,
    expected_interrupt: bool,
) -> bool:
    if expected_agent == "any_interrupt_or_narrative":
        ok = actual_agent in ("faq", "context_docs", "fallback", "narrative_collection")
    else:
        ok = (actual_agent == expected_agent)
    icon = "✓" if ok else "✗"
    print(f"\n  {icon}  {label}")
    print(f"       utterance   : \"{utterance[:80]}{'…' if len(utterance) > 80 else ''}\"")
    print(f"       expected    : agent={expected_agent}  interrupt={expected_interrupt}")
    print(f"       got         : agent={actual_agent}  interrupt={actual_interrupt}")
    if not ok:
        print(f"       [BUG] {description}")
    return ok


def _run_all() -> int:
    passed = failed = 0

    all_groups = [
        ("Rhetorical / Social Embedded Questions  [BUG: currently routes to FAQ]", RHETORICAL_CASES),
        ("Genuine Off-Topic Questions  [Should interrupt to FAQ — existing behaviour]", GENUINE_QUESTION_CASES),
        ("Pure Narrative Baseline  [Should already work]", PURE_NARRATIVE_CASES),
    ]

    for group_title, cases in all_groups:
        _print_header(group_title)
        for case in cases:
            agent_id, interrupt = _router_decision(case["utterance"])
            ok = _print_case(
                label=case["label"],
                utterance=case["utterance"],
                description=case["description"],
                actual_agent=agent_id,
                actual_interrupt=interrupt,
                expected_agent=case["expected_agent"],
                expected_interrupt=case["expected_interrupt"],
            )
            if ok:
                passed += 1
            else:
                failed += 1

    # ── Show the broken experience end-to-end ────────────────────────────────
    _print_header("End-to-End: What the caller actually hears (broken path)")
    bug_case = RHETORICAL_CASES[0]
    print(f"\n  Utterance: \"{bug_case['utterance']}\"")
    print()

    print("  Step 1 — Router selects:")
    agent_id, interrupt = _router_decision(bug_case["utterance"])
    print(f"           → {agent_id}  (interrupt={interrupt})")

    if agent_id != "narrative_collection":
        print("\n  Step 2 — FAQ agent runs (no matching FAQ for 'can you help me?'):")
        status, speak = _run_agent(agent_id, bug_case["utterance"])
        print(f"           status : {status.value}")
        print(f"           speak  : {speak!r:.120}")

        print("\n  Step 3 — FAQ fails → context_docs → fallback chain")
        print(f"           Caller hears a generic non-answer.")
        print()
        print("  Step 4 — Resume stack re-invokes narrative_collection:")
        _, resume_speak = _run_agent("narrative_collection", "")
        print(f"           speak  : {resume_speak!r:.120}")
        print()
        print("  Result: Caller described their matter, got an unhelpful response,")
        print("          then was asked to describe their matter again → CONFUSION")
    else:
        print("  ✓ Router correctly identified this as narrative content.")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'═' * 72}")
    print(f"  RESULTS: {passed} passed, {failed} failed")
    print(f"{'═' * 72}")

    # ── Proposed fix description ──────────────────────────────────────────────
    print()
    print("  PROPOSED FIX (Option 1 — Narrative-aware intent classifier):")
    print()
    print(textwrap.dedent("""\
      Add _classify_utterance() to NarrativeCollectionAgent that uses a single
      focused LLM call to distinguish:

        rhetorical_question — Social/confirmatory question embedded in narrative.
                              ("can you help?", "does that make sense?", "am I in
                              the right place?"). These are part of the narrative.
                              → collect as narrative segment, reply with affirming
                              filler: "Yes, absolutely — please continue."

        genuine_question    — A concrete, answerable question about firm policy,
                              fees, location, process, etc.
                              → return UNHANDLED (existing interrupt chain handles).

        pure_narrative      — No question at all.
                              → existing filler behaviour, no change.

      Why in NarrativeCollectionAgent, not the Router?
        • The agent knows it is in narrative-collection mode.
        • The router should remain stateless and general.
        • One focused LLM call (max_tokens=32) is cheaper than re-prompting the
          full router decider with an extra rule.
        • Easy to unit-test in isolation.
    """))

    return 0 if failed == 0 else 1


# ── pytest wrappers ───────────────────────────────────────────────────────────

class TestRhetoricalEmbeddedQuestions:
    """
    All rhetorical-question utterances must route to narrative_collection.
    This test class documents the BUG — it is expected to FAIL before the fix
    and PASS after.
    """

    @pytest.mark.parametrize("case", RHETORICAL_CASES, ids=[c["label"] for c in RHETORICAL_CASES])
    def test_rhetorical_routes_to_narrative(self, case):
        agent_id, interrupt = _router_decision(case["utterance"])
        assert agent_id == "narrative_collection", (
            f"[BUG] '{case['label']}'\n"
            f"  utterance : {case['utterance']!r}\n"
            f"  expected  : narrative_collection\n"
            f"  got       : {agent_id} (interrupt={interrupt})\n"
            f"  note      : {case['description']}"
        )


class TestGenuineQuestionsInterrupt:
    """
    Genuine off-topic questions must be handled correctly.
    For pure policy questions: interrupt to FAQ.
    For mixed (narrative + question) utterances: either faq OR narrative_collection
    is valid — the parallel FAQ mechanism ensures the question is answered either way.
    """

    @pytest.mark.parametrize(
        "case", GENUINE_QUESTION_CASES, ids=[c["label"] for c in GENUINE_QUESTION_CASES]
    )
    def test_genuine_question_handled(self, case):
        agent_id, interrupt = _router_decision(case["utterance"])
        expected = case["expected_agent"]
        if expected == "any_interrupt_or_narrative":
            # Both routing paths are valid for mixed utterances
            assert agent_id in ("faq", "context_docs", "fallback", "narrative_collection"), (
                f"[REGRESSION] '{case['label']}'\n"
                f"  utterance : {case['utterance']!r}\n"
                f"  expected  : faq/context_docs/fallback/narrative_collection\n"
                f"  got       : {agent_id} (interrupt={interrupt})"
            )
        else:
            assert agent_id in ("faq", "context_docs", "fallback"), (
                f"[REGRESSION] '{case['label']}'\n"
                f"  utterance : {case['utterance']!r}\n"
                f"  expected  : faq/context_docs/fallback\n"
                f"  got       : {agent_id} (interrupt={interrupt})"
            )


class TestPureNarrativeBaseline:
    """Pure narrative utterances must route to narrative_collection. Baseline check."""

    @pytest.mark.parametrize(
        "case", PURE_NARRATIVE_CASES, ids=[c["label"] for c in PURE_NARRATIVE_CASES]
    )
    def test_pure_narrative_routes_correctly(self, case):
        agent_id, interrupt = _router_decision(case["utterance"])
        assert agent_id == "narrative_collection", (
            f"[REGRESSION] '{case['label']}'\n"
            f"  utterance : {case['utterance']!r}\n"
            f"  expected  : narrative_collection\n"
            f"  got       : {agent_id}"
        )


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.exit(_run_all())
