"""
Live LLM simulation for IntakeQualificationAgent.

Verifies that the LLM consistently rejects a divorce case given Nexus Law's
two practice areas (Personal Injury and Employment Law only).

Requires:
    CEREBRAS_API_KEY env var set.

Run:
    cd data-plane
    PYTHONPATH=. python3 -m pytest tests/simulate_intake_qualification.py -v -s
    # or directly:
    PYTHONPATH=. python3 tests/simulate_intake_qualification.py
"""
from __future__ import annotations

import sys

import pytest

from app.agents.intake_qualification import IntakeQualificationAgent
from app.agents.base import AgentStatus


# ── Nexus Law config — Personal Injury + Employment Law only ──────────────────

NEXUS_PRACTICE_AREAS = [
    {
        "name": "Personal Injury",
        "description": "Car accidents, slip and fall, premises liability, wrongful death.",
        "qualification_criteria": (
            "Caller suffered physical injury due to another party's negligence in California."
        ),
        "disqualification_signals": (
            "Matters outside California, pure property damage, criminal charges, "
            "family law, divorce, child custody, estate planning, immigration."
        ),
        "ambiguous_signals": "Unclear whether an injury-causing event occurred.",
        "referral_suggestion": None,
    },
    {
        "name": "Employment Law",
        "description": "Wrongful termination, discrimination, harassment, unpaid wages.",
        "qualification_criteria": (
            "Caller experienced workplace wrongdoing by an employer or co-worker in California."
        ),
        "disqualification_signals": (
            "Self-employment disputes, business-to-business contracts, "
            "family law, divorce, estate planning, criminal matters."
        ),
        "ambiguous_signals": "Caller is an independent contractor with unclear employment status.",
        "referral_suggestion": (
            "We recommend contacting the State Bar Lawyer Referral Service "
            "for matters outside our scope."
        ),
    },
]

NEXUS_CONFIG = {
    "assistant": {
        "persona_name": "Aria at Nexus Law",
        "system_prompt": (
            "Nexus Law is a personal injury and employment law firm based in Los Angeles. "
            "We do NOT handle family law, divorce, estate planning, immigration, or "
            "criminal defence matters."
        ),
    },
    "practice_areas": NEXUS_PRACTICE_AREAS,
    "global_policy_documents": [],
}

AGENT = IntakeQualificationAgent()
HISTORY: list[dict] = []


def _cfg(narrative_summary: str, case_type: str) -> dict:
    return {
        **NEXUS_CONFIG,
        "_collected": {
            "narrative_summary": narrative_summary,
            "case_type": case_type,
        },
    }


# ── Divorce scenarios — each must result in not_qualified / FAILED ─────────────

DIVORCE_CASES = [
    {
        "label": "Contested divorce",
        "narrative": (
            "My spouse and I are going through a contested divorce. "
            "We disagree on asset division and I need legal representation."
        ),
        "case_type": "divorce",
    },
    {
        "label": "Child custody dispute",
        "narrative": (
            "I am in the middle of a custody battle for my two children "
            "and I need an attorney to help me get primary custody."
        ),
        "case_type": "family law",
    },
    {
        "label": "Divorce with alimony",
        "narrative": (
            "I need a lawyer for my divorce proceedings. My spouse is asking "
            "for alimony and I want to fight that claim."
        ),
        "case_type": "family law / divorce",
    },
    {
        "label": "Uncontested divorce filing",
        "narrative": (
            "My wife and I both agree to get divorced. We just need help "
            "drafting and filing the divorce papers."
        ),
        "case_type": "divorce",
    },
]


class TestDivorceCaseLiveQualification:
    """
    Each divorce scenario must be rejected (not_qualified → FAILED).
    These tests call the real LLM — divorce is so far outside Personal Injury
    and Employment Law that the decision should be deterministic.
    """

    @pytest.mark.parametrize("case", DIVORCE_CASES, ids=[c["label"] for c in DIVORCE_CASES])
    def test_divorce_is_always_not_qualified(self, case):
        resp = AGENT.process("", {}, _cfg(case["narrative"], case["case_type"]), HISTORY)

        decision = resp.hidden_collected.get("qualification_decision")
        reason = resp.hidden_collected.get("qualification_reason", "")

        assert resp.status == AgentStatus.FAILED, (
            f"[{case['label']}] Expected FAILED but got {resp.status.value}. "
            f"LLM decision={decision!r}, reason={reason!r}"
        )
        assert decision == "not_qualified", (
            f"[{case['label']}] Expected not_qualified but got {decision!r}. "
            f"Reason: {reason!r}"
        )

    @pytest.mark.parametrize("case", DIVORCE_CASES, ids=[c["label"] for c in DIVORCE_CASES])
    def test_divorce_refusal_message_is_spoken(self, case):
        """The agent must speak a polite refusal — never go silent."""
        resp = AGENT.process("", {}, _cfg(case["narrative"], case["case_type"]), HISTORY)

        assert resp.speak, f"[{case['label']}] Agent spoke nothing — refusal message is missing."
        speak_lower = resp.speak.lower()
        assert any(
            word in speak_lower
            for word in ("sorry", "unfortunately", "outside", "unable", "cannot", "can't")
        ), (
            f"[{case['label']}] Refusal message does not contain a polite decline. "
            f"Got: {resp.speak!r}"
        )


# ── Sanity check: qualified cases still pass ──────────────────────────────────

QUALIFIED_CASES = [
    {
        "label": "Car accident",
        "narrative": "I was rear-ended on the 405 freeway and suffered a back injury.",
        "case_type": "personal injury",
    },
    {
        "label": "Wrongful termination",
        "narrative": "I was fired two weeks after reporting sexual harassment to HR.",
        "case_type": "employment",
    },
]


class TestQualifiedCasesLive:

    @pytest.mark.parametrize("case", QUALIFIED_CASES, ids=[c["label"] for c in QUALIFIED_CASES])
    def test_in_scope_case_is_qualified(self, case):
        resp = AGENT.process("", {}, _cfg(case["narrative"], case["case_type"]), HISTORY)

        decision = resp.hidden_collected.get("qualification_decision")
        assert resp.status == AgentStatus.COMPLETED, (
            f"[{case['label']}] Expected COMPLETED but got {resp.status.value}. "
            f"Decision={decision!r}"
        )
        assert decision in ("qualified", "ambiguous"), (
            f"[{case['label']}] Expected qualified or ambiguous, got {decision!r}"
        )


# ── CLI runner ────────────────────────────────────────────────────────────────

def _run_all() -> int:
    passed = failed = 0
    all_cases = [
        ("DIVORCE — should be REJECTED", c, "not_qualified")
        for c in DIVORCE_CASES
    ] + [
        ("IN-SCOPE — should be QUALIFIED", c, "qualified_or_ambiguous")
        for c in QUALIFIED_CASES
    ]

    print("\n" + "═" * 70)
    print("  Intake Qualification — Live LLM Simulation")
    print("  Firm: Nexus Law  |  Practice areas: Personal Injury, Employment Law")
    print("═" * 70)

    for group_label, case, expected in all_cases:
        cfg = _cfg(case["narrative"], case["case_type"])
        resp = AGENT.process("", {}, cfg, HISTORY)
        decision = resp.hidden_collected.get("qualification_decision", "?")

        if expected == "not_qualified":
            ok = resp.status == AgentStatus.FAILED and decision == "not_qualified"
        else:
            ok = resp.status == AgentStatus.COMPLETED and decision in ("qualified", "ambiguous")

        icon = "✓" if ok else "✗"
        if ok:
            passed += 1
        else:
            failed += 1

        print(f"\n  {icon}  [{group_label}] {case['label']}")
        print(f"       status   : {resp.status.value}")
        print(f"       decision : {decision}")
        print(f"       reason   : {resp.hidden_collected.get('qualification_reason', '')}")
        print(f"       speak    : {resp.speak!r:.90}")

    print("\n" + "═" * 70)
    print(f"  RESULTS: {passed} passed, {failed} failed")
    print("═" * 70)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(_run_all())
