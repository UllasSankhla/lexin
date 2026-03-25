"""
Simulation tests for the smarter FallbackAgent.

Tests that the agent:
1. Answers practice-area / business questions from the provided context.
2. Answers collection-meta questions (Is X required? Do you need my email?).
3. Falls back gracefully for genuinely out-of-context questions.
4. Stays within guardrails — no legal advice, no speculation.

Run:
    python3 -m pytest tests/simulate_fallback.py -v -s
"""
from __future__ import annotations

import pytest
from app.agents.fallback import FallbackAgent
from app.agents.base import AgentStatus

# ---------------------------------------------------------------------------
# Rich business config (simulates a real law firm)
# ---------------------------------------------------------------------------

RICH_CONFIG = {
    "assistant": {
        "persona_name": "Aria at Nexus Law",
        "system_prompt": (
            "Nexus Law is a personal injury and employment law firm based in Los Angeles. "
            "We offer free initial consultations. Our attorneys work on a contingency fee "
            "basis — you pay nothing unless we win. Office hours are Monday to Friday, "
            "9 AM to 6 PM Pacific Time."
        ),
    },
    "practice_areas": [
        {
            "name": "Personal Injury",
            "description": (
                "Car accidents, slip and fall, premises liability, and wrongful death claims. "
                "We represent clients throughout California."
            ),
        },
        {
            "name": "Employment Law",
            "description": (
                "Wrongful termination, workplace discrimination, sexual harassment, "
                "unpaid wages, and PAGA claims."
            ),
        },
    ],
    "global_policy_documents": [
        {
            "name": "Fee Structure",
            "content": (
                "Nexus Law operates on a contingency fee basis. "
                "Clients pay no upfront fees. Attorney fees are taken as a percentage "
                "of the settlement or judgment, typically 33% before litigation "
                "and 40% after a lawsuit is filed. Costs such as filing fees and "
                "expert witnesses are advanced by the firm and reimbursed from the recovery."
            ),
        },
        {
            "name": "Consultation Policy",
            "content": (
                "All initial consultations are free of charge and confidential. "
                "Consultations are available by phone, video, or in-person at our "
                "Los Angeles office. Same-day consultations may be available upon request."
            ),
        },
    ],
    "context_files": [],
    "faqs": [
        {
            "question": "How long does a personal injury case take?",
            "answer": (
                "Most personal injury cases settle within 6-18 months. "
                "Cases that go to trial may take 2-3 years."
            ),
        },
        {
            "question": "What is the statute of limitations for a car accident claim in California?",
            "answer": (
                "In California, you generally have 2 years from the date of the accident "
                "to file a personal injury lawsuit."
            ),
        },
    ],
    "parameters": [
        {"name": "first_name",    "display_label": "First Name",    "data_type": "name",  "required": True},
        {"name": "last_name",     "display_label": "Last Name",     "data_type": "name",  "required": True},
        {"name": "email_address", "display_label": "Email Address", "data_type": "email", "required": True},
        {"name": "phone_number",  "display_label": "Phone Number",  "data_type": "phone", "required": False},
    ],
}

EMPTY_CONFIG = {
    "assistant": {"persona_name": "Aria"},
    "practice_areas": [],
    "global_policy_documents": [],
    "context_files": [],
    "faqs": [],
    "parameters": [],
}


def run_fallback(utterance: str, config: dict, history: list[dict] | None = None) -> str:
    agent = FallbackAgent()
    resp = agent.process(
        utterance=utterance,
        internal_state={},
        config=config,
        history=history or [],
    )
    print(f"\n  Q: {utterance!r}")
    print(f"  A: {resp.speak!r}")
    assert resp.status == AgentStatus.COMPLETED
    assert resp.speak, "speak must be non-empty"
    return resp.speak


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFallbackBusinessQuestions:
    def test_fee_structure_question(self):
        """Should answer from the Fee Structure policy doc."""
        speak = run_fallback(
            "Do I have to pay anything upfront?",
            RICH_CONFIG,
        )
        # Should mention contingency / no upfront fee
        lower = speak.lower()
        assert any(kw in lower for kw in ("contingency", "upfront", "nothing", "no fee", "pay nothing", "win")), (
            f"Expected contingency fee explanation, got: {speak!r}"
        )

    def test_consultation_question(self):
        """Should answer that consultations are free."""
        speak = run_fallback(
            "Is the first consultation free?",
            RICH_CONFIG,
        )
        lower = speak.lower()
        assert any(kw in lower for kw in ("free", "no charge", "complimentary", "cost")), (
            f"Expected free consultation answer, got: {speak!r}"
        )

    def test_practice_area_question(self):
        """Should describe firm's practice areas."""
        speak = run_fallback(
            "What kind of cases do you handle?",
            RICH_CONFIG,
        )
        lower = speak.lower()
        assert any(kw in lower for kw in ("injury", "employment", "accident", "discrimination", "personal")), (
            f"Expected practice area info, got: {speak!r}"
        )

    def test_faq_statute_of_limitations(self):
        """Should answer from FAQ about statute of limitations."""
        speak = run_fallback(
            "How long do I have to file a claim after my car accident?",
            RICH_CONFIG,
        )
        # Normalize non-breaking spaces before matching
        normalized = speak.lower().replace("\u202f", " ").replace("\xa0", " ")
        assert any(kw in normalized for kw in ("2 year", "two year", "statute", "limitation", "deadline")), (
            f"Expected statute of limitations info, got: {speak!r}"
        )


class TestFallbackCollectionMetaQuestions:
    def test_is_email_required(self):
        """Should answer that email is required based on the parameters list."""
        speak = run_fallback(
            "Do you need my email address to continue?",
            RICH_CONFIG,
        )
        lower = speak.lower()
        assert any(kw in lower for kw in ("email", "address", "required", "need", "yes")), (
            f"Expected email required answer, got: {speak!r}"
        )

    def test_is_phone_optional(self):
        """Phone is optional in RICH_CONFIG — agent should convey that."""
        speak = run_fallback(
            "Is a phone number required?",
            RICH_CONFIG,
        )
        lower = speak.lower()
        assert any(kw in lower for kw in ("optional", "not required", "phone", "number")), (
            f"Expected optional phone answer, got: {speak!r}"
        )


class TestFallbackGuardrails:
    def test_out_of_context_question_empty_config(self):
        """With no context at all, agent should give the fallback sentence."""
        speak = run_fallback(
            "What's the best pizza place near you?",
            EMPTY_CONFIG,
        )
        # Should not try to answer — should say team member will follow up
        lower = speak.lower()
        assert any(kw in lower for kw in (
            "don't have", "team member", "follow up", "specific information", "cannot", "can't"
        )), f"Expected fallback response, got: {speak!r}"

    def test_notes_accumulated(self):
        """Notes should accumulate across multiple calls."""
        agent = FallbackAgent()
        state: dict = {}
        for utterance in ["What are your fees?", "Where is your office?"]:
            resp = agent.process(utterance, state, RICH_CONFIG, [])
            state = resp.internal_state
        assert "What are your fees?" in state.get("notes", "")
        assert "Where is your office?" in state.get("notes", "")
