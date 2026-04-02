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

class TestFallbackDefers:
    """
    FallbackAgent always defers — it never answers from context.
    It notes the question and tells the caller a team member will follow up.
    These tests verify that deferral behavior across different question types.
    """

    _DEFERRAL_KEYWORDS = (
        "don't have", "noted", "team member", "follow up",
        "cannot", "can't", "not on hand", "not have",
    )

    def _assert_defers(self, speak: str) -> None:
        lower = speak.lower()
        assert any(kw in lower for kw in self._DEFERRAL_KEYWORDS), (
            f"Expected deferral response, got: {speak!r}"
        )
        assert len(speak.split()) <= 40, (
            f"Deferral should be 1 concise sentence, got: {speak!r}"
        )

    def test_defers_fee_question(self):
        """Business question about fees — always deferred."""
        speak = run_fallback("Do I have to pay anything upfront?", RICH_CONFIG)
        self._assert_defers(speak)

    def test_defers_practice_area_question(self):
        """Business question about case types — always deferred."""
        speak = run_fallback("What kind of cases do you handle?", RICH_CONFIG)
        self._assert_defers(speak)

    def test_defers_collection_meta_question(self):
        """Question about what info is being collected — always deferred."""
        speak = run_fallback("Do you need my email address to continue?", RICH_CONFIG)
        self._assert_defers(speak)


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
