"""
Unit tests for IntakeQualificationAgent.

Firm under test: Nexus Law — Personal Injury + Employment Law only.
Divorce / family law is not a practice area → must always be not_qualified.

All LLM calls are patched — no network access or API keys required.

Run:
    cd data-plane
    python3 -m pytest tests/test_intake_qualification.py -v
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from app.agents.intake_qualification import (
    IntakeQualificationAgent,
    _build_practice_areas_prompt,
    _get_referral_suggestion,
)
from app.agents.base import AgentStatus


# ── Fixtures: Nexus Law — two practice areas only ─────────────────────────────

NEXUS_PRACTICE_AREAS = [
    {
        "name": "Personal Injury",
        "description": "Car accidents, slip and fall, premises liability, wrongful death.",
        "qualification_criteria": (
            "Caller suffered physical injury due to another party's negligence in California."
        ),
        "disqualification_signals": (
            "Matters outside California, pure property damage only, criminal charges, "
            "family law, divorce, custody, estate planning."
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
            "Self-employment disputes, business-to-business contracts, family law, "
            "divorce, estate planning."
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
            "Nexus Law is a personal injury and employment law firm based in Los Angeles."
        ),
    },
    "practice_areas": NEXUS_PRACTICE_AREAS,
    "global_policy_documents": [],
    "parameters": [
        {"name": "first_name", "display_label": "First Name", "data_type": "name", "required": True},
        {"name": "email_address", "display_label": "Email Address", "data_type": "email", "required": True},
    ],
}

AGENT = IntakeQualificationAgent()
HISTORY: list[dict] = []


def _cfg(narrative_summary: str, case_type: str, extra: dict | None = None) -> dict:
    """Build a config dict with _collected populated."""
    cfg = {
        **NEXUS_CONFIG,
        "_collected": {
            "narrative_summary": narrative_summary,
            "case_type": case_type,
        },
    }
    if extra:
        cfg.update(extra)
    return cfg


# ── _build_practice_areas_prompt ──────────────────────────────────────────────

class TestBuildPracticeAreasPrompt:

    def test_empty_areas_returns_fallback_text(self):
        result = _build_practice_areas_prompt([])
        assert "general legal matters" in result

    def test_string_areas_are_rendered(self):
        result = _build_practice_areas_prompt(["Personal Injury", "Employment Law"])
        assert "Personal Injury" in result
        assert "Employment Law" in result

    def test_rich_areas_include_all_criteria_fields(self):
        result = _build_practice_areas_prompt(NEXUS_PRACTICE_AREAS)
        assert "Personal Injury" in result
        assert "Employment Law" in result
        assert "Qualification criteria" in result
        assert "Disqualification signals" in result
        assert "Ambiguous signals" in result

    def test_policy_doc_content_truncated_to_1500_chars(self):
        # Use "Z" — does not appear in structural header text ("FIRM PRACTICE AREAS", etc.)
        areas = [
            {
                "name": "Test Area",
                "policy_documents": [
                    {"name": "Big Policy", "content": "Z" * 3000, "description": "test"},
                ],
            }
        ]
        result = _build_practice_areas_prompt(areas)
        assert result.count("Z") <= 1500


# ── _get_referral_suggestion ──────────────────────────────────────────────────

class TestGetReferralSuggestion:

    def test_returns_referral_for_matched_area(self):
        result = _get_referral_suggestion(NEXUS_PRACTICE_AREAS, "Employment Law")
        assert result is not None
        assert "State Bar" in result

    def test_returns_none_when_matched_area_has_no_referral(self):
        result = _get_referral_suggestion(NEXUS_PRACTICE_AREAS, "Personal Injury")
        assert result is None

    def test_returns_none_for_unknown_area_name(self):
        result = _get_referral_suggestion(NEXUS_PRACTICE_AREAS, "Family Law")
        assert result is None

    def test_returns_none_when_matched_area_is_none(self):
        result = _get_referral_suggestion(NEXUS_PRACTICE_AREAS, None)
        assert result is None

    def test_returns_none_when_areas_list_is_empty(self):
        result = _get_referral_suggestion([], "Personal Injury")
        assert result is None


# ── Divorce case — the primary case under test ────────────────────────────────

class TestDivorceCaseQualification:
    """
    Divorce / family law is not one of Nexus Law's two practice areas.
    The outcome must always be not_qualified → FAILED status → no scheduling.
    """

    def test_divorce_case_returns_failed_status(self):
        with (
            patch("app.agents.intake_qualification.llm_json_call") as mock_json,
            patch("app.agents.intake_qualification.llm_text_call") as mock_text,
        ):
            mock_json.return_value = {
                "decision": "not_qualified",
                "matched_area": None,
                "reason": (
                    "Divorce and family law are not within this firm's practice areas "
                    "of personal injury and employment law."
                ),
            }
            mock_text.return_value = (
                "I'm sorry to hear about your situation. Unfortunately, divorce and "
                "family law matters fall outside our practice areas — we'd encourage "
                "you to reach out to a family law attorney."
            )

            resp = AGENT.process(
                "",
                {},
                _cfg(
                    narrative_summary=(
                        "Caller is seeking legal assistance with a contested divorce "
                        "and child custody arrangement."
                    ),
                    case_type="family law / divorce",
                ),
                HISTORY,
            )

        assert resp.status == AgentStatus.FAILED

    def test_divorce_case_records_not_qualified_decision(self):
        with (
            patch("app.agents.intake_qualification.llm_json_call") as mock_json,
            patch("app.agents.intake_qualification.llm_text_call"),
        ):
            mock_json.return_value = {
                "decision": "not_qualified",
                "matched_area": None,
                "reason": "Divorce is family law, outside scope.",
            }

            resp = AGENT.process(
                "",
                {},
                _cfg(
                    narrative_summary="Caller wants help with their divorce.",
                    case_type="divorce",
                ),
                HISTORY,
            )

        assert resp.hidden_collected["qualification_decision"] == "not_qualified"

    def test_divorce_case_speak_is_non_empty(self):
        """The agent must always speak a refusal — never go silent."""
        with (
            patch("app.agents.intake_qualification.llm_json_call") as mock_json,
            patch("app.agents.intake_qualification.llm_text_call") as mock_text,
        ):
            mock_json.return_value = {
                "decision": "not_qualified",
                "matched_area": None,
                "reason": "Family law outside scope.",
            }
            mock_text.return_value = (
                "I'm sorry, divorce matters are outside our practice areas. "
                "We recommend consulting a family law attorney."
            )

            resp = AGENT.process(
                "",
                {},
                _cfg(
                    narrative_summary="I need help with my divorce and property split.",
                    case_type="divorce",
                ),
                HISTORY,
            )

        assert resp.speak
        assert len(resp.speak) > 10

    def test_divorce_case_speak_is_politely_declining(self):
        """Refusal message must acknowledge the caller's situation and decline politely."""
        with (
            patch("app.agents.intake_qualification.llm_json_call") as mock_json,
            patch("app.agents.intake_qualification.llm_text_call") as mock_text,
        ):
            mock_json.return_value = {
                "decision": "not_qualified",
                "matched_area": None,
                "reason": "Divorce / family law not in scope.",
            }
            mock_text.return_value = (
                "I'm sorry, but divorce matters fall outside our practice areas. "
                "We recommend consulting a family law attorney."
            )

            resp = AGENT.process(
                "",
                {},
                _cfg(
                    narrative_summary="My spouse and I are separating and I need legal help.",
                    case_type="divorce",
                ),
                HISTORY,
            )

        speak_lower = resp.speak.lower()
        # Must express a refusal in some form
        assert any(
            word in speak_lower
            for word in ("sorry", "unfortunately", "outside", "unable", "cannot", "can't")
        ), f"Refusal message missing polite decline: {resp.speak!r}"

    def test_divorce_case_does_not_produce_completed_status(self):
        """COMPLETED would route to scheduling — must never happen for not_qualified."""
        with (
            patch("app.agents.intake_qualification.llm_json_call") as mock_json,
            patch("app.agents.intake_qualification.llm_text_call"),
        ):
            mock_json.return_value = {
                "decision": "not_qualified",
                "matched_area": None,
                "reason": "Divorce outside scope.",
            }

            resp = AGENT.process(
                "",
                {},
                _cfg(
                    narrative_summary="Help with divorce proceedings.",
                    case_type="divorce",
                ),
                HISTORY,
            )

        assert resp.status != AgentStatus.COMPLETED


# ── Qualified paths (sanity check) ────────────────────────────────────────────

class TestQualifiedPaths:

    def test_personal_injury_returns_completed(self):
        with patch("app.agents.intake_qualification.llm_json_call") as mock_json:
            mock_json.return_value = {
                "decision": "qualified",
                "matched_area": "Personal Injury",
                "reason": "Car accident on California highway matches personal injury criteria.",
            }

            resp = AGENT.process(
                "",
                {},
                _cfg(
                    narrative_summary="I was rear-ended on the 405 and have back injuries.",
                    case_type="personal injury",
                ),
                HISTORY,
            )

        assert resp.status == AgentStatus.COMPLETED
        assert resp.hidden_collected["qualification_decision"] == "qualified"
        assert resp.hidden_collected["matched_area"] == "Personal Injury"

    def test_employment_law_returns_completed(self):
        with patch("app.agents.intake_qualification.llm_json_call") as mock_json:
            mock_json.return_value = {
                "decision": "qualified",
                "matched_area": "Employment Law",
                "reason": "Wrongful termination and retaliation match employment law criteria.",
            }

            resp = AGENT.process(
                "",
                {},
                _cfg(
                    narrative_summary="I was fired after reporting harassment — I believe it was retaliation.",
                    case_type="employment",
                ),
                HISTORY,
            )

        assert resp.status == AgentStatus.COMPLETED
        assert resp.hidden_collected["qualification_decision"] == "qualified"

    def test_qualified_case_speaks_calendar_filler(self):
        """Qualified cases should transition by speaking a calendar filler, not a refusal."""
        with patch("app.agents.intake_qualification.llm_json_call") as mock_json:
            mock_json.return_value = {
                "decision": "qualified",
                "matched_area": "Personal Injury",
                "reason": "Slip-and-fall on commercial premises in California.",
            }

            resp = AGENT.process(
                "",
                {},
                _cfg(
                    narrative_summary="I slipped on a wet floor at a grocery store.",
                    case_type="personal injury",
                ),
                HISTORY,
            )

        assert resp.speak
        # Must NOT contain a refusal — should be a scheduling transition phrase
        assert "outside" not in resp.speak.lower()
        assert "unfortunately" not in resp.speak.lower()


# ── Ambiguous path ────────────────────────────────────────────────────────────

class TestAmbiguousPath:

    def test_ambiguous_returns_completed(self):
        """Ambiguous → COMPLETED so a human can assess during the consultation."""
        with patch("app.agents.intake_qualification.llm_json_call") as mock_json:
            mock_json.return_value = {
                "decision": "ambiguous",
                "matched_area": "Employment Law",
                "reason": "Caller is a freelancer — employment status unclear.",
            }

            resp = AGENT.process(
                "",
                {},
                _cfg(
                    narrative_summary="I'm a freelancer and the company I work with won't pay me.",
                    case_type="unknown",
                ),
                HISTORY,
            )

        assert resp.status == AgentStatus.COMPLETED
        assert resp.hidden_collected["qualification_decision"] == "ambiguous"


# ── LLM error handling ────────────────────────────────────────────────────────

class TestLLMErrorHandling:

    def test_llm_exception_falls_back_to_ambiguous(self):
        """LLM failure must NOT result in a hard rejection — fall back to ambiguous."""
        with patch("app.agents.intake_qualification.llm_json_call") as mock_json:
            mock_json.side_effect = ValueError("LLM returned empty content")

            resp = AGENT.process(
                "",
                {},
                _cfg(narrative_summary="Some legal matter.", case_type="unknown"),
                HISTORY,
            )

        assert resp.status == AgentStatus.COMPLETED
        assert resp.hidden_collected.get("qualification_decision") == "ambiguous"

    def test_unexpected_decision_value_normalised_to_ambiguous(self):
        with patch("app.agents.intake_qualification.llm_json_call") as mock_json:
            mock_json.return_value = {
                "decision": "maybe",  # not a valid value
                "matched_area": None,
                "reason": "Not sure.",
            }

            resp = AGENT.process(
                "",
                {},
                _cfg(narrative_summary="Unclear matter.", case_type="unknown"),
                HISTORY,
            )

        assert resp.status == AgentStatus.COMPLETED
        assert resp.hidden_collected["qualification_decision"] == "ambiguous"

    def test_llm_text_call_returns_empty_uses_hardcoded_fallback(self):
        """If the speak-generation LLM call returns empty, use the hardcoded fallback."""
        with (
            patch("app.agents.intake_qualification.llm_json_call") as mock_json,
            patch("app.agents.intake_qualification.llm_text_call") as mock_text,
        ):
            mock_json.return_value = {
                "decision": "not_qualified",
                "matched_area": None,
                "reason": "Estate planning outside scope.",
            }
            mock_text.return_value = ""  # empty → falsy → use fallback

            resp = AGENT.process(
                "",
                {},
                _cfg(narrative_summary="I need a will drafted.", case_type="estate planning"),
                HISTORY,
            )

        assert resp.status == AgentStatus.FAILED
        assert resp.speak  # fallback speak is non-empty
        assert any(kw in resp.speak.lower() for kw in ("unable to assist", "focuses on", "practice areas", "outside"))


# ── Referral suggestion injection ─────────────────────────────────────────────

class TestReferralSuggestion:

    def test_referral_injected_into_speak_system_prompt(self):
        """
        When the matched area has a referral_suggestion, the refusal prompt
        must include the referral text so the LLM can mention it naturally.
        """
        with (
            patch("app.agents.intake_qualification.llm_json_call") as mock_json,
            patch("app.agents.intake_qualification.llm_text_call") as mock_text,
        ):
            mock_json.return_value = {
                "decision": "not_qualified",
                "matched_area": "Employment Law",
                "reason": "Business-to-business contract dispute, not an employer-employee issue.",
            }
            mock_text.return_value = (
                "Unfortunately this falls outside our scope. "
                "We recommend the State Bar Lawyer Referral Service."
            )

            resp = AGENT.process(
                "",
                {},
                _cfg(
                    narrative_summary="I have a contract dispute with a vendor company.",
                    case_type="business law",
                ),
                HISTORY,
            )

        system_prompt_used = mock_text.call_args[0][0]
        assert "State Bar" in system_prompt_used

    def test_no_referral_uses_generic_speak_system(self):
        """When matched area has no referral, the generic decline prompt is used."""
        with (
            patch("app.agents.intake_qualification.llm_json_call") as mock_json,
            patch("app.agents.intake_qualification.llm_text_call") as mock_text,
        ):
            mock_json.return_value = {
                "decision": "not_qualified",
                "matched_area": "Personal Injury",  # no referral configured
                "reason": "Out-of-state matter, not California.",
            }
            mock_text.return_value = "Unfortunately we only handle California matters."

            resp = AGENT.process(
                "",
                {},
                _cfg(
                    narrative_summary="I was in a car accident in Texas.",
                    case_type="personal injury",
                ),
                HISTORY,
            )

        system_prompt_used = mock_text.call_args[0][0]
        # Generic prompt does NOT inject a referral suggestion
        assert "State Bar" not in system_prompt_used


# ── Already-decided guard ─────────────────────────────────────────────────────

class TestAlreadyDecidedGuard:

    def test_already_qualified_state_skips_llm(self):
        with patch("app.agents.intake_qualification.llm_json_call") as mock_json:
            state = {
                "decision": "qualified",
                "matched_area": "Personal Injury",
                "reason": "Pre-decided.",
            }

            resp = AGENT.process("", state, _cfg("Any matter.", "personal injury"), HISTORY)

        mock_json.assert_not_called()
        assert resp.status == AgentStatus.COMPLETED

    def test_already_not_qualified_state_skips_llm(self):
        with (
            patch("app.agents.intake_qualification.llm_json_call") as mock_json,
            patch("app.agents.intake_qualification.llm_text_call") as mock_text,
        ):
            mock_text.return_value = "We cannot assist with this matter."
            state = {
                "decision": "not_qualified",
                "matched_area": None,
                "reason": "Pre-decided.",
            }

            resp = AGENT.process("", state, _cfg("Divorce case.", "divorce"), HISTORY)

        mock_json.assert_not_called()
        assert resp.status == AgentStatus.FAILED


# ── No practice areas configured ─────────────────────────────────────────────

class TestNoPracticeAreas:

    def test_no_areas_defaults_to_general_fallback_text(self):
        prompt = _build_practice_areas_prompt([])
        assert "general legal matters" in prompt

    def test_no_areas_ambiguous_llm_returns_completed(self):
        with patch("app.agents.intake_qualification.llm_json_call") as mock_json:
            mock_json.return_value = {
                "decision": "ambiguous",
                "matched_area": None,
                "reason": "No criteria defined.",
            }

            cfg = {
                "practice_areas": [],
                "global_policy_documents": [],
                "_collected": {
                    "narrative_summary": "I need legal help.",
                    "case_type": "unknown",
                },
            }
            resp = AGENT.process("", {}, cfg, HISTORY)

        assert resp.status == AgentStatus.COMPLETED
