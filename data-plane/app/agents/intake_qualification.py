"""Intake qualification agent — decides if the caller's matter is within firm scope.

Reads the full conversation history and full_narrative collected by
NarrativeCollectionAgent, plus config.practice_areas (rich objects with
criteria fields and optional policy document excerpts), and makes a single
LLM qualification decision using complete caller context.

Outcomes
--------
qualified   → COMPLETED  (proceed to scheduling)
ambiguous   → COMPLETED  (proceed to scheduling with a note)
not_qualified → FAILED   (graph routes to end("not_qualified"))

The agent speaks a brief, professional outcome message and does NOT ask the
caller questions — it is a one-shot auto-decision node.
"""
from __future__ import annotations

import logging
import random

from app.agents.base import AgentBase, AgentStatus, SubagentResponse
from app.agents.llm_utils import llm_structured_call, llm_text_call
from app.agents.agent_schemas import QualificationResult

logger = logging.getLogger(__name__)

_CALENDAR_FILLERS = [
    "Let me find some time for you on our calendar.",
    "Let me pull up our schedule and find a time that works for you.",
    "One moment while I check our availability for you.",
    "Let me look at what times we have open for you.",
    "Give me just a moment while I check the calendar.",
]

# ── Prompt constants ──────────────────────────────────────────────────────────

_QUALIFY_SYSTEM = """\
You are an intake coordinator AI at a law firm. Given the full conversation \
transcript and the caller's complete narrative, decide whether the firm CAN \
handle this matter based on the firm's practice area profiles.

Use ALL available information from the transcript — including any membership \
details, insurance information, referral context, or specific circumstances \
the caller mentioned — not just the narrative summary.

Each practice area may include:
  - Qualification criteria: signals that the matter IS in scope.
  - Disqualification signals: signals that the matter is NOT in scope.
  - Ambiguous signals: signals that require a human to confirm.
  - Policy excerpts: internal documents with additional rules.

Decision rules:
  "qualified"     — caller's matter clearly matches at least one practice area's
                    qualification criteria.
  "ambiguous"     — matter matches ambiguous signals or no criteria fields are
                    defined for the matched area.
  "not_qualified" — matter clearly matches disqualification signals across ALL
                    practice areas and matches none of the qualification criteria.

Reply ONLY valid JSON:
{"decision": "qualified"|"ambiguous"|"not_qualified", "matched_area": "<area name or null>", "reason": "<one sentence>"}"""


_NOT_QUALIFIED_SPEAK_SYSTEM = (
    "You are an AI receptionist at a law firm. The caller's matter does not fall "
    "within the firm's practice areas. In two to three sentences: apologise that "
    "you are unable to assist with their specific matter, state the practice areas "
    "the firm does handle (provided below), and close warmly. "
    "Do not be dismissive. Voice-call style — no lists, no bullet points."
)

_NOT_QUALIFIED_REFERRAL_SYSTEM = (
    "You are an AI receptionist at a law firm. The caller's matter does not fall "
    "within the firm's practice areas. The firm has provided a referral suggestion. "
    "In two to three sentences: apologise that you are unable to assist, state the "
    "practice areas the firm does handle (provided below), include the referral "
    "suggestion naturally, and close warmly. Voice-call style — no lists, no bullet points."
)


def _build_practice_areas_prompt(practice_areas: list) -> str:
    """
    Render the practice_areas config list into a structured prompt block.

    Handles both the rich-object format (list of dicts with criteria fields)
    and the legacy flat-string format (list of str) for backward compatibility.
    """
    if not practice_areas:
        return "Practice areas: general legal matters (no specific areas configured)."

    lines: list[str] = ["FIRM PRACTICE AREAS\n" + "=" * 40]

    for i, area in enumerate(practice_areas, start=1):
        if isinstance(area, str):
            lines.append(f"\n[{i}] {area}")
            continue

        name = area.get("name", "Unnamed area")
        lines.append(f"\n[{i}] {name}")

        if area.get("description"):
            lines.append(f"    Description: {area['description']}")

        if area.get("qualification_criteria"):
            lines.append(f"    Qualification criteria: {area['qualification_criteria']}")

        if area.get("disqualification_signals"):
            lines.append(f"    Disqualification signals: {area['disqualification_signals']}")

        if area.get("ambiguous_signals"):
            lines.append(f"    Ambiguous signals: {area['ambiguous_signals']}")

        policy_docs: list[dict] = area.get("policy_documents", [])
        if policy_docs:
            lines.append("    Policy excerpts:")
            for doc in policy_docs:
                doc_name = doc.get("name", "document")
                # Truncate each document to 1500 chars to keep token budget manageable
                excerpt = (doc.get("content") or "")[:1500]
                label = f"{doc_name} — {doc['description']}" if doc.get("description") else doc_name
                lines.append(f"      [{label}]")
                lines.append(f"      {excerpt}")

    return "\n".join(lines)


def _build_global_policy_prompt(global_docs: list) -> str:
    """Render global (non-area-specific) policy documents into a prompt block."""
    if not global_docs:
        return ""

    lines: list[str] = ["\nGLOBAL FIRM POLICIES\n" + "=" * 40]
    for doc in global_docs:
        doc_name = doc.get("name", "document")
        excerpt = (doc.get("content") or "")[:1500]
        label = f"{doc_name} — {doc['description']}" if doc.get("description") else doc_name
        lines.append(f"\n[{label}]")
        lines.append(excerpt)

    return "\n".join(lines)


def _get_referral_suggestion(practice_areas: list, matched_area: str | None) -> str | None:
    """Return the referral_suggestion for the matched area, if any."""
    if not matched_area or not practice_areas:
        return None
    for area in practice_areas:
        if isinstance(area, dict) and area.get("name") == matched_area:
            return area.get("referral_suggestion") or None
    return None


class IntakeQualificationAgent(AgentBase):
    """
    Single-shot qualification node.  Invoked once (utterance is ignored — this
    node is meant to auto-run immediately after narrative_collection completes).

    Internal state keys
    -------------------
    decision     : "qualified" | "ambiguous" | "not_qualified" | None
    matched_area : str | None — practice area name matched by the LLM
    reason       : str — LLM's one-line rationale
    """

    def process(
        self,
        utterance: str,
        internal_state: dict,
        config: dict,
        history: list[dict],
    ) -> SubagentResponse:
        # If already decided (shouldn't normally be re-invoked, but guard anyway)
        if internal_state.get("decision"):
            return self._respond_from_state(internal_state, config)

        # ── Gather inputs ────────────────────────────────────────────────────
        practice_areas: list = config.get("practice_areas", [])
        global_policy_docs: list = config.get("global_policy_documents", [])

        areas_block = _build_practice_areas_prompt(practice_areas)
        global_block = _build_global_policy_prompt(global_policy_docs)

        # Build a readable transcript from the full call history so the LLM
        # has complete context (memberships, referrals, specific circumstances).
        transcript_lines = [
            f"  {t['role'].upper()}: {t['content']}"
            for t in history
            if t.get("content", "").strip()
        ]
        transcript_block = "\n".join(transcript_lines) if transcript_lines else "  (no transcript available)"

        user_message = (
            f"{areas_block}"
            f"{global_block}\n\n"
            f"FULL CONVERSATION TRANSCRIPT:\n{transcript_block}"
        )

        # ── LLM qualification call ───────────────────────────────────────────
        decision = "ambiguous"
        matched_area = None
        reason = "Could not determine qualification."
        try:
            result = llm_structured_call(
                _QUALIFY_SYSTEM,
                user_message,
                QualificationResult,
                max_tokens=1024,
                tag="intake_qualify",
            )
            decision = result.decision
            matched_area = result.matched_area or None
            reason = result.reason or reason
        except Exception as exc:
            logger.warning("IntakeQualification: LLM call failed: %s", exc)

        internal_state["decision"] = decision
        internal_state["matched_area"] = matched_area
        internal_state["reason"] = reason

        logger.info(
            "IntakeQualification: decision=%s | matched_area=%s | reason=%r",
            decision, matched_area, reason,
        )

        return self._respond_from_state(internal_state, config)

    # ── Private helpers ────────────────────────────────────────────────────

    def _respond_from_state(self, internal_state: dict, config: dict) -> SubagentResponse:
        decision = internal_state.get("decision", "ambiguous")
        matched_area = internal_state.get("matched_area")
        reason = internal_state.get("reason", "")

        practice_areas: list = config.get("practice_areas", [])

        if decision == "not_qualified":
            referral = _get_referral_suggestion(practice_areas, matched_area)
            speak_system = (
                _NOT_QUALIFIED_REFERRAL_SYSTEM
                if referral else _NOT_QUALIFIED_SPEAK_SYSTEM
            )
            area_names = [
                a["name"] if isinstance(a, dict) else str(a)
                for a in practice_areas
            ]
            areas_str = ", ".join(area_names) if area_names else "general legal matters"
            user_msg = f"Caller's matter: {reason}\nFirm's supported practice areas: {areas_str}"
            if referral:
                user_msg += f"\nReferral suggestion: {referral}"
            speak = llm_text_call(speak_system, user_msg, tag="intake_reject_speak") or (
                f"I'm sorry, but we're unable to assist with that type of matter. "
                f"Our firm focuses on {areas_str}. "
                f"We wish you the best and hope you find the right help."
            )
            return SubagentResponse(
                status=AgentStatus.FAILED,
                speak=speak,
                internal_state=internal_state,
                hidden_collected={
                    "qualification_decision": decision,
                    "qualification_reason": reason,
                    "matched_area": matched_area or "",
                },
            )

        # qualified or ambiguous → proceed to scheduling
        speak = random.choice(_CALENDAR_FILLERS)
        return SubagentResponse(
            status=AgentStatus.COMPLETED,
            speak=speak,
            internal_state=internal_state,
            hidden_collected={
                "qualification_decision": decision,
                "qualification_reason": reason,
                "matched_area": matched_area or "",
            },
        )
