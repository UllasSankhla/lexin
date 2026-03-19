"""Intake qualification agent — decides if the caller's matter is within firm scope.

Reads collected data from prior agents (narrative_summary, case_type, caller
fields like insurance/metlife_id) plus config.practice_areas and makes a single
LLM qualification decision.

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

from app.agents.base import AgentBase, AgentStatus, SubagentResponse
from app.agents.llm_utils import llm_json_call, llm_text_call

logger = logging.getLogger(__name__)

_QUALIFY_SYSTEM = (
    "You are an intake coordinator AI at a law firm. Given a caller's narrative "
    "summary, detected case type, and the firm's practice areas, decide whether "
    "the firm CAN handle this matter.\n\n"
    "Rules:\n"
    "- 'qualified'     — case type clearly matches one of the firm's practice areas.\n"
    "- 'ambiguous'     — case type is 'unknown' or only loosely related; the firm "
    "might handle it but a human should confirm.\n"
    "- 'not_qualified' — case type clearly falls outside all of the firm's "
    "practice areas (e.g. criminal defence at a family law firm).\n\n"
    "Reply ONLY valid JSON:\n"
    "{\"decision\": \"qualified\"|\"ambiguous\"|\"not_qualified\", "
    "\"reason\": \"<one sentence>\"}"
)

_QUALIFIED_SPEAK_SYSTEM = (
    "You are an AI receptionist at a law firm. The caller's matter has been "
    "assessed and it falls within the firm's practice areas. Acknowledge this "
    "warmly and briefly (one sentence) — say you'll proceed to schedule a "
    "consultation. Do not repeat case details."
)

_AMBIGUOUS_SPEAK_SYSTEM = (
    "You are an AI receptionist at a law firm. The caller's matter may fall "
    "within the firm's practice areas but requires confirmation by a lawyer. "
    "Acknowledge this warmly and briefly (one sentence) — say you'll schedule a "
    "consultation so a specialist can assess further. Do not repeat case details."
)

_NOT_QUALIFIED_SPEAK_SYSTEM = (
    "You are an AI receptionist at a law firm. The caller's matter does not fall "
    "within the firm's practice areas. Decline politely, acknowledge their "
    "situation with empathy, and suggest they seek a firm that specialises in "
    "their area of need. Two sentences maximum. Do not be dismissive."
)


class IntakeQualificationAgent(AgentBase):
    """
    Single-shot qualification node.  Invoked once (utterance is ignored — this
    node is meant to auto-run immediately after narrative_collection completes).

    Internal state keys
    -------------------
    decision  : "qualified" | "ambiguous" | "not_qualified" | None
    reason    : str — LLM's one-line rationale
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
            return self._respond_from_state(internal_state)

        # ── Gather inputs ────────────────────────────────────────────────────
        collected: dict = {}
        # Traverse history to gather collected fields from prior agents
        for entry in history:
            if entry.get("role") == "collected":
                collected.update(entry.get("data", {}))

        narrative_summary = collected.get("narrative_summary", "")
        case_type = collected.get("case_type", "unknown")
        practice_areas = config.get("practice_areas", [])
        areas_str = ", ".join(practice_areas) if practice_areas else "general legal matters"

        # ── LLM qualification call ───────────────────────────────────────────
        decision = "ambiguous"
        reason = "Could not determine qualification."
        try:
            result = llm_json_call(
                _QUALIFY_SYSTEM,
                (
                    f"Practice areas this firm handles: {areas_str}\n\n"
                    f"Detected case type: {case_type}\n\n"
                    f"Narrative summary: {narrative_summary}"
                ),
                max_tokens=128,
            )
            decision = result.get("decision", "ambiguous")
            reason = result.get("reason", reason)
            if decision not in ("qualified", "ambiguous", "not_qualified"):
                logger.warning(
                    "IntakeQualification: unexpected decision value %r — treating as ambiguous",
                    decision,
                )
                decision = "ambiguous"
        except Exception as exc:
            logger.warning("IntakeQualification: LLM call failed: %s", exc)

        internal_state["decision"] = decision
        internal_state["reason"] = reason

        logger.info(
            "IntakeQualification: decision=%s | case_type=%s | reason=%r",
            decision, case_type, reason,
        )

        return self._respond_from_state(internal_state)

    # ── Private helpers ────────────────────────────────────────────────────

    def _respond_from_state(self, internal_state: dict) -> SubagentResponse:
        decision = internal_state.get("decision", "ambiguous")
        reason = internal_state.get("reason", "")

        if decision == "not_qualified":
            speak = llm_text_call(_NOT_QUALIFIED_SPEAK_SYSTEM, reason) or (
                "Thank you for reaching out. Unfortunately, this matter falls outside "
                "our current practice areas. We encourage you to seek a firm that "
                "specialises in your area of need."
            )
            return SubagentResponse(
                status=AgentStatus.FAILED,
                speak=speak,
                internal_state=internal_state,
                collected={"qualification_decision": decision, "qualification_reason": reason},
            )

        # qualified or ambiguous → proceed to scheduling
        system = _QUALIFIED_SPEAK_SYSTEM if decision == "qualified" else _AMBIGUOUS_SPEAK_SYSTEM
        speak = llm_text_call(system, reason) or (
            "Great, your matter looks like something we can assist with — "
            "let's go ahead and schedule a consultation."
        )
        return SubagentResponse(
            status=AgentStatus.COMPLETED,
            speak=speak,
            internal_state=internal_state,
            collected={"qualification_decision": decision, "qualification_reason": reason},
        )
