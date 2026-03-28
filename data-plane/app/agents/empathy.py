"""Empathetic acknowledgment agent — acknowledges the caller's situation with genuine warmth.

Invoked once at the start of a call when the caller first describes their legal
matter or personal circumstances. Produces a brief, contextually-specific
empathetic response BEFORE the intake flow asks for contact details.

Design:
- One-shot: always returns COMPLETED immediately.
- No questions asked — purely acknowledgment.
- Uses the caller's own words to make the response feel genuine, not formulaic.
- Keeps it to 1-2 sentences — concise enough that the intake flow can continue
  without the caller feeling stalled.
"""
from __future__ import annotations

import logging

from app.agents.base import AgentBase, AgentStatus, SubagentResponse
from app.agents.llm_utils import llm_text_call

logger = logging.getLogger(__name__)

_SYSTEM = """\
You are an empathetic AI receptionist at a law firm. The caller has just described
their situation or legal matter. In 1-2 sentences, acknowledge their situation with
genuine warmth and care.

Rules:
- Be specific to what they said — reference their actual situation, not a generic reply
- Do NOT ask any questions
- Do NOT make promises about outcomes or whether the firm can help
- Do NOT mention the firm's name or services
- Keep it brief: 1-2 sentences only
- Voice-call style: natural, warm, conversational
- If the caller describes an injury, accident, or hardship, express genuine concern
- If the caller describes a dispute or legal problem, acknowledge the difficulty
"""

_FALLBACKS = [
    "I'm so sorry to hear that — that sounds like a very difficult situation.",
    "Thank you for sharing that with me. I can hear how stressful this has been.",
    "I'm sorry you're going through this. We'll do our best to help you.",
]

_fallback_idx = 0


class EmpathyAgent(AgentBase):
    """
    One-shot empathetic acknowledgment agent.

    Invoked proactively by the planner when the caller first describes their
    situation (utterance_type=NARRATIVE). Returns COMPLETED immediately so the
    planner can continue with data_collection or narrative_collection.
    """

    def process(
        self,
        utterance: str,
        internal_state: dict,
        config: dict,
        history: list[dict],
    ) -> SubagentResponse:
        global _fallback_idx

        if not utterance.strip():
            # Empty utterance — shouldn't happen, but guard gracefully
            speak = _FALLBACKS[0]
        else:
            user_msg = f'Caller said: "{utterance}"\n\nGenerate a brief empathetic acknowledgment.'
            try:
                speak = llm_text_call(_SYSTEM, user_msg, max_tokens=150)
            except Exception as exc:
                logger.warning("EmpathyAgent LLM call failed: %s — using fallback", exc)
                speak = _FALLBACKS[_fallback_idx % len(_FALLBACKS)]
                _fallback_idx += 1

            if not speak or not speak.strip():
                speak = _FALLBACKS[_fallback_idx % len(_FALLBACKS)]
                _fallback_idx += 1

        logger.info("EmpathyAgent: speak=%r", speak[:100])
        return SubagentResponse(
            status=AgentStatus.COMPLETED,
            speak=speak,
            internal_state=internal_state,
        )
