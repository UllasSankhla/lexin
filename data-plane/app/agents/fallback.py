"""Fallback agent — acknowledges unanswered questions and records notes."""
from __future__ import annotations

import logging
from app.agents.base import AgentBase, AgentStatus, SubagentResponse
from app.agents.llm_utils import llm_text_call

logger = logging.getLogger(__name__)


class FallbackAgent(AgentBase):
    def process(
        self,
        utterance: str,
        internal_state: dict,
        config: dict,
        history: list[dict],
    ) -> SubagentResponse:
        # Accumulate notes across calls
        existing_notes = internal_state.get("notes", "")
        new_note = f"Unanswered query: {utterance}"
        updated_notes = (existing_notes + "\n" + new_note).strip()
        internal_state["notes"] = updated_notes

        speak = llm_text_call(
            "You are an AI receptionist. Acknowledge that you cannot answer the caller's question right now and let them know someone will follow up. One sentence, warm and professional.",
            f"Caller asked: \"{utterance}\"",
            max_tokens=60,
        )
        return SubagentResponse(
            status=AgentStatus.COMPLETED,
            speak=speak,
            notes=updated_notes,
            requires_router_resume=True,
            internal_state=internal_state,
        )
