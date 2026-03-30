"""Context docs agent — answers general business questions from uploaded documents."""
from __future__ import annotations

import logging
from app.agents.base import AgentBase, AgentStatus, SubagentResponse
from app.agents.llm_utils import llm_structured_call, llm_text_call
from app.agents.agent_schemas import ContextDocsResult

logger = logging.getLogger(__name__)

_ANSWER_SYSTEM = (
    "You are an AI receptionist answering a caller's question using provided business documents. "
    "If the documents don't contain a clear answer, return {\"found\": false}. "
    "Otherwise return {\"found\": true, \"answer\": \"<concise voice-friendly answer>\"}"
)


class ContextDocsAgent(AgentBase):
    def process(
        self,
        utterance: str,
        internal_state: dict,
        config: dict,
        history: list[dict],
    ) -> SubagentResponse:
        context_files = config.get("context_files", [])
        if not context_files:
            return SubagentResponse(
                status=AgentStatus.FAILED,
                speak="",
                internal_state=internal_state,
            )

        docs_text = "\n\n".join(
            f"[{cf['name']}]\n{cf.get('content', '')[:1000]}"
            for cf in context_files
        )
        try:
            result = llm_structured_call(
                _ANSWER_SYSTEM,
                f"DOCUMENTS:\n{docs_text}\n\nCALLER QUESTION: \"{utterance}\"",
                ContextDocsResult,
            )
            if result.found and result.answer:
                return SubagentResponse(
                    status=AgentStatus.COMPLETED,
                    speak=result.answer,
                    internal_state=internal_state,
                )
        except Exception as exc:
            logger.warning("ContextDocsAgent failed: %s", exc)

        return SubagentResponse(
            status=AgentStatus.FAILED,
            speak="",
            internal_state=internal_state,
        )
