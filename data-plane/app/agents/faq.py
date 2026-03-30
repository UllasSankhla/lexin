"""FAQ agent — answers questions from the curated FAQ list."""
from __future__ import annotations

import logging
from app.agents.base import AgentBase, AgentStatus, SubagentResponse
from app.agents.llm_utils import llm_structured_call, llm_text_call
from app.agents.agent_schemas import LegalDeflectSignal, FAQMatchResult

logger = logging.getLogger(__name__)

_MATCH_SYSTEM = (
    "You match a caller's question to a FAQ list. "
    "Return ONLY valid JSON: {\"matched\": true, \"index\": <0-based int>} "
    "or {\"matched\": false} if no FAQ closely matches.\n\n"
    "IMPORTANT: If the question is about the caller's specific legal matter "
    "(H1B portability, I-140 status, visa strategy, EAD renewals, employment law "
    "case assessment, immigration law specifics, or any legal outcome question), "
    "always return {\"matched\": false} — these must be handled by the attorney, not the FAQ."
)

_LEGAL_DEFLECT_SYSTEM = (
    "Determine if the caller's question is a substantive legal question about their specific "
    "matter (e.g. H1B portability, I-140/priority date, visa strategy, EAD renewals, "
    "employment discrimination case assessment, immigration law specifics, legal outcomes). "
    "Return ONLY valid JSON: {\"is_legal_question\": true} or {\"is_legal_question\": false}."
)


class FAQAgent(AgentBase):
    def process(
        self,
        utterance: str,
        internal_state: dict,
        config: dict,
        history: list[dict],
    ) -> SubagentResponse:
        # Detect and deflect substantive legal questions before attempting FAQ match
        try:
            legal_check = llm_structured_call(
                _LEGAL_DEFLECT_SYSTEM,
                f"Caller asked: \"{utterance}\"",
                LegalDeflectSignal,
            )
            if legal_check.is_legal_question:
                speak = (
                    "That's a great question for our legal team — I'll make sure to note it "
                    "so the attorney can address it directly in your consultation."
                )
                logger.info("FAQAgent: deflecting legal question to attorney")
                return SubagentResponse(
                    status=AgentStatus.COMPLETED,
                    speak=speak,
                    internal_state=internal_state,
                )
        except Exception as exc:
            logger.warning("FAQAgent legal check failed: %s", exc)

        faqs = config.get("faqs", [])
        if not faqs:
            return SubagentResponse(
                status=AgentStatus.COMPLETED,
                speak="I'm sorry, I don't have an answer to that right now.",
                internal_state=internal_state,
            )

        faq_list = "\n".join(
            f"{i}. Q: {f['question']}" for i, f in enumerate(faqs)
        )
        try:
            match = llm_structured_call(
                _MATCH_SYSTEM,
                f"FAQ list:\n{faq_list}\n\nCaller asked: \"{utterance}\"",
                FAQMatchResult,
            )
            if match.matched and match.index is not None:
                idx = int(match.index)
                if 0 <= idx < len(faqs):
                    faq = faqs[idx]
                    answer = llm_text_call(
                        "You are an AI receptionist answering a caller's question. Be concise and conversational.",
                        f"FAQ answer to use: {faq['answer']}\nRephrase naturally for voice. Two sentences max.",
                    )
                    return SubagentResponse(
                        status=AgentStatus.COMPLETED,
                        speak=answer,
                        internal_state=internal_state,
                    )
        except Exception as exc:
            logger.warning("FAQAgent match failed: %s", exc)

        # No match — signal router to try context_docs or fallback
        return SubagentResponse(
            status=AgentStatus.FAILED,
            speak="",
            internal_state=internal_state,
        )
