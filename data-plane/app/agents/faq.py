"""FAQ agent — answers questions from the curated FAQ list."""
from __future__ import annotations

import logging
from app.agents.base import AgentBase, AgentStatus, SubagentResponse
from app.agents.llm_utils import llm_json_call, llm_text_call

logger = logging.getLogger(__name__)

_MATCH_SYSTEM = (
    "You match a caller's question to a FAQ list. "
    "Return ONLY valid JSON: {\"matched\": true, \"index\": <0-based int>} "
    "or {\"matched\": false} if no FAQ closely matches."
)


class FAQAgent(AgentBase):
    def process(
        self,
        utterance: str,
        internal_state: dict,
        config: dict,
        history: list[dict],
    ) -> SubagentResponse:
        faqs = config.get("faqs", [])
        if not faqs:
            return SubagentResponse(
                status=AgentStatus.COMPLETED,
                speak="I'm sorry, I don't have an answer to that right now.",
                requires_router_resume=True,
                internal_state=internal_state,
            )

        faq_list = "\n".join(
            f"{i}. Q: {f['question']}" for i, f in enumerate(faqs)
        )
        try:
            match = llm_json_call(
                _MATCH_SYSTEM,
                f"FAQ list:\n{faq_list}\n\nCaller asked: \"{utterance}\"",
            )
            if match.get("matched") and match.get("index") is not None:
                idx = int(match["index"])
                if 0 <= idx < len(faqs):
                    faq = faqs[idx]
                    answer = llm_text_call(
                        "You are an AI receptionist answering a caller's question. Be concise and conversational.",
                        f"FAQ answer to use: {faq['answer']}\nRephrase naturally for voice. Two sentences max.",
                    )
                    return SubagentResponse(
                        status=AgentStatus.COMPLETED,
                        speak=answer,
                        requires_router_resume=True,
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
