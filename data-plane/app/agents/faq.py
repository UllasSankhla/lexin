"""FAQ agent — answers one or more questions from the curated FAQ list."""
from __future__ import annotations

import logging
from app.agents.base import AgentBase, AgentStatus, SubagentResponse
from app.agents.llm_utils import llm_structured_call, llm_text_call
from app.agents.agent_schemas import FAQMultiMatchResult

logger = logging.getLogger(__name__)

_MULTI_MATCH_SYSTEM = """\
The caller may have asked one or more questions in a single utterance.
Your job is to identify each distinct question and classify it.

For each question found:
  - Set is_legal=true if it is a substantive legal question about the caller's
    specific matter (H1B portability, I-140 status, visa strategy, EAD renewals,
    employment law case assessment, immigration law specifics, legal outcomes,
    case strategy). These must be handled by the attorney, not the FAQ list.
  - Set is_legal=false for general questions about the firm (fees, hours,
    location, practice areas, process, booking).
  - Set faq_index to the 0-based index of the closest matching FAQ from the
    list provided. Set to null if no FAQ closely matches or if is_legal=true.

Return ONLY valid JSON matching the schema. If the utterance has no questions,
return {"questions": []}.
"""


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
                internal_state=internal_state,
            )

        # ── Single multi-match call: detect questions, deflect legal, match FAQs ──
        faq_list = "\n".join(f"{i}. Q: {f['question']}" for i, f in enumerate(faqs))
        user_msg = f"FAQ list:\n{faq_list}\n\nCaller said: \"{utterance}\""

        try:
            result = llm_structured_call(
                _MULTI_MATCH_SYSTEM,
                user_msg,
                FAQMultiMatchResult,
                tag="faq_multi_match",
            )
        except Exception as exc:
            logger.warning("FAQAgent multi-match failed: %s", exc)
            return SubagentResponse(
                status=AgentStatus.FAILED,
                speak="",
                internal_state=internal_state,
            )

        if not result.questions:
            return SubagentResponse(
                status=AgentStatus.FAILED,
                speak="",
                internal_state=internal_state,
            )

        # ── Separate legal deflects from answerable FAQ matches ───────────────────
        has_legal = any(q.is_legal for q in result.questions)
        matched_indices = [
            q.faq_index for q in result.questions
            if not q.is_legal and q.faq_index is not None and 0 <= q.faq_index < len(faqs)
        ]

        logger.info(
            "FAQAgent: %d question(s) detected — %d legal deflect(s), %d FAQ match(es): indices=%s",
            len(result.questions), sum(q.is_legal for q in result.questions),
            len(matched_indices), matched_indices,
        )

        # Deduplicate while preserving order (same FAQ matched twice → answer once)
        seen: set[int] = set()
        unique_indices = [i for i in matched_indices if not (i in seen or seen.add(i))]
        matched_faqs = [faqs[i] for i in unique_indices]

        # ── Build combined answer ─────────────────────────────────────────────────
        answer_parts: list[str] = []

        if matched_faqs:
            if len(matched_faqs) == 1:
                # Single match — rephrase for voice
                answer_parts.append(matched_faqs[0]["answer"])
                combine_instruction = "Rephrase this answer naturally for voice. Two sentences max."
            else:
                # Multiple matches — ask LLM to combine into one flowing response
                answers_block = "\n".join(
                    f"Q: {faqs[i]['question']}\nA: {faqs[i]['answer']}"
                    for i in unique_indices
                )
                answer_parts.append(answers_block)
                combine_instruction = (
                    "The caller asked multiple questions. Combine all the answers above into "
                    "a single natural voice response that addresses each question in order. "
                    "Be concise — no more than 3-4 sentences total."
                )

            try:
                speak = llm_text_call(
                    "You are an AI receptionist answering a caller's question. "
                    "Be concise and conversational.",
                    f"{combine_instruction}\n\n{answer_parts[0]}",
                    tag="faq_answer",
                )
            except Exception as exc:
                logger.warning("FAQAgent answer generation failed: %s", exc)
                # Fall back to raw answer text
                speak = " ".join(f["answer"] for f in matched_faqs)
        else:
            speak = ""

        # ── Append legal deflect notice if needed ─────────────────────────────────
        if has_legal:
            deflect = (
                "That's a great question for our legal team — I'll make sure to note it "
                "so the attorney can address it directly in your consultation."
            )
            speak = (speak + " " + deflect).strip() if speak else deflect
            logger.info("FAQAgent: appended legal deflect notice")

        if not speak:
            return SubagentResponse(
                status=AgentStatus.FAILED,
                speak="",
                internal_state=internal_state,
            )

        return SubagentResponse(
            status=AgentStatus.COMPLETED,
            speak=speak,
            internal_state=internal_state,
        )
