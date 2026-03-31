"""Fallback agent — context-aware business Q&A with guardrails.

Receives the full business context (practice areas, policy documents, context
files, FAQs, parameters) and attempts to answer the caller's question using
only that grounded knowledge.  Falls back to a "team member will follow up"
response only when the question is genuinely unanswerable from context.
"""
from __future__ import annotations

import logging
from app.agents.base import AgentBase, AgentStatus, SubagentResponse
from app.agents.llm_utils import llm_text_call, ConversationHistory

logger = logging.getLogger(__name__)

_MAX_DOC_CHARS = 1500   # per policy doc / context file excerpt
_MAX_FAQ_CHARS = 3000   # total FAQ block budget


def _build_business_context(config: dict) -> str:
    """Assemble a structured knowledge block from all available config."""
    parts: list[str] = []

    assistant = config.get("assistant", {})
    persona = assistant.get("persona_name", "the firm")
    parts.append(f"FIRM: {persona}")
    if assistant.get("system_prompt"):
        parts.append(f"FIRM DESCRIPTION:\n{assistant['system_prompt'][:800]}")

    # Practice areas — names and descriptions only (policy docs are often large)
    areas = config.get("practice_areas", [])
    if areas:
        lines = []
        for a in areas:
            line = f"  • {a['name']}"
            if a.get("description"):
                line += f": {a['description']}"
            lines.append(line)
        parts.append("PRACTICE AREAS:\n" + "\n".join(lines))

    # Global policy documents
    global_docs = config.get("global_policy_documents", [])
    if global_docs:
        doc_parts = []
        for doc in global_docs:
            content = (doc.get("content") or "")[:_MAX_DOC_CHARS]
            if content:
                doc_parts.append(f"[{doc['name']}]\n{content}")
        if doc_parts:
            parts.append("POLICY DOCUMENTS:\n" + "\n\n".join(doc_parts))

    # Uploaded context files
    context_files = config.get("context_files", [])
    if context_files:
        cf_parts = []
        for cf in context_files:
            content = (cf.get("content") or "")[:_MAX_DOC_CHARS]
            if content:
                label = cf.get("name") or cf.get("description") or "Document"
                cf_parts.append(f"[{label}]\n{content}")
        if cf_parts:
            parts.append("REFERENCE DOCUMENTS:\n" + "\n\n".join(cf_parts))

    # FAQs (budget-capped)
    faqs = config.get("faqs", [])
    if faqs:
        faq_lines: list[str] = []
        total = 0
        for f in faqs:
            entry = f"  Q: {f['question']}\n  A: {f['answer']}"
            total += len(entry)
            if total > _MAX_FAQ_CHARS:
                break
            faq_lines.append(entry)
        if faq_lines:
            parts.append("FREQUENTLY ASKED QUESTIONS:\n" + "\n\n".join(faq_lines))

    # Parameters being collected — lets the agent answer collection-process
    # questions like "Do you need my email?" or "Is my phone number required?"
    parameters = config.get("parameters", [])
    if parameters:
        param_lines = [
            f"  • {p['display_label']} ({'required' if p.get('required', True) else 'optional'})"
            for p in parameters
        ]
        parts.append("INFORMATION BEING COLLECTED ON THIS CALL:\n" + "\n".join(param_lines))

    return "\n\n".join(parts)


_SYSTEM_TEMPLATE = """\
You are {persona}, an AI receptionist. You are in the MIDDLE of an ongoing voice \
call — the caller is already speaking with you. Do NOT greet them, say "Hi", \
say "Thanks for calling", or introduce yourself in any way.

The caller has just asked a question mid-conversation. You MUST give an extremely \
conservative, generic response — do not attempt to answer the question itself.

GUARDRAILS — you MUST follow all of these without exception:
1. NEVER answer the question. Always defer to the team.
2. Respond in exactly 1 sentence: acknowledge you don't have that information \
right now and that you have noted it for the team to follow up.
   Example: "I don't have that information on hand, but I've noted your question \
and a team member will make sure to follow up with you on that."
3. Do NOT greet, introduce, or re-introduce yourself mid-call.
4. Do NOT speculate, invent, or answer from the business context below.
5. Keep your response to 1 sentence — warm, conversational, voice-call style.

\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501
BUSINESS CONTEXT
\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501
{business_context}
"""


class FallbackAgent(AgentBase):
    def process(
        self,
        utterance: str,
        internal_state: dict,
        config: dict,
        history: list[dict],
    ) -> SubagentResponse:
        existing_notes = internal_state.get("notes", "")

        persona = config.get("assistant", {}).get("persona_name", "the firm's receptionist")
        business_context = _build_business_context(config)

        system = _SYSTEM_TEMPLATE.format(
            persona=persona,
            business_context=business_context,
        )
        user_msg = f'CALLER ASKED: "{utterance}"'
        llm_history = ConversationHistory.from_list(internal_state.get("llm_history"))

        try:
            speak = llm_text_call(system, user_msg, max_tokens=2048, history=llm_history, tag="fallback")
        except Exception as exc:
            logger.warning("FallbackAgent LLM call failed: %s", exc)
            speak = (
                "I'm sorry, I don't have that information handy — "
                "a team member will be in touch to help you."
            )

        llm_history.add("user", user_msg)
        llm_history.add("assistant", speak)
        internal_state["llm_history"] = llm_history.to_list()

        # Accumulate notes and structured question list for post-call review
        new_note = f"Question: {utterance}"
        updated_notes = (existing_notes + "\n" + new_note).strip() if existing_notes else new_note
        internal_state["notes"] = updated_notes
        caller_questions: list = internal_state.get("caller_questions", [])
        caller_questions.append(utterance)
        internal_state["caller_questions"] = caller_questions

        logger.info("FallbackAgent: answered utterance=%r speak=%r", utterance[:80], speak[:80])

        return SubagentResponse(
            status=AgentStatus.COMPLETED,
            speak=speak,
            notes=updated_notes,
            internal_state=internal_state,
        )
