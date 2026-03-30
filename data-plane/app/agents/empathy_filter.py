"""Empathy filter — lightly enhances every outbound AI response with a brief
empathetic touch: caller's name, acknowledgement of their situation, etc.

This is NOT the EmpathyAgent (which generates a dedicated empathy paragraph on
first contact).  This filter post-processes the assembled speak_text for every
turn, adding at most a short phrase (3–8 tokens) so the whole conversation feels
warm without being repetitive or verbose.

Responses that are skipped (returned unchanged):
  - Very short utterances (< 6 words) — e.g. simple "Got it." filler
  - Confirmation readbacks ("Is that correct?", "Shall I confirm?")
  - Error/recovery phrases
  - When the LLM call fails (fail-open: return original)
"""
from __future__ import annotations

import logging

from app.agents.llm_utils import llm_text_call

logger = logging.getLogger(__name__)

_SKIP_PHRASES = (
    "is that correct?",
    "shall i confirm?",
    "could you try again?",
    "could you please say that again?",
    "didn't quite get that",
)

_EMPATHY_SYSTEM = """\
You are an empathy enhancer for a voice intake assistant at a law firm.

You will receive:
- CALLER NAME: the caller's first name (may be empty if not yet collected)
- CALLER SITUATION: a one-line summary of why they are calling (may be empty)
- RESPONSE: the AI's draft response to speak aloud

Your task: return a lightly enhanced version of RESPONSE that feels warmer and \
more human. You may:
- Add the caller's first name naturally at the start (e.g. "Of course, Sarah — ...")
- Add a single brief acknowledgement phrase (3–8 words max) at the start
- Slightly rephrase a cold opening word ("What's" → "And what's", "Could I" → \
"Could I also", etc.)

Rules — strictly follow ALL of these:
- Do NOT change any factual content, field values, or questions
- Do NOT add empathy if the response already opens with empathy or the caller's name
- Do NOT add empathy if the response is a slot/booking readback or a yes/no question
- Do NOT make the response longer than the original by more than 10 words
- If no enhancement is natural or needed, return the RESPONSE unchanged
- Output ONLY the final spoken text — no labels, no explanation, no JSON
"""


def _should_skip(speak_text: str) -> bool:
    """Return True for responses that should not be filtered."""
    text = speak_text.strip().lower()
    if len(text.split()) < 6:
        return True
    for phrase in _SKIP_PHRASES:
        if phrase in text:
            return True
    return False


def apply_empathy_filter(
    speak_text: str,
    collected: dict,
    transcript_turns: list[dict],
) -> str:
    """Return an empathy-enhanced version of speak_text, or the original on skip/error."""
    if not speak_text or not speak_text.strip():
        return speak_text

    if _should_skip(speak_text):
        return speak_text

    # Build context for the LLM
    first_name = ""
    full_name = collected.get("full_name", "")
    if full_name:
        first_name = full_name.strip().split()[0]

    # Derive a one-line situation summary from recent history
    situation = ""
    for turn in reversed(transcript_turns):
        if turn.get("role") == "user":
            candidate = turn.get("content", "").strip()
            if len(candidate.split()) >= 8:
                situation = candidate[:200]
                break

    user_msg = (
        f"CALLER NAME: {first_name or '(unknown)'}\n"
        f"CALLER SITUATION: {situation or '(not yet described)'}\n"
        f"RESPONSE: {speak_text}"
    )

    try:
        enhanced = llm_text_call(_EMPATHY_SYSTEM, user_msg, max_tokens=256, tag="empathy_filter")
        if enhanced and enhanced.strip():
            logger.debug("EmpathyFilter: %r → %r", speak_text[:60], enhanced[:60])
            return enhanced.strip()
    except Exception as exc:
        logger.warning("EmpathyFilter: LLM call failed (%s) — returning original", exc)

    return speak_text
