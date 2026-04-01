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
You are a voice assistant response polisher for a law firm intake call.

You will receive CALLER NAME, CALLER SITUATION, and RESPONSE.
Return the RESPONSE with a small, natural warmth improvement — OR return it \
exactly unchanged if no improvement fits.

ALWAYS output the full response text. Never output an empty string.

Allowed changes (pick at most one):
- Add the caller's first name naturally: "Of course, Sarah — ..." or "Sure thing, Sarah — ..."
- Add a short warm opener (3–8 words) chosen to fit the context naturally.
- Soften a cold opening word: "What's" → "And what's", "Could you" → "Could I also ask you"

Vary your word choice. NEVER use "Absolutely" — it sounds robotic with repetition.
Example openers (use these as inspiration, not templates to copy):
  "Of course — ...", "Sure — ...", "Happy to help — ...", "Got it — ...",
  "Certainly — ...", "Of course, [name] — ...", "Thanks for that — ...",
  "Great — ...", "Perfect — ...", "Noted — ...", "Understood — ..."

Do not change any field values, dates, questions, or factual content.
Do not add more than 10 words to the response.
Do not add a name or empathy phrase if the response already starts with one.
Output ONLY the spoken text — no labels, no JSON, no explanation.
"""


# Responses longer than this are multi-agent composites.  Running them through
# the filter risks the LLM early-stopping mid-sentence and adds latency for no gain.
_MAX_WORDS_FOR_FILTER = 40


def _should_skip(speak_text: str) -> bool:
    """Return True for responses that should not be filtered."""
    text = speak_text.strip().lower()
    words = text.split()
    if len(words) < 6:
        return True
    if len(words) > _MAX_WORDS_FOR_FILTER:
        return True
    for phrase in _SKIP_PHRASES:
        if phrase in text:
            return True
    return False


def apply_empathy_filter(
    speak_text: str,
    collected: dict,
    transcript_turns: list[dict],
    agents_ran: frozenset[str] = frozenset(),
) -> str:
    """Return an empathy-enhanced version of speak_text, or the original on skip/error."""
    if not speak_text or not speak_text.strip():
        return speak_text

    # EmpathyAgent already produced a warm paragraph — adding another layer is
    # redundant and risks truncation when the LLM reproduces a long composite.
    if "empathy" in agents_ran:
        logger.debug("EmpathyFilter: skipping — EmpathyAgent already ran this turn")
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

    # Skip if there's nothing to personalise with — LLM would just echo the response
    if not first_name and not situation:
        return speak_text

    user_msg = (
        f"CALLER NAME: {first_name or '(unknown)'}\n"
        f"CALLER SITUATION: {situation or '(not yet described)'}\n"
        f"RESPONSE: {speak_text}"
    )

    try:
        enhanced = llm_text_call(_EMPATHY_SYSTEM, user_msg, max_tokens=512, tag="empathy_filter")
        if enhanced and enhanced.strip():
            logger.debug("EmpathyFilter: %r → %r", speak_text[:60], enhanced[:60])
            return enhanced.strip()
    except Exception as exc:
        logger.warning("EmpathyFilter: LLM call failed (%s) — returning original", exc)

    return speak_text
