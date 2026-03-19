"""Narrative collection agent — listens to a free-form caller description.

Unlike DataCollectionAgent (one field = one question/answer cycle), this agent
accumulates free-form speech across multiple turns:

  caller speaks → agent responds with a filler → caller speaks → ...
  → narrative seems complete → agent asks "Is there anything else?" →
  caller says no → agent summarises + extracts case_type → COMPLETED

Interruption handling
---------------------
narrative_collection is NOT interrupt_eligible, which means it IS the
active_primary() while collecting.  When the router sees a question mid-
narrative (e.g. "What are your fees?") it routes to faq with interrupt=True,
pushing narrative_collection onto the resume stack.  After faq completes,
on_complete=Edge("resume") re-invokes narrative_collection with utterance="".
internal_state["segments"] is preserved across interruptions so the full
narrative is always concatenated in the final output.
"""
from __future__ import annotations

import logging
import random

from app.agents.base import AgentBase, AgentStatus, SubagentResponse
from app.agents.llm_utils import llm_json_call, llm_text_call

logger = logging.getLogger(__name__)

# Minimum segments to collect before checking for narrative completion.
# Prevents asking "Is there anything else?" after a single short sentence.
_MIN_SEGMENTS = 2
_MIN_WORDS_IN_SEGMENT = 4

# After asking "Is there anything else?" this many times, complete regardless.
# Prevents an infinite loop when the LLM repeatedly misclassifies "No" as not-done.
_MAX_DONE_ASKS = 2

# Filler phrases spoken while the caller is mid-narrative.
_FILLERS = [
    "I see, please go on.",
    "Understood, continue.",
    "Got it.",
    "I'm listening, please go ahead.",
    "Okay, continue.",
]

_COMPLETION_CHECK_SYSTEM = (
    "You decide if a caller has given a reasonably complete description of their "
    "legal matter — not just an opening sentence but enough context to understand "
    "the situation. Reply ONLY valid JSON: {\"complete\": true} or {\"complete\": false}."
)

_DONE_INTENT_SYSTEM = (
    "The caller was asked: 'Is there anything else you would like to add about your matter?'\n"
    "Determine whether they want to STOP (nothing more to add) or CONTINUE (have more to say).\n\n"
    "STOP \u2192 {\"done\": true}\n"
    "  Examples: 'No' | 'No, that\\'s all' | 'Nope' | 'That\\'s it' | "
    "'Nothing else' | 'I\\'m done' | 'That covers it'\n\n"
    "CONTINUE \u2192 {\"done\": false}\n"
    "  Examples: 'Yes actually' | 'Wait, I should mention' | "
    "'There\\'s more' | 'I forgot to say'\n\n"
    "Reply ONLY valid JSON: {\"done\": true} or {\"done\": false}."
)

_SUMMARY_SYSTEM = (
    "Summarise a caller's legal matter in 2–3 concise sentences suitable for "
    "an intake coordinator to read quickly. Then identify the single best-matching "
    "practice area from the list provided, or 'unknown' if none match clearly. "
    "Return ONLY valid JSON: "
    "{\"summary\": \"<2-3 sentence summary>\", \"case_type\": \"<practice area or unknown>\"}"
)


class NarrativeCollectionAgent(AgentBase):
    """
    Collects a free-form caller narrative over multiple STT turns.

    Internal state keys
    -------------------
    stage         : "collecting" | "asking_done"
    segments      : list[str]   — accumulated caller utterances (never reset)
    summary       : str | None  — populated on COMPLETED
    case_type     : str | None  — populated on COMPLETED
    """

    def process(
        self,
        utterance: str,
        internal_state: dict,
        config: dict,
        history: list[dict],
    ) -> SubagentResponse:
        # ── Initialise on first call ────────────────────────────────────────
        if not internal_state:
            internal_state = {
                "stage": "collecting",
                "segments": [],
                "summary": None,
                "case_type": None,
                # how many times "Is there anything else?" has been asked
                "done_ask_count": 0,
                # meaningful-segment count at the time of the last completion check;
                # prevents re-asking "anything else?" without any new content added
                "segments_at_done_check": 0,
            }

        stage = internal_state.get("stage", "collecting")
        segments: list[str] = internal_state.get("segments", [])

        # ── Resume invocation (empty utterance after FAQ interrupt) ─────────
        if not utterance.strip():
            if stage == "asking_done":
                return SubagentResponse(
                    status=AgentStatus.IN_PROGRESS,
                    speak="Is there anything else you'd like to add?",
                    internal_state=internal_state,
                )
            # stage == "collecting"
            resume_prompt = self._get_resume_prompt(config)
            return SubagentResponse(
                status=AgentStatus.IN_PROGRESS,
                speak=resume_prompt,
                internal_state=internal_state,
            )

        # ── Stage: asking_done — did the caller say yes or no? ──────────────
        if stage == "asking_done":
            # Safety cap: if we've already asked the maximum number of times,
            # complete regardless of what the caller says next.
            done_ask_count = internal_state.get("done_ask_count", 0)
            if done_ask_count >= _MAX_DONE_ASKS:
                logger.info(
                    "NarrativeCollection: max done asks (%d) reached — completing", _MAX_DONE_ASKS
                )
                return self._complete(segments, config, internal_state)

            done = self._detect_done_intent(utterance)
            if done:
                return self._complete(segments, config, internal_state)
            else:
                # Caller wants to add more.
                # Do NOT append the gate response ("No", "Actually...", etc.) to
                # segments — it is a conversational reply, not narrative content.
                # The caller's next actual statement will be appended normally.
                internal_state["stage"] = "collecting"
                logger.info(
                    "NarrativeCollection: caller wants to continue | segments=%d", len(segments)
                )
                return SubagentResponse(
                    status=AgentStatus.IN_PROGRESS,
                    speak="Of course, please go ahead.",
                    internal_state=internal_state,
                )

        # ── Stage: collecting — accumulate and check for completion ─────────
        segments.append(utterance)
        internal_state["segments"] = segments
        logger.info(
            "NarrativeCollection: accumulated segment %d | words=%d",
            len(segments), len(utterance.split()),
        )

        # Only check for completion once we have enough meaningful content AND
        # at least one new meaningful segment has been added since we last asked
        # "Is there anything else?" — prevents immediately re-asking after the
        # caller says they want to continue.
        meaningful_count = len([s for s in segments if len(s.split()) >= _MIN_WORDS_IN_SEGMENT])
        segments_at_done_check = internal_state.get("segments_at_done_check", 0)
        has_new_content = meaningful_count > segments_at_done_check

        if has_new_content and self._has_enough_content(segments) and self._narrative_complete(segments):
            done_ask_count = internal_state.get("done_ask_count", 0) + 1
            internal_state["done_ask_count"] = done_ask_count
            internal_state["segments_at_done_check"] = meaningful_count
            internal_state["stage"] = "asking_done"
            logger.info(
                "NarrativeCollection: narrative seems complete — asking if done (ask #%d)",
                done_ask_count,
            )
            return SubagentResponse(
                status=AgentStatus.IN_PROGRESS,
                speak="Is there anything else you'd like to add?",
                internal_state=internal_state,
            )

        # Still collecting — respond with a filler
        filler = self._pick_filler(len(segments))
        return SubagentResponse(
            status=AgentStatus.IN_PROGRESS,
            speak=filler,
            internal_state=internal_state,
        )

    # ── Private helpers ────────────────────────────────────────────────────

    def _get_resume_prompt(self, config: dict) -> str:
        topic = config.get("assistant", {}).get("narrative_topic", "your matter")
        return f"Please go ahead, I'm still listening about {topic}."

    def _has_enough_content(self, segments: list[str]) -> bool:
        """At least _MIN_SEGMENTS segments, each with enough words."""
        meaningful = [s for s in segments if len(s.split()) >= _MIN_WORDS_IN_SEGMENT]
        return len(meaningful) >= _MIN_SEGMENTS

    def _narrative_complete(self, segments: list[str]) -> bool:
        """Ask LLM whether the narrative seems like a complete description."""
        # Use only the last two segments for the completion check to keep the
        # prompt short and fast.
        recent = " ".join(segments[-2:])
        total_words = sum(len(s.split()) for s in segments)
        try:
            result = llm_json_call(
                _COMPLETION_CHECK_SYSTEM,
                f"Total words collected so far: {total_words}\nLast caller statement: \"{recent}\"",
                max_tokens=64,
            )
            complete = bool(result.get("complete", False))
            logger.debug("NarrativeCollection: completion_check=%s", complete)
            return complete
        except Exception as exc:
            logger.warning("NarrativeCollection: completion check failed: %s", exc)
            return False

    def _detect_done_intent(self, utterance: str) -> bool:
        """Return True if the caller indicates they have nothing more to add."""
        try:
            result = llm_json_call(
                _DONE_INTENT_SYSTEM,
                f"Caller said: \"{utterance}\"",
                max_tokens=32,
            )
            done = bool(result.get("done", False))
            logger.debug("NarrativeCollection: done_intent=%s for %r", done, utterance[:60])
            return done
        except Exception as exc:
            logger.warning("NarrativeCollection: done intent detection failed: %s", exc)
            # Default to "not done" on error so we don't prematurely end
            return False

    def _complete(
        self,
        segments: list[str],
        config: dict,
        internal_state: dict,
    ) -> SubagentResponse:
        """Summarise the narrative and return COMPLETED."""
        full_narrative = " ".join(segments)
        practice_areas = config.get("practice_areas", [])
        areas_str = ", ".join(practice_areas) if practice_areas else "general legal matters"

        summary = "Unable to summarise."
        case_type = "unknown"
        try:
            result = llm_json_call(
                _SUMMARY_SYSTEM,
                f"Practice areas the firm handles: {areas_str}\n\nCaller narrative:\n{full_narrative}",
            )
            summary = result.get("summary", summary)
            case_type = result.get("case_type", case_type)
        except Exception as exc:
            logger.warning("NarrativeCollection: summarisation failed: %s", exc)

        internal_state["summary"] = summary
        internal_state["case_type"] = case_type
        internal_state["stage"] = "done"

        logger.info(
            "NarrativeCollection: COMPLETED | segments=%d | case_type=%s | summary=%r",
            len(segments), case_type, summary[:80],
        )

        speak = llm_text_call(
            "You are an AI receptionist at a law firm. Acknowledge that you've noted "
            "the caller's matter and will proceed. One warm, professional sentence. "
            "Do not repeat details back.",
            f"The caller described a {case_type} matter.",
        )

        return SubagentResponse(
            status=AgentStatus.COMPLETED,
            speak=speak or "Thank you for sharing that. I've noted the details of your matter.",
            collected={
                "narrative_summary": summary,
                "case_type": case_type,
                "full_narrative": full_narrative,
            },
            internal_state=internal_state,
        )

    def _pick_filler(self, segment_count: int) -> str:
        """Pick a contextually appropriate filler based on how far along we are."""
        # On the last filler before we'd check completion, nudge caller to wrap up
        if segment_count >= _MIN_SEGMENTS + 2:
            return random.choice([
                "I see. Is there anything else you'd like to add about your situation?",
                "Got it. Please continue if there's more you'd like to share.",
            ])
        return random.choice(_FILLERS)
