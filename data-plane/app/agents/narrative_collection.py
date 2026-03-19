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

# Filler phrases spoken while the caller is mid-narrative.
# Must NOT include "Is there anything else?" — that question is only asked
# once, explicitly, when transitioning to asking_done.
_FILLERS = [
    "I see, please go on.",
    "Understood, continue.",
    "Got it.",
    "I'm listening, please go ahead.",
    "Okay, continue.",
]

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
    stage    : "collecting" | "asking_done"
    segments : list[str] — accumulated caller utterances (never reset)
    summary  : str | None — populated on COMPLETED
    case_type: str | None — populated on COMPLETED

    Flow (simplified — ask once only)
    ----------------------------------
    1. Collect caller utterances, respond with fillers.
    2. Once _has_enough_content, ask "Is there anything else?" (ONCE).
    3. Caller responds:
       - Sounds like "done" → complete with existing segments.
       - Sounds like "more to add" → collect that utterance, then complete.
    There is no loop.  The agent asks at most one "anything else?" per call.
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

        # ── Stage: asking_done — one response, then always complete ────────
        if stage == "asking_done":
            done = self._detect_done_intent(utterance)
            if not done:
                # Caller has something to add — collect it as a final segment.
                segments.append(utterance)
                internal_state["segments"] = segments
                logger.info(
                    "NarrativeCollection: collecting final addition | segments=%d", len(segments)
                )
            # Complete regardless — we only ask once.
            return self._complete(segments, config, internal_state)

        # ── Stage: collecting — accumulate and check for completion ─────────
        segments.append(utterance)
        internal_state["segments"] = segments
        logger.info(
            "NarrativeCollection: accumulated segment %d | words=%d",
            len(segments), len(utterance.split()),
        )

        # Once we have enough content, ask "Is there anything else?" — once only.
        if self._has_enough_content(segments):
            internal_state["stage"] = "asking_done"
            logger.info("NarrativeCollection: enough content — asking if done")
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
        """At least _MIN_SEGMENTS segments with enough words each."""
        meaningful = [s for s in segments if len(s.split()) >= _MIN_WORDS_IN_SEGMENT]
        return len(meaningful) >= _MIN_SEGMENTS

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
            # narrative_summary is the only field shown to the caller in the UI
            collected={
                "narrative_summary": summary,
            },
            # full_narrative and case_type are backend-only: logged + passed to webhook, not displayed
            hidden_collected={
                "full_narrative": full_narrative,
                "case_type": case_type,
            },
            internal_state=internal_state,
        )

    def _pick_filler(self, segment_count: int) -> str:
        return random.choice(_FILLERS)
