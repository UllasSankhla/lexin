"""Narrative collection agent — listens to a free-form caller description.

Unlike DataCollectionAgent (one field = one question/answer cycle), this agent
accumulates free-form speech across multiple turns:

  caller speaks → agent responds with a filler → caller speaks → ...
  → narrative seems complete → agent asks "Is there anything else?" →
  caller says no → stores full_narrative → COMPLETED

No summarisation is done here. The full raw narrative is passed to
IntakeQualificationAgent so it can make a richer decision using the
complete caller context. Post-call summarisation is handled by
SummarizationTool at call end.

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
from app.agents.llm_utils import llm_json_call, ConversationHistory

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


class NarrativeCollectionAgent(AgentBase):
    """
    Collects a free-form caller narrative over multiple STT turns.

    Internal state keys
    -------------------
    stage    : "collecting" | "asking_done"
    segments : list[str] — accumulated caller utterances (never reset)

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
            }

        stage = internal_state.get("stage", "collecting")
        segments: list[str] = internal_state.get("segments", [])
        llm_history = ConversationHistory.from_list(internal_state.get("llm_history"))

        # ── Resume invocation (empty utterance after FAQ interrupt or first call) ─
        if not utterance.strip():
            if stage == "asking_done":
                return SubagentResponse(
                    status=AgentStatus.IN_PROGRESS,
                    speak="Is there anything else you'd like to add?",
                    internal_state=internal_state,
                )
            # stage == "collecting" — first call has no segments yet, later calls are resumes
            if not segments:
                prompt = self._get_opening_prompt(config)
            else:
                prompt = self._get_resume_prompt(config)
            return SubagentResponse(
                status=AgentStatus.IN_PROGRESS,
                speak=prompt,
                internal_state=internal_state,
            )

        # ── Stage: asking_done — one response, then always complete ────────
        if stage == "asking_done":
            done = self._detect_done_intent(utterance, llm_history)
            llm_history.add("user", f'Caller said: "{utterance}"')
            llm_history.add("assistant", '{"done": ' + str(done).lower() + '}')
            internal_state["llm_history"] = llm_history.to_list()
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
            speak = "Is there anything else you'd like to add?"
            llm_history.add("user", f'Caller said: "{utterance}"')
            llm_history.add("assistant", speak)
            internal_state["llm_history"] = llm_history.to_list()
            return SubagentResponse(
                status=AgentStatus.IN_PROGRESS,
                speak=speak,
                internal_state=internal_state,
            )

        # Still collecting — respond with a filler
        filler = self._pick_filler(len(segments))
        llm_history.add("user", f'Caller said: "{utterance}"')
        llm_history.add("assistant", filler)
        internal_state["llm_history"] = llm_history.to_list()
        return SubagentResponse(
            status=AgentStatus.IN_PROGRESS,
            speak=filler,
            internal_state=internal_state,
        )

    # ── Private helpers ────────────────────────────────────────────────────

    def _get_opening_prompt(self, config: dict) -> str:
        topic = config.get("assistant", {}).get("narrative_topic", "your matter")
        return (
            f"I'd now like to understand {topic} in more detail. "
            "Please take your time and share as much information as you can — "
            "the more you tell us, the better our team can understand how to help you."
        )

    def _get_resume_prompt(self, config: dict) -> str:
        topic = config.get("assistant", {}).get("narrative_topic", "your matter")
        return f"Please go ahead, I'm still listening about {topic}."

    def _has_enough_content(self, segments: list[str]) -> bool:
        """At least _MIN_SEGMENTS segments with enough words each."""
        meaningful = [s for s in segments if len(s.split()) >= _MIN_WORDS_IN_SEGMENT]
        return len(meaningful) >= _MIN_SEGMENTS

    def _detect_done_intent(self, utterance: str, history: ConversationHistory) -> bool:
        """Return True if the caller indicates they have nothing more to add."""
        try:
            result = llm_json_call(
                _DONE_INTENT_SYSTEM,
                f"Caller said: \"{utterance}\"",
                max_tokens=1024,
                history=history,
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
        """Store the full narrative and return COMPLETED. No summarisation here —
        that happens at call end via SummarizationTool."""
        full_narrative = " ".join(segments)
        internal_state["stage"] = "done"

        logger.info(
            "NarrativeCollection: COMPLETED | segments=%d | chars=%d",
            len(segments), len(full_narrative),
        )

        return SubagentResponse(
            status=AgentStatus.COMPLETED,
            speak="Thank you for sharing that. Let me check if we can assist with your matter.",
            hidden_collected={
                "full_narrative": full_narrative,
            },
            internal_state=internal_state,
        )

    def _pick_filler(self, segment_count: int) -> str:
        return random.choice(_FILLERS)
