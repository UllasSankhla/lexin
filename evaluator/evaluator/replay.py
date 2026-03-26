"""Replay engine — drives the real agent pipeline with caller utterances from a TestCase.

No calendar API calls are made (MockSchedulingAgent stubs slots and bookings).
No webhook or summarization tools are invoked.
All LLM calls (router, agents) use the real Cerebras client.
"""
from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ── Path setup — inject data-plane package into sys.path ─────────────────────
_DATA_PLANE = Path(__file__).parents[2] / "data-plane"
if str(_DATA_PLANE) not in sys.path:
    sys.path.insert(0, str(_DATA_PLANE))

# Provide fallback env vars so data-plane settings load without raising.
# The real API key is picked up from the environment if already set.
os.environ.setdefault("DEEPGRAM_API_KEY",       "eval-not-used")
os.environ.setdefault("CONTROL_PLANE_API_KEY",  "eval-not-used")

# Must import AFTER the path and env setup above.
from app.agents.base import AgentStatus, SubagentResponse          # noqa: E402
from app.agents.data_collection import DataCollectionAgent          # noqa: E402
from app.agents.narrative_collection import NarrativeCollectionAgent  # noqa: E402
from app.agents.intake_qualification import IntakeQualificationAgent  # noqa: E402
from app.agents.faq import FAQAgent                                 # noqa: E402
from app.agents.context_docs import ContextDocsAgent                # noqa: E402
from app.agents.fallback import FallbackAgent                       # noqa: E402
from app.agents.farewell import FarewellAgent                       # noqa: E402
from app.agents.scheduling import SchedulingAgent, _detect_confirmation  # noqa: E402
from app.agents.graph_config import APPOINTMENT_BOOKING             # noqa: E402
from app.agents.workflow import WorkflowGraph                       # noqa: E402
from app.agents.router import Router                                # noqa: E402
from app.services.calendar_service import TimeSlot                  # noqa: E402
import app.agents.scheduling as _sched_module                       # noqa: E402

from evaluator.models import TestCase, ReplayTurn, ReplayResult

logger = logging.getLogger(__name__)

_MAX_TURNS = 60     # safety cap on total turns processed
_MAX_CHAIN  = 5     # max edge-chain depth per single utterance


# ── Fake calendar data ────────────────────────────────────────────────────────

_FAKE_SLOTS = [
    TimeSlot(
        slot_id="eval-slot-1",
        description="Monday at 10:00 AM",
        start=datetime(2025, 3, 31, 15, 0, tzinfo=timezone.utc),
        end=datetime(2025, 3, 31, 16, 0, tzinfo=timezone.utc),
    ),
    TimeSlot(
        slot_id="eval-slot-2",
        description="Tuesday at 2:00 PM",
        start=datetime(2025, 4, 1, 19, 0, tzinfo=timezone.utc),
        end=datetime(2025, 4, 1, 20, 0, tzinfo=timezone.utc),
    ),
    TimeSlot(
        slot_id="eval-slot-3",
        description="Wednesday at 11:00 AM",
        start=datetime(2025, 4, 2, 16, 0, tzinfo=timezone.utc),
        end=datetime(2025, 4, 2, 17, 0, tzinfo=timezone.utc),
    ),
]

# Patch book_time_slot at module level so no Calendly API calls are made.
_sched_module.book_time_slot = lambda slot, collected, config: {
    "booking_id": "EVAL-FAKE-BOOKING",
    "slot_description": getattr(slot, "description", "N/A"),
    "status": "confirmed",
}


# ── Mock scheduling agent ─────────────────────────────────────────────────────

class MockSchedulingAgent(SchedulingAgent):
    """SchedulingAgent with the Calendly slot-fetch replaced by hardcoded stubs."""

    def _present_slots(
        self,
        utterance: str,
        internal_state: dict,
        config: dict,
    ) -> SubagentResponse:
        internal_state["available_slots"] = [
            {
                "slot_id": s.slot_id,
                "description": s.description,
                "start_time": s.start.isoformat(),
                "end_time": s.end.isoformat(),
                "event_type_uri": s.event_type_uri,
            }
            for s in _FAKE_SLOTS
        ]
        internal_state["stage"] = "awaiting_choice"
        internal_state["retry_count"] = 0
        internal_state["matched_event_type_uri"] = None

        speak = self._speak_slots(_FAKE_SLOTS)
        return SubagentResponse(
            status=AgentStatus.IN_PROGRESS,
            speak=speak,
            internal_state=internal_state,
        )


# ── Registry builder ──────────────────────────────────────────────────────────

def _build_registry() -> dict:
    return {
        "farewell":             FarewellAgent(),
        "faq":                  FAQAgent(),
        "context_docs":         ContextDocsAgent(),
        "fallback":             FallbackAgent(),
        "data_collection":      DataCollectionAgent(),
        "narrative_collection": NarrativeCollectionAgent(),
        "intake_qualification": IntakeQualificationAgent(),
        "scheduling":           MockSchedulingAgent(),
    }


# ── Synchronous edge-chain follower ───────────────────────────────────────────

def _invoke_and_follow(
    agent_id: str,
    utterance: str,
    turn_idx: int,
    graph: WorkflowGraph,
    registry: dict,
    router: Router,
    config: dict,
    collected: dict,
    transcript_turns: list[dict],
    chain_depth: int = 0,
) -> tuple[str, Optional[str]]:
    """
    Invoke an agent and follow the resulting edge, mirroring handler.py's
    _invoke_and_follow — but synchronous and without tools or async transport.

    Returns (speak_text, finalize_reason | None).
    finalize_reason is set when an edge targets "end".
    """
    enriched_config = {
        **config,
        "_collected":       dict(collected),
        "_booking":         {},
        "_notes":           "",
        "_tool_results":    {},
        "_workflow_stages": graph.primary_goal_summary(),
    }
    agent = registry[agent_id]
    agent_state = graph.states[agent_id]

    recent_history = [
        {"role": t["role"], "content": t["content"]}
        for t in transcript_turns[-8:]
    ]

    response: SubagentResponse = agent.process(
        utterance,
        dict(agent_state.internal_state),
        enriched_config,
        recent_history,
    )

    graph.update(agent_id, response, turn_idx)

    # Merge collected fields
    if response.collected:
        collected.update(response.collected)
    if response.hidden_collected:
        for k, v in response.hidden_collected.items():
            if k not in collected:
                collected[k] = v

    # Handle UNHANDLED — re-route once
    if response.status == AgentStatus.UNHANDLED:
        reason = response.internal_state.get("cannot_process_reason", "unknown")
        logger.info("Agent %s UNHANDLED (reason=%s) — re-routing", agent_id, reason)
        if chain_depth < _MAX_CHAIN:
            re_agent_id, _ = router.select(utterance, recent_history, hint=f"unhandled:{reason}")
            if re_agent_id != agent_id:
                return _invoke_and_follow(
                    re_agent_id, utterance, turn_idx, graph, registry, router,
                    config, collected, transcript_turns, chain_depth + 1,
                )
        return "", None

    edge = graph.get_edge(agent_id, response.status)
    speak = response.speak or ""

    if edge.target == "decider":
        return speak, None
    if edge.target == "end":
        return speak, edge.reason or "completed"
    if chain_depth >= _MAX_CHAIN:
        logger.warning("Edge chain depth %d exceeded at %s — stopping", chain_depth, agent_id)
        return speak, None
    if edge.target == "resume":
        resume_id = router.pop_resume()
        if resume_id:
            resume_speak, fr = _invoke_and_follow(
                resume_id, "", turn_idx, graph, registry, router,
                config, collected, transcript_turns, chain_depth + 1,
            )
            return (speak + " " + resume_speak).strip(), fr
        return speak, None

    # Follow edge to the next agent (auto-chain: narrative → qualification → scheduling)
    chain_speak, finalize_reason = _invoke_and_follow(
        edge.target, "", turn_idx, graph, registry, router,
        config, collected, transcript_turns, chain_depth + 1,
    )
    return (speak + " " + chain_speak).strip(), finalize_reason


# ── Public API ────────────────────────────────────────────────────────────────

def replay_test_case(test_case: TestCase, config: dict) -> ReplayResult:
    """
    Replay all caller turns from a TestCase through the real agent pipeline.

    Returns a ReplayResult with one ReplayTurn per caller utterance that was
    actually processed (up to _MAX_TURNS).  No calendar API calls are made and
    no webhooks are fired.
    """
    graph     = WorkflowGraph(APPOINTMENT_BOOKING)
    router    = Router(graph)
    registry  = _build_registry()
    collected: dict       = {}
    transcript_turns: list[dict] = []
    replay_turns: list[ReplayTurn] = []
    finalize_reason: Optional[str] = None
    error_msg: Optional[str] = None

    try:
        # Prime the pipeline: set data_collection IN_PROGRESS and get opening question.
        graph.states["data_collection"].status = AgentStatus.IN_PROGRESS
        opening_speak, _ = _invoke_and_follow(
            "data_collection", "", 0, graph, registry, router,
            config, collected, transcript_turns,
        )
        if opening_speak:
            transcript_turns.append({"role": "assistant", "content": opening_speak})

        # Feed each caller turn through the pipeline.
        for i, caller_turn in enumerate(test_case.caller_turns):
            if finalize_reason is not None or i >= _MAX_TURNS:
                break

            utterance = caller_turn.text
            transcript_turns.append({"role": "user", "content": utterance})
            turn_idx = i + 1

            recent_history = [
                {"role": t["role"], "content": t["content"]}
                for t in transcript_turns[-8:]
            ]

            try:
                # Check WAITING_CONFIRM enforcement (same as router.select does internally)
                agent_id, interrupt = router.select(utterance, recent_history)

                speak_text, fr = _invoke_and_follow(
                    agent_id, utterance, turn_idx, graph, registry, router,
                    config, collected, transcript_turns,
                )

                if not speak_text or not speak_text.strip():
                    speak_text = "I'm sorry, I didn't quite get that. Could you please say that again?"

                transcript_turns.append({"role": "assistant", "content": speak_text})

                replay_turns.append(ReplayTurn(
                    caller_utterance=utterance,
                    ai_response=speak_text,
                    agent_id=agent_id,
                    agent_status=graph.states[agent_id].status.value,
                    turn_index=caller_turn.turn_index,
                ))

                if fr is not None:
                    finalize_reason = fr

            except Exception as exc:
                logger.error(
                    "Replay error at turn %d for %s: %s",
                    turn_idx, test_case.conversation_id, exc, exc_info=True,
                )
                # Record partial turn and continue
                replay_turns.append(ReplayTurn(
                    caller_utterance=utterance,
                    ai_response="[ERROR]",
                    agent_id="unknown",
                    agent_status="error",
                    turn_index=caller_turn.turn_index,
                ))

    except Exception as exc:
        logger.error(
            "Replay fatal error for %s: %s",
            test_case.conversation_id, exc, exc_info=True,
        )
        error_msg = str(exc)

    logger.info(
        "Replay complete | conv=%s | caller_turns=%d | replayed=%d | end_reason=%s",
        test_case.conversation_id,
        len(test_case.caller_turns),
        len(replay_turns),
        finalize_reason or "incomplete",
    )

    return ReplayResult(
        conversation_id=test_case.conversation_id,
        test_case=test_case,
        replay_turns=replay_turns,
        error=error_msg,
    )
