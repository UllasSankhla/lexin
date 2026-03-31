"""Main WebSocket connection handler — supports voice and text modes.

Orchestration flow (both modes)
================================
1.  Transport starts (voice: STT session opens; text: no-op).
2.  Greeting and opening data_collection invocation → response delivered via transport.
3.  Two concurrent tasks run:
      _ws_read_loop       — feeds WebSocket frames to the transport
      _utterance_processor — consumes transport.utterance_queue one at a time
4.  _process_utterance routes each utterance through the agent system via Router.
5.  Agent response carries:
      .speak          → delivered via transport (voice: TTS stream; text: JSON message)
      .collected      → DB persist + client event
      .booking        → DB persist + client event
"""
from __future__ import annotations

import asyncio
import itertools
import json
import logging
import time
from datetime import datetime, timezone
from typing import Literal

from fastapi import WebSocket, WebSocketDisconnect

from app.config import settings
from app.agents.graph_config import APPOINTMENT_BOOKING
from app.agents.workflow import WorkflowGraph
from app.agents.planner import Planner
from app.agents.registry import build_registry
from app.agents.base import AgentStatus
from app.services.config_client import get_cached_config
from app.services.transcript_store import save_transcript
from app.tools.base import ToolTrigger, ToolContext
from app.tools.registry import resolve_tool
from app.websocket.session import CallSession
from app.agents.empathy_filter import apply_empathy_filter
from app.websocket.state_machine import CallState
from app.transport.base import BaseTransport
from app.transport.voice_transport import VoiceTransport
from app.transport.text_transport import TextTransport

logger = logging.getLogger(__name__)


async def handle_call(
    ws: WebSocket,
    session: CallSession,
    db_session,
    mode: Literal["voice", "text"] = "voice",
) -> None:
    """Orchestrate a call session over a WebSocket connection."""
    from app.models.call_analytics import CallAnalytics
    from app.models.call_record import CallRecord

    seq = [0]
    config = session.config
    assistant_cfg = config.get("assistant", {})

    # ── Pipeline initialisation ───────────────────────────────────────────────
    graph = WorkflowGraph(APPOINTMENT_BOOKING)
    session.graph = graph
    turn_counter = [0]
    collected_all: dict[str, str] = {}
    booking_result: dict = {}
    notes_buffer: str = ""
    transcript_turns: list[dict] = []

    planner = Planner(graph)
    registry = build_registry(session.call_id, transcript_turns)
    utterance_lock = asyncio.Lock()
    consecutive_errors = [0]

    # Shared mutable store across all tool invocations for this call.
    tool_shared: dict = {}
    pending_tool_tasks: dict[str, tuple[asyncio.Task, str | None]] = {}

    # ── Send helpers (priority queue: 0=audio, 1=text, 2=sentinel) ───────────
    _send_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
    _send_counter = itertools.count()

    async def _send_loop() -> None:
        while True:
            _priority, _, item = await _send_queue.get()
            if item is None:
                _send_queue.task_done()
                break
            kind, data = item
            try:
                if kind == "text":
                    await ws.send_text(data)
                else:
                    await ws.send_bytes(data)
            except Exception as exc:
                logger.warning("Send loop error: %s", exc)
            _send_queue.task_done()

    async def safe_send_text(msg_type: str, payload: dict) -> None:
        seq[0] += 1
        frame = json.dumps({
            "type": msg_type, "seq": seq[0], "ts": time.time(), "payload": payload,
        })
        logger.debug("→ [%d] %s %s", seq[0], msg_type, payload)
        await _send_queue.put((1, next(_send_counter), ("text", frame)))

    async def safe_send_binary(data: bytes) -> None:
        logger.debug("→ [binary] %d bytes (audio)", len(data))
        await _send_queue.put((0, next(_send_counter), ("binary", data)))

    # ── Transport ─────────────────────────────────────────────────────────────
    if mode == "text":
        transport: BaseTransport = TextTransport(safe_send_text)
    else:
        transport = VoiceTransport(safe_send_text, safe_send_binary, session, assistant_cfg)

    # ── Core: process a caller utterance through the agent system ─────────────

    async def _persist_response(response) -> None:
        """Persist collected fields, notes, and booking from an agent response."""
        from app.models.gathered_parameter import GatheredParameter
        if response.collected:
            for field_name, field_value in response.collected.items():
                if field_name not in collected_all:
                    collected_all[field_name] = field_value
                    remaining = sum(
                        1 for p in config.get("parameters", [])
                        if p["name"] not in collected_all
                    )
                    await safe_send_text("server.parameter_collected", {
                        "parameter_name": field_name,
                        "value": field_value,
                        "remaining_count": remaining,
                    })
                    db_session.add(GatheredParameter(
                        call_id=session.call_id,
                        parameter_name=field_name,
                        raw_value=field_value,
                        normalized_value=field_value,
                        validated=True,
                    ))
            db_session.commit()
        if response.hidden_collected:
            for field_name, field_value in response.hidden_collected.items():
                if field_name not in collected_all:
                    collected_all[field_name] = field_value
                    logger.info(
                        "hidden_collected | field=%s | value=%r",
                        field_name, str(field_value)[:120],
                    )
        if response.notes:
            nonlocal notes_buffer
            notes_buffer = response.notes
            session.notes_buffer = notes_buffer
        if response.booking:
            nonlocal booking_result
            booking_result = response.booking
            await safe_send_text("server.booking_confirmed", {
                "booking_id": response.booking.get("booking_id"),
                "slot_description": response.booking.get("slot_description", ""),
            })
            from app.models.call_record import CallRecord
            record = db_session.get(CallRecord, session.call_id)
            if record:
                record.booking_id = response.booking.get("booking_id")
                record.booking_status = response.booking.get("status")
                db_session.commit()

    def _make_tool_ctx(
        transcript_path: str | None = None,
        duration_sec: float | None = None,
        booking_override: dict | None = None,
    ) -> ToolContext:
        return ToolContext(
            call_id=session.call_id,
            config=config,
            collected=dict(collected_all),
            transcript_lines=list(session.transcript_lines),
            booking_result=booking_override if booking_override is not None else booking_result,
            duration_sec=duration_sec if duration_sec is not None else session.duration_sec(),
            transcript_path=transcript_path,
            shared=tool_shared,
        )

    async def _invoke_tools(trigger: ToolTrigger, agent_id: str | None = None) -> None:
        if trigger == ToolTrigger.CALL_END:
            return
        bindings = [
            b for b in graph.workflow.tools
            if b.trigger == trigger
            and (b.agent_id is None or b.agent_id == agent_id)
        ]
        if not bindings:
            return
        for binding in bindings:
            tool = resolve_tool(binding.tool_class)
            ctx = _make_tool_ctx()
            task = asyncio.create_task(tool.run(ctx))
            if not binding.fire_and_forget:
                pending_tool_tasks[binding.tool_class] = (task, binding.await_before_agent)
                logger.debug(
                    "Tool %s started (pending, await_before=%s)",
                    binding.tool_class, binding.await_before_agent or "any",
                )
            else:
                logger.debug("Tool %s started (fire-and-forget)", binding.tool_class)

    async def _await_pending_tools(current_agent_id: str) -> None:
        for name, (task, await_before) in list(pending_tool_tasks.items()):
            if await_before is not None and await_before != current_agent_id:
                continue
            try:
                result = await task
                logger.info(
                    "Tool %s completed — success=%s data=%s",
                    name, result.success, result.data,
                )
            except Exception as exc:
                logger.warning("Tool %s raised an exception: %s", name, exc)
            del pending_tool_tasks[name]

    async def _invoke_and_follow(
        agent_id: str,
        utterance: str,
        current_turn: int,
        loop,
        call_history: list[dict],
        chain_depth: int = 0,
    ) -> tuple[str, str | None, float]:
        MAX_CHAIN = 5
        await _await_pending_tools(agent_id)

        enriched_config = {
            **config,
            "_collected":       dict(collected_all),
            "_booking":         booking_result,
            "_notes":           notes_buffer,
            "_tool_results":    tool_shared,
            "_workflow_stages": graph.primary_goal_summary(),
        }

        agent = registry[agent_id]
        agent_state = graph.states[agent_id]

        response = await loop.run_in_executor(
            None,
            lambda: agent.process(
                utterance,
                dict(agent_state.internal_state),
                enriched_config,
                call_history,
            )
        )

        graph.update(agent_id, response, current_turn)

        # ── UNHANDLED: re-route this utterance to the appropriate agent ────────
        # data_collection (or any agent) may return UNHANDLED when it cannot
        # interpret the utterance (off-topic question, wants to cancel, noise).
        # Rather than returning an empty speak or a generic recovery phrase,
        # we re-ask the router so the correct interrupt agent (faq, fallback,
        # etc.) can handle it. The interrupted agent's state is preserved in the
        # graph and will be resumed via the resume stack when that agent finishes.
        if response.status == AgentStatus.UNHANDLED:
            reason = response.internal_state.get("cannot_process_reason", "unknown")
            logger.info(
                "Agent %s returned UNHANDLED (reason=%s) — re-routing utterance %r",
                agent_id, reason, utterance[:60],
            )
            if chain_depth >= MAX_CHAIN:
                logger.warning(
                    "UNHANDLED re-route skipped — MAX_CHAIN (%d) reached at %s",
                    MAX_CHAIN, agent_id,
                )
                return "", None, 0.0

            re_steps = await loop.run_in_executor(
                None,
                lambda: planner.plan(utterance, call_history),
            )
            re_agent_id = next(
                (s.agent_id for s in re_steps if s.action == "invoke" and s.agent_id != agent_id),
                None,
            )

            if not re_agent_id or re_agent_id == agent_id:
                # Planner could not find a different agent — avoid infinite loop
                logger.warning(
                    "Planner found no alternative for UNHANDLED %s — returning empty speak",
                    agent_id,
                )
                return "", None, 0.0

            re_speak, finalize_reason, re_conf = await _invoke_and_follow(
                re_agent_id, utterance, current_turn, loop, call_history, chain_depth + 1
            )
            # If the interrupted agent was data_collection with a pending
            # confirmation, re-invoke it with an empty utterance so it re-surfaces
            # the pending question and the graph is back in WAITING_CONFIRM.
            # Without this the caller receives only the FAQ answer and "Yes" on
            # the following turn is misread as a response to the FAQ.
            if not finalize_reason and agent_id == "data_collection":
                dc_state = graph.states.get("data_collection")
                if dc_state and dc_state.internal_state.get("pending_confirmation"):
                    dc_speak, dc_fr, dc_conf = await _invoke_and_follow(
                        "data_collection", "", current_turn, loop, call_history, chain_depth + 1
                    )
                    combined = (re_speak + " " + dc_speak).strip()
                    return combined, dc_fr, dc_conf
            return re_speak, finalize_reason, re_conf
        # ── End UNHANDLED handling ─────────────────────────────────────────────

        edge = graph.get_edge(agent_id, response.status)

        logger.info(
            "Agent response | agent=%s | status=%s | speak=%r | edge=%s→%s | collected=%s",
            agent_id,
            response.status.value,
            (response.speak or "")[:120],
            response.status.value, edge.target,
            list(response.collected.keys()) if response.collected else [],
        )

        await _persist_response(response)

        if response.status == AgentStatus.COMPLETED:
            await _invoke_tools(ToolTrigger.AGENT_COMPLETE, agent_id)

        speak = response.speak or ""

        if edge.target == "decider":
            return speak, None, response.confidence
        if edge.target == "end":
            return speak, edge.reason or "completed", response.confidence
        if chain_depth >= MAX_CHAIN:
            logger.warning(
                "Edge chain depth %d exceeded at agent=%s — stopping, target=%s",
                chain_depth, agent_id, edge.target,
            )
            return speak, None, response.confidence
        if edge.target == "resume":
            resume_id = planner.pop_resume()
            if resume_id:
                resume_speak, finalize_reason, resume_conf = await _invoke_and_follow(
                    resume_id, "", current_turn, loop, call_history, chain_depth + 1
                )
                return (speak + " " + resume_speak).strip(), finalize_reason, resume_conf
            return speak, None, response.confidence

        chain_speak, finalize_reason, chain_conf = await _invoke_and_follow(
            edge.target, "", current_turn, loop, call_history, chain_depth + 1
        )
        return (speak + " " + chain_speak).strip(), finalize_reason, chain_conf

    async def _invoke_parallel_both(
        utterance: str,
        current_turn: int,
        loop,
        call_history: list[dict],
    ) -> tuple[str, str | None]:
        """
        Fix 5: Parallel invocation for BOTH utterances (field data + legal narrative).
        Runs data_collection and narrative_collection concurrently; selects the best
        response using confidence scores.
        """
        available_ids = {n.id for n in graph.available_nodes()}
        tasks = []
        agent_ids_parallel = []

        if "data_collection" in available_ids:
            tasks.append(_invoke_and_follow(
                "data_collection", utterance, current_turn, loop, call_history
            ))
            agent_ids_parallel.append("data_collection")

        if "narrative_collection" in available_ids:
            tasks.append(_invoke_and_follow(
                "narrative_collection", utterance, current_turn, loop, call_history
            ))
            agent_ids_parallel.append("narrative_collection")

        if not tasks:
            return "", None
        if len(tasks) == 1:
            speak, fr, _ = await tasks[0]
            return speak, fr

        results = await asyncio.gather(*tasks, return_exceptions=True)

        valid: list[tuple[str, str | None, float, str]] = []
        for i, res in enumerate(results):
            if isinstance(res, Exception):
                logger.warning("Parallel invocation of %s raised: %s", agent_ids_parallel[i], res)
                continue
            speak, fr, conf = res
            valid.append((speak, fr, conf, agent_ids_parallel[i]))

        if not valid:
            return "", None

        # If any agent triggered finalize, propagate it
        for speak, fr, conf, aid in valid:
            if fr is not None:
                return speak, fr

        # WAITING_CONFIRM always wins
        for speak, fr, conf, aid in valid:
            ag_state = graph.states.get(aid)
            if ag_state and ag_state.status == AgentStatus.WAITING_CONFIRM:
                return speak, fr

        # Build confidence-aware speaks list and use combine_speaks
        speaks_for_combine = [
            (speak, aid, conf)
            for speak, fr, conf, aid in valid
            if speak.strip()
        ]
        if not speaks_for_combine:
            return "", None
        if len(speaks_for_combine) == 1:
            return speaks_for_combine[0][0], None

        combined = planner.combine_speaks(speaks_for_combine)
        logger.info(
            "Parallel BOTH invocation: agents=%s confs=%s",
            [s[1] for s in speaks_for_combine],
            [f"{s[2]:.2f}" for s in speaks_for_combine],
        )
        return combined, None

    async def _quick_faq_query(utterance: str, history: list[dict], loop) -> str | None:
        """
        Query the FAQ agent directly without modifying graph state.

        Used in parallel with narrative_collection when the caller embeds a
        question inside their narrative opening. Returns the FAQ speak text if
        a match is found, otherwise None. Failures are swallowed — this is a
        best-effort enrichment, not a required path.
        """
        faq_agent = registry.get("faq")
        if faq_agent is None:
            return None
        try:
            enriched = {
                **config,
                "_collected":       dict(collected_all),
                "_booking":         booking_result,
                "_notes":           notes_buffer,
                "_tool_results":    tool_shared,
                "_workflow_stages": graph.primary_goal_summary(),
            }
            response = await loop.run_in_executor(
                None,
                lambda: faq_agent.process(utterance, {}, enriched, history),
            )
            if response.status == AgentStatus.COMPLETED and response.speak:
                return response.speak
        except Exception as exc:
            logger.warning("Parallel FAQ query failed: %s", exc)
        return None


    async def _process_utterance(caller_text: str) -> None:
        nonlocal collected_all, booking_result, notes_buffer
        if session.state_machine.is_terminal():
            return

        session.add_user_turn(caller_text)
        transcript_turns.append({"role": "user", "content": caller_text})
        await safe_send_text("server.thinking", {})
        turn_counter[0] += 1
        current_turn = turn_counter[0]
        logger.info("CALLER | turn=%d | text=%r", current_turn, caller_text)

        call_history = [
            {"role": t["role"], "content": t["content"]}
            for t in transcript_turns
        ]

        try:
            loop = asyncio.get_running_loop()

            # ── Planner: generate execution plan ─────────────────────────────
            steps = await loop.run_in_executor(
                None, lambda: planner.plan(caller_text, call_history)
            )
            # ── Execute each step in the plan ─────────────────────────────────
            # All invoke steps run concurrently via asyncio.gather. Results are
            # processed in speech order (planner intent order) for speak assembly.
            speaks: list[tuple[str, str, float]] = []  # (speak_text, agent_id, confidence)
            finalize_reason = None
            agent_id = "unknown"
            waiting_confirm_speak: str | None = None

            # Separate reset_fields (applied immediately) from invoke steps
            invoke_steps = []
            for step in steps:
                if step.action == "reset_fields" and step.fields:
                    planner.reset_fields(step.fields, collected_all)
                elif step.action == "invoke" and step.agent_id:
                    invoke_steps.append(step)

            if invoke_steps:
                planned_agent_ids = {s.agent_id for s in invoke_steps}

                # Push the active primary onto the resume stack once if any
                # planned step is interrupt-eligible and the primary is not
                # already being handled by an explicit step in this plan.
                primary_id = planner.active_primary_for_resume()
                if primary_id and primary_id not in planned_agent_ids:
                    if any(
                        graph.nodes.get(s.agent_id) and graph.nodes[s.agent_id].interrupt_eligible
                        for s in invoke_steps
                    ):
                        planner.push_resume(primary_id)

                # Run all invoke steps concurrently. Each agent writes only to
                # its own graph node, so there are no shared-state conflicts.
                results = await asyncio.gather(
                    *[
                        _invoke_and_follow(
                            s.agent_id,
                            "" if s.use_empty_utterance else caller_text,
                            current_turn, loop, call_history,
                        )
                        for s in invoke_steps
                    ],
                    return_exceptions=True,
                )
                logger.info(
                    "Steps executed%s: %s",
                    " (parallel)" if len(invoke_steps) > 1 else "",
                    [s.agent_id for s in invoke_steps],
                )

                # Process results in speech order (planner intent order).
                # WAITING_CONFIRM always wins: prepend any prior speaks and stop.
                for idx, res in enumerate(results):
                    agent_id = invoke_steps[idx].agent_id
                    if isinstance(res, Exception):
                        logger.warning("Step %s raised: %s", agent_id, res)
                        continue
                    s_speak, s_fr, s_conf = res
                    if s_fr is not None and finalize_reason is None:
                        finalize_reason = s_fr
                    ag_state = graph.states.get(agent_id)
                    if ag_state and ag_state.status == AgentStatus.WAITING_CONFIRM:
                        prior = planner.combine_speaks(speaks)
                        waiting_confirm_speak = (prior + " " + s_speak).strip() if prior else s_speak
                        break
                    if s_speak:
                        speaks.append((s_speak, agent_id, s_conf))

            consecutive_errors[0] = 0

            if waiting_confirm_speak is not None:
                speak_text = waiting_confirm_speak
            else:
                speak_text = planner.combine_speaks(speaks)

            if not speak_text or not speak_text.strip():
                logger.warning("speak_text empty after full chain — using recovery phrase")
                speak_text = "I'm sorry, I didn't quite get that. Could you please say that again?"

            agents_ran = frozenset(s.agent_id for s in invoke_steps)
            speak_text = apply_empathy_filter(speak_text, collected_all, transcript_turns, agents_ran)

            session.add_assistant_turn(speak_text)
            transcript_turns.append({"role": "assistant", "content": speak_text})

            audio_id = f"resp-{seq[0] + 1}"
            logger.info(
                "REPLY | turn=%d | agents=%s | words=%d | audio_id=%s | text=%r",
                current_turn,
                [s.agent_id for s in invoke_steps] if invoke_steps else [],
                len(speak_text.split()),
                audio_id,
                speak_text,
            )
            await safe_send_text("server.response_text", {"text": speak_text, "audio_id": audio_id})
            await transport.send_response(speak_text, audio_id)

            if finalize_reason is not None:
                await _finalize_call(finalize_reason)

        except Exception as exc:
            consecutive_errors[0] += 1
            logger.error(
                "Agent processing error (consecutive=%d): %s",
                consecutive_errors[0], exc, exc_info=True,
            )
            error_policy = graph.workflow.error_policy
            if consecutive_errors[0] >= error_policy.max_consecutive_errors:
                logger.error(
                    "Call %s: %d consecutive errors — ending call",
                    session.call_id, error_policy.max_consecutive_errors,
                )
                try:
                    await transport.send_response(
                        "I'm sorry, I'm having technical difficulties. "
                        "Someone will follow up with you shortly. Goodbye!",
                        f"err-{seq[0] + 1}",
                    )
                except Exception:
                    pass
                await _finalize_call(error_policy.on_max_errors.reason or "agent_error")
            else:
                try:
                    await safe_send_text("server.error", {
                        "code": "agent_error", "message": "Processing failed", "fatal": False,
                    })
                    recovery = error_policy.transient_error_speak
                    await safe_send_text("server.response_text", {
                        "text": recovery, "audio_id": f"err-{seq[0] + 1}",
                    })
                    await transport.send_response(recovery, f"err-{seq[0] + 1}")
                except Exception:
                    pass

    # ── Completion ────────────────────────────────────────────────────────────

    async def _run_call_end_tools(ctx: ToolContext) -> None:
        bindings = [b for b in graph.workflow.tools if b.trigger == ToolTrigger.CALL_END]
        for binding in bindings:
            tool = resolve_tool(binding.tool_class)
            try:
                result = await tool.run(ctx)
                logger.info(
                    "CALL_END tool %s: success=%s data=%s",
                    binding.tool_class, result.success, result.data,
                )
            except Exception as exc:
                logger.warning("CALL_END tool %s raised: %s", binding.tool_class, exc)

    async def _finalize_call(reason: str, booking_details: dict | None = None) -> None:
        duration = session.duration_sec()
        session.state_machine.transition(CallState.DONE)

        logger.info(
            "Call %s ending — reason: %s | duration: %.1fs",
            session.call_id, reason, duration,
        )

        transcript_path = await save_transcript(session)

        record = db_session.get(CallRecord, session.call_id)
        if record:
            record.state           = "done"
            record.completed_at    = datetime.now(timezone.utc)
            record.duration_sec    = duration
            record.transcript_path = transcript_path
            record.end_reason      = reason
            db_session.commit()

        for event in session.analytics_events:
            db_session.add(CallAnalytics(call_id=session.call_id, **event))
        db_session.commit()

        await safe_send_text("server.call_ended", {
            "call_id": session.call_id,
            "reason": reason,
            "duration_sec": round(duration, 2),
        })

        call_end_ctx = _make_tool_ctx(
            transcript_path=transcript_path,
            duration_sec=duration,
            booking_override=booking_details if booking_details else (booking_result if booking_result else None),
        )
        asyncio.create_task(_run_call_end_tools(call_end_ctx))

        logger.info(
            "Call %s finalized — reason: %s | duration: %.1fs",
            session.call_id, reason, duration,
        )

    # ── Main WebSocket loop ───────────────────────────────────────────────────

    send_loop_task = asyncio.create_task(_send_loop())
    watchdog_task: asyncio.Task | None = None
    try:
        await transport.start()

        session.state_machine.transition(CallState.ACTIVE)
        record = db_session.get(CallRecord, session.call_id)
        if record:
            record.state        = "active"
            record.connected_at = datetime.now(timezone.utc)
            db_session.commit()

        # Greeting
        greeting_text = assistant_cfg.get("greeting_message", "Hello! How can I help you today?")
        session.add_assistant_turn(greeting_text)
        await safe_send_text("server.session_ready", {
            "call_id":      session.call_id,
            "persona_name": assistant_cfg.get("persona_name", "Assistant"),
        })
        await safe_send_text("server.response_text", {"text": greeting_text, "audio_id": "greet-001"})
        await transport.send_response(greeting_text, "greet-001")

        # Opening — invoke data_collection with empty utterance to get first question
        loop = asyncio.get_running_loop()
        graph.states["data_collection"].status = AgentStatus.IN_PROGRESS
        opening_speak, _, _opening_conf = await _invoke_and_follow(
            "data_collection", "", 0, loop, [], chain_depth=0
        )
        opening_text = opening_speak or assistant_cfg.get("greeting_message", "Hello! How can I help you today?")
        session.add_assistant_turn(opening_text)
        transcript_turns.append({"role": "assistant", "content": opening_text})
        await safe_send_text("server.response_text", {"text": opening_text, "audio_id": "open-001"})
        await transport.send_response(opening_text, "open-001")

        max_duration = assistant_cfg.get("max_call_duration_sec", settings.max_call_duration_sec)

        async def watchdog() -> None:
            await asyncio.sleep(max_duration)
            if not session.state_machine.is_terminal():
                logger.warning(
                    "Call %s terminated — reason: max_duration_exceeded (%ds limit reached)",
                    session.call_id, max_duration,
                )
                await safe_send_text("server.error", {
                    "code": "max_duration_exceeded",
                    "message": f"Call time limit of {max_duration}s reached",
                    "fatal": True,
                })
                await _finalize_call("max_duration_exceeded")

        watchdog_task = asyncio.create_task(watchdog())

        # ── Concurrent read loop + utterance processor ────────────────────────

        async def _ws_read_loop() -> str:
            """Feed WebSocket frames to the transport. Returns the exit reason."""
            while not session.state_machine.is_terminal():
                try:
                    message = await asyncio.wait_for(ws.receive(), timeout=max_duration)
                except asyncio.TimeoutError:
                    logger.warning(
                        "Call %s terminated — reason: receive_loop_timeout "
                        "(no WebSocket frame for %ds)",
                        session.call_id, max_duration,
                    )
                    return "timeout"
                except WebSocketDisconnect:
                    logger.info(
                        "Call %s terminated — reason: websocket_disconnect at %.1fs",
                        session.call_id, session.duration_sec(),
                    )
                    return "disconnect"

                if "bytes" in message and message["bytes"]:
                    logger.debug("← [binary] %d bytes (audio)", len(message["bytes"]))
                    await transport.handle_binary_frame(message["bytes"])
                elif "text" in message and message["text"]:
                    try:
                        msg = json.loads(message["text"])
                        logger.debug("← %s", msg.get("type", ""))
                        if await transport.handle_text_frame(msg):
                            logger.info(
                                "Call %s terminated — reason: client_hangup at %.1fs",
                                session.call_id, session.duration_sec(),
                            )
                            return "hangup"
                    except json.JSONDecodeError:
                        logger.warning("← unparseable text frame: %.80s", message["text"])
            return "terminal"

        async def _utterance_processor() -> None:
            """Consume utterances from the transport queue one at a time."""
            while not session.state_machine.is_terminal():
                try:
                    utterance = await asyncio.wait_for(
                        transport.utterance_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                async with utterance_lock:
                    await _process_utterance(utterance)

        ws_task = asyncio.create_task(_ws_read_loop())
        proc_task = asyncio.create_task(_utterance_processor())

        done, pending = await asyncio.wait(
            [ws_task, proc_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

        # Propagate any unhandled exception from a completed task
        for task in done:
            exc = task.exception()
            if exc:
                raise exc

    except WebSocketDisconnect:
        logger.info(
            "Call %s terminated — reason: websocket_disconnect at %.1fs",
            session.call_id, session.duration_sec(),
        )
    except Exception as exc:
        logger.error(
            "Call %s terminated — reason: unhandled_exception (%s) at %.1fs",
            session.call_id, exc, session.duration_sec(), exc_info=True,
        )
        record = db_session.get(CallRecord, session.call_id)
        if record:
            record.state         = "error"
            record.error_message = str(exc)
            db_session.commit()
    finally:
        if watchdog_task:
            watchdog_task.cancel()
        await transport.close()
        if not session.state_machine.is_terminal():
            logger.warning(
                "Call %s terminated — reason: abnormal_exit at %.1fs",
                session.call_id, session.duration_sec(),
            )
            await _finalize_call("abnormal_exit")
        await _send_queue.put((2, next(_send_counter), None))
        try:
            await asyncio.wait_for(send_loop_task, timeout=5.0)
        except (asyncio.TimeoutError, Exception):
            send_loop_task.cancel()
