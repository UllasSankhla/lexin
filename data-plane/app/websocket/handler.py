"""Main WebSocket connection handler for voice calls.

Orchestration flow
==================
1.  STT session opens (Deepgram nova-2-phonecall); greeting is spoken.
2.  data_collection agent is invoked with empty utterance → first field question spoken.
3.  Read loop: binary frames → STT; text frames → control messages.
4.  STT on_final fires → _process_utterance(text).
5.  _process_utterance routes to the appropriate agent via Router.
6.  Agent response carries:
      .speak          → TTS + transcript
      .collected      → DB persist + client event
      .booking        → DB persist + client event
"""
from __future__ import annotations

import asyncio
import itertools
import json
import logging
import struct
import time
from datetime import datetime, timezone

from fastapi import WebSocket, WebSocketDisconnect

from app.config import settings
from app.agents.graph_config import APPOINTMENT_BOOKING
from app.agents.workflow import WorkflowGraph
from app.agents.router import Router
from app.agents.registry import build_registry
from app.agents.base import AgentStatus
from app.pipeline.stt import STTSession
from app.pipeline.tts import TTSClient
from app.services.config_client import get_cached_config
from app.services.summary_generator import generate_call_summary
from app.services.transcript_store import save_transcript
from app.services.webhook_dispatcher import dispatch_webhooks
from app.websocket.session import CallSession
from app.websocket.state_machine import CallState

logger = logging.getLogger(__name__)


# ── Frame helpers ─────────────────────────────────────────────────────────────

def _encode_audio_frame(audio_id: str, audio_bytes: bytes) -> bytes:
    id_bytes = audio_id.encode("utf-8")
    return struct.pack("<I", len(id_bytes)) + id_bytes + audio_bytes


async def _send_json(ws: WebSocket, msg_type: str, payload: dict, seq: list) -> None:
    seq[0] += 1
    logger.debug("→ [%d] %s %s", seq[0], msg_type, payload)
    await ws.send_text(json.dumps({
        "type": msg_type,
        "seq": seq[0],
        "ts": time.time(),
        "payload": payload,
    }))


# ── Main handler ──────────────────────────────────────────────────────────────

async def handle_call(ws: WebSocket, session: CallSession, db_session) -> None:
    """Orchestrate a voice call session over a WebSocket connection."""
    from app.models.call_analytics import CallAnalytics
    from app.models.call_record import CallRecord

    seq = [0]
    config = session.config
    assistant_cfg = config.get("assistant", {})

    # ── Pipeline initialisation ───────────────────────────────────────────────
    graph = WorkflowGraph(APPOINTMENT_BOOKING)
    session.graph = graph
    turn_counter = [0]
    collected_all: dict[str, str] = {}   # accumulates confirmed fields across turns
    booking_result: dict = {}
    notes_buffer: str = ""
    transcript_turns: list[dict] = []    # for webhook agent

    router = Router(graph)
    registry = build_registry(session.call_id, transcript_turns)

    tts = TTSClient(voice=assistant_cfg.get("persona_voice"))

    utterance_lock = asyncio.Lock()
    consecutive_errors = [0]   # reset on any successful turn; triggers end-call after threshold

    # TTS streaming state — used for barge-in (interruption) detection.
    tts_playing = False
    tts_cancel = asyncio.Event()

    # ── Send helpers (priority queue: 0=audio, 1=text, 2=sentinel) ───────────
    # Single send loop serialises all WebSocket writes; audio frames are
    # prioritised over text control messages so playback stays smooth.

    _send_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
    _send_counter = itertools.count()

    async def _send_loop() -> None:
        while True:
            _priority, _, item = await _send_queue.get()
            if item is None:  # sentinel — drain complete, stop loop
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

    # Merge small TTS PCM chunks into ~8 KB frames before sending to reduce
    # the number of WebSocket frames and smooth out audio delivery.
    _TTS_MERGE_BYTES = 8192

    async def _send_tts(text: str, audio_id: str) -> None:
        nonlocal tts_playing
        if not text or not text.strip():
            logger.warning("_send_tts called with empty text — skipping (audio_id=%s)", audio_id)
            return
        tts_cancel.clear()
        tts_playing = True
        t0 = time.monotonic()
        chunks_sent = 0
        try:
            await safe_send_text("server.tts_stream_start", {"audio_id": audio_id})
            pending = bytearray()
            async for chunk, first_chunk_latency in tts.stream(text):
                if tts_cancel.is_set():
                    logger.info(
                        "TTS stream interrupted (barge-in) after %d chunks — audio_id=%s",
                        chunks_sent, audio_id,
                    )
                    await safe_send_text("server.tts_interrupted", {"audio_id": audio_id})
                    return
                if first_chunk_latency is not None:
                    session.record_analytics("tts_first_chunk", "tts", first_chunk_latency)
                pending.extend(chunk)
                if len(pending) >= _TTS_MERGE_BYTES:
                    await safe_send_binary(_encode_audio_frame(audio_id, bytes(pending)))
                    chunks_sent += 1
                    pending = bytearray()
            # Flush remaining audio then signal stream end to the client
            if pending:
                await safe_send_binary(_encode_audio_frame(audio_id, bytes(pending)))
                chunks_sent += 1
            await safe_send_text("server.tts_stream_end", {"audio_id": audio_id})
            session.record_analytics("tts_stream_complete", "tts", (time.monotonic() - t0) * 1000)
        except Exception as exc:
            logger.error("TTS streaming error: %s", exc)
            await safe_send_text("server.error", {"code": "tts_error", "message": "Voice synthesis failed", "fatal": False})
        finally:
            tts_playing = False

    # ── STT callbacks ─────────────────────────────────────────────────────────

    async def on_interim(text: str, confidence: float) -> None:
        if tts_playing and not tts_cancel.is_set():
            logger.info(
                "Barge-in detected while TTS playing — cancelling stream | text=%r",
                text[:60],
            )
            tts_cancel.set()
        await safe_send_text("server.transcript_interim", {
            "text": text, "confidence": confidence, "is_final": False,
        })

    async def on_final(text: str, confidence: float, elapsed_ms: float) -> None:
        session.record_analytics("stt_utterance_final", "stt", elapsed_ms)
        await safe_send_text("server.transcript_final", {"text": text, "confidence": confidence})

        if utterance_lock.locked():
            if not tts_cancel.is_set():
                logger.debug(
                    "Dropping utterance %r — previous turn still processing, no barge-in",
                    text[:40],
                )
                return
            logger.info(
                "Barge-in utterance queued — waiting for utterance_lock | text=%r",
                text[:40],
            )
        async with utterance_lock:
            await _process_utterance(text)

    stt = STTSession(
        on_interim=on_interim,
        on_final=on_final,
        language=assistant_cfg.get("language", "en-US"),
        spell_rules=config.get("spell_rules", []),
    )

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

    async def _run_auto_agents(current_turn: int, loop) -> None:
        """Invoke auto_run agents whose dependencies are now satisfied."""
        for auto_node in graph.auto_run_ready():
            auto_agent = registry[auto_node.id]
            auto_state = graph.states[auto_node.id]
            enriched_config2 = {
                **config,
                "_collected": collected_all,
                "_booking": booking_result,
                "_notes": notes_buffer,
            }
            auto_response = await loop.run_in_executor(
                None,
                lambda: auto_agent.process(
                    "", dict(auto_state.internal_state), enriched_config2, []
                )
            )
            graph.update(auto_node.id, auto_response, current_turn)
            logger.info(
                "Agent response (auto-run) | agent=%s | status=%s",
                auto_node.id, auto_response.status.value,
            )
            dispatch_params = auto_response.internal_state.get("_webhook_dispatch")
            if dispatch_params:
                asyncio.create_task(dispatch_webhooks(
                    config=config,
                    call_id=dispatch_params["call_id"],
                    duration_sec=session.duration_sec(),
                    collected=dispatch_params["collected"],
                    transcript_path=None,
                    caller_name=dispatch_params.get("caller_name", ""),
                    ai_summary=dispatch_params.get("ai_summary", ""),
                    booking_details=dispatch_params.get("booking"),
                ))

    async def _invoke_and_follow(
        agent_id: str,
        utterance: str,
        current_turn: int,
        loop,
        recent_history: list[dict],
        chain_depth: int = 0,
    ) -> tuple[str, str | None]:
        """
        Invoke agent_id, persist its response, then follow the declared edge.
        Returns (accumulated_speak_text, finalize_reason_or_None).
        finalize_reason is non-None when edge target is "end".
        Chains synchronously within the same turn for edges like "scheduling",
        "resume", etc. Stops and returns when edge is "decider".
        Max chain depth = 5 to prevent cycles.
        """
        MAX_CHAIN = 5

        enriched_config = {
            **config,
            "_collected": collected_all,
            "_booking": booking_result,
            "_notes": notes_buffer,
        }

        agent = registry[agent_id]
        agent_state = graph.states[agent_id]

        response = await loop.run_in_executor(
            None,
            lambda: agent.process(
                utterance,
                dict(agent_state.internal_state),
                enriched_config,
                recent_history,
            )
        )

        graph.update(agent_id, response, current_turn)
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

        speak = response.speak or ""

        # ── Follow edge ───────────────────────────────────────────────────────

        if edge.target == "decider":
            return speak, None

        if edge.target == "end":
            return speak, edge.reason or "completed"

        if chain_depth >= MAX_CHAIN:
            logger.warning(
                "Edge chain depth %d exceeded at agent=%s — stopping, target=%s",
                chain_depth, agent_id, edge.target,
            )
            return speak, None

        if edge.target == "resume":
            resume_id = router.pop_resume()
            if resume_id:
                resume_speak, finalize_reason = await _invoke_and_follow(
                    resume_id, "", current_turn, loop, recent_history, chain_depth + 1
                )
                return (speak + " " + resume_speak).strip(), finalize_reason
            return speak, None

        # "<node_id>" — chain to another node immediately
        chain_speak, finalize_reason = await _invoke_and_follow(
            edge.target, "", current_turn, loop, recent_history, chain_depth + 1
        )
        return (speak + " " + chain_speak).strip(), finalize_reason

    async def _process_utterance(caller_text: str) -> None:
        nonlocal collected_all, booking_result, notes_buffer
        if session.state_machine.is_terminal():
            return

        session.add_user_turn(caller_text)
        transcript_turns.append({"role": "user", "content": caller_text})
        await safe_send_text("server.thinking", {})
        turn_counter[0] += 1
        current_turn = turn_counter[0]

        recent_history = [
            {"role": t["role"], "content": t["content"]}
            for t in transcript_turns[-8:]
        ]

        t0 = time.monotonic()
        try:
            loop = asyncio.get_event_loop()

            # ── Step 1: select agent (WAITING_CONFIRM enforced before LLM) ───
            agent_id, interrupt = await loop.run_in_executor(
                None, lambda: router.select(caller_text, recent_history)
            )

            # ── Step 2: invoke and follow edges until "decider" or "end" ─────
            speak_text, finalize_reason = await _invoke_and_follow(
                agent_id, caller_text, current_turn, loop, recent_history, chain_depth=0
            )

            session.record_analytics("workflow_turn", "llm", (time.monotonic() - t0) * 1000)
            consecutive_errors[0] = 0

            # ── Safety net: never speak empty string ──────────────────────────
            if not speak_text or not speak_text.strip():
                logger.warning("speak_text empty after full chain — using recovery phrase")
                speak_text = "I'm sorry, I didn't quite get that. Could you please say that again?"

            session.add_assistant_turn(speak_text)
            transcript_turns.append({"role": "assistant", "content": speak_text})

            audio_id = f"resp-{seq[0] + 1}"
            await safe_send_text("server.response_text", {"text": speak_text, "audio_id": audio_id})

            # ── Auto-run agents (webhook fires when scheduling COMPLETED) ─────
            await _run_auto_agents(current_turn, loop)

            # ── TTS ───────────────────────────────────────────────────────────
            await _send_tts(speak_text, audio_id)

            # ── Finalize if edge said "end" ───────────────────────────────────
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
                    "Call %s: %d consecutive errors — ending call cleanly",
                    session.call_id, error_policy.max_consecutive_errors,
                )
                try:
                    await _send_tts(
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
                    await _send_tts(recovery, f"err-{seq[0] + 1}")
                except Exception:
                    pass

    # ── Completion ────────────────────────────────────────────────────────────

    async def _handle_completion(booking_details: dict | None = None) -> None:
        session.state_machine.transition(CallState.COMPLETING)
        await safe_send_text("server.call_completing", {
            "collected_parameters": collected_all,
            "booking": booking_details,
        })
        await asyncio.sleep(1)
        await _finalize_call("completed", booking_details=booking_details)

    async def _finalize_call(reason: str, booking_details: dict | None = None) -> None:
        duration = session.duration_sec()
        session.state_machine.transition(CallState.DONE)

        logger.info(
            "Call %s ending — reason: %s | duration: %.1fs",
            session.call_id, reason, duration,
        )

        transcript_path = await save_transcript(session)

        caller_name, ai_summary = "Unknown Caller", ""
        try:
            loop = asyncio.get_event_loop()
            caller_name, ai_summary = await loop.run_in_executor(
                None,
                lambda: generate_call_summary(
                    session.transcript_lines,
                    collected_all,
                ),
            )
        except Exception as exc:
            logger.warning("Summary generation failed: %s", exc)

        record = db_session.get(CallRecord, session.call_id)
        if record:
            record.state           = "done"
            record.completed_at    = datetime.now(timezone.utc)
            record.duration_sec    = duration
            record.transcript_path = transcript_path
            record.caller_name     = caller_name
            record.ai_summary      = ai_summary
            record.end_reason      = reason
            db_session.commit()

        # Record the end-of-call event in analytics so the reason is queryable
        session.record_analytics("call.ended", "system", duration * 1000)
        for event in session.analytics_events:
            db_session.add(CallAnalytics(call_id=session.call_id, **event))
        db_session.commit()

        await safe_send_text("server.call_ended", {
            "call_id": session.call_id,
            "reason": reason,
            "duration_sec": round(duration, 2),
        })

        asyncio.create_task(dispatch_webhooks(
            config=config,
            call_id=session.call_id,
            duration_sec=duration,
            collected=collected_all,
            transcript_path=transcript_path,
            caller_name=caller_name,
            ai_summary=ai_summary,
            booking_details=booking_details,
        ))
        logger.info(
            "Call %s finalized — reason: %s | duration: %.1fs | caller: %s",
            session.call_id, reason, duration, caller_name,
        )

    # ── Main WebSocket loop ───────────────────────────────────────────────────

    send_loop_task = asyncio.create_task(_send_loop())
    watchdog_task: asyncio.Task | None = None
    try:
        stt_latency = await stt.start()
        session.record_analytics("stt_session_open", "stt", stt_latency)

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
        await _send_tts(greeting_text, "greet-001")

        # Opening — invoke data_collection with empty utterance to get first question
        loop = asyncio.get_event_loop()
        graph.states["data_collection"].status = AgentStatus.IN_PROGRESS
        opening_speak, _ = await _invoke_and_follow(
            "data_collection", "", 0, loop, [], chain_depth=0
        )
        opening_text = opening_speak or assistant_cfg.get("greeting_message", "Hello! How can I help you today?")
        session.add_assistant_turn(opening_text)
        transcript_turns.append({"role": "assistant", "content": opening_text})
        await safe_send_text("server.response_text", {"text": opening_text, "audio_id": "open-001"})
        await _send_tts(opening_text, "open-001")

        # Max-duration watchdog
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

        # Read loop — timeout is the full max duration so the watchdog (not this
        # timeout) is responsible for enforcing the call length limit.  A short
        # timeout here was previously causing calls to drop mid-conversation
        # whenever the client stopped sending audio frames (e.g. while TTS plays).
        while not session.state_machine.is_terminal():
            try:
                message = await asyncio.wait_for(
                    ws.receive(), timeout=max_duration
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Call %s terminated — reason: receive_loop_timeout "
                    "(no WebSocket frame for %ds, call likely stalled or client disappeared)",
                    session.call_id, max_duration,
                )
                break

            if "bytes" in message and message["bytes"]:
                logger.debug("← [binary] %d bytes (audio)", len(message["bytes"]))
                await stt.send_audio(message["bytes"])
            elif "text" in message and message["text"]:
                try:
                    msg = json.loads(message["text"])
                    msg_type = msg.get("type", "")
                    logger.debug("← %s", msg_type)
                    if msg_type == "client.audio_end":
                        await stt.finish_utterance()
                    elif msg_type == "client.hangup":
                        logger.info(
                            "Call %s terminated — reason: client_hangup (caller ended the call at %.1fs)",
                            session.call_id, session.duration_sec(),
                        )
                        break
                    elif msg_type == "client.ready":
                        logger.debug("Client ready signal received")
                except json.JSONDecodeError:
                    logger.warning("← unparseable text frame: %.80s", message["text"])

    except WebSocketDisconnect:
        logger.info(
            "Call %s terminated — reason: websocket_disconnect (client disconnected at %.1fs)",
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
        await stt.close()
        if not session.state_machine.is_terminal():
            logger.warning(
                "Call %s terminated — reason: abnormal_exit (session not terminal at %.1fs, finalizing now)",
                session.call_id, session.duration_sec(),
            )
            await _finalize_call("abnormal_exit")
        # Drain the send queue (sentinel priority 2 runs after all pending frames)
        await _send_queue.put((2, next(_send_counter), None))
        try:
            await asyncio.wait_for(send_loop_task, timeout=5.0)
        except (asyncio.TimeoutError, Exception):
            send_loop_task.cancel()
