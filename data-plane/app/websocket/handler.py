"""Main WebSocket connection handler for voice calls.

Orchestration flow
==================
1.  STT session opens (Deepgram nova-2-phonecall); greeting is spoken.
2.  BookingWorkflow.get_opening() is called → first field question is spoken.
3.  Read loop: binary frames → STT; text frames → control messages.
4.  STT on_final fires → _process_utterance(text).
5.  _process_utterance delegates entirely to BookingWorkflow.process_utterance().
6.  WorkflowResult carries:
      .speak          → TTS + transcript
      .field_confirmed → DB persist + client event
      .booking_details → DB persist + client event
      .call_complete   → trigger _handle_completion()
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
from app.pipeline.booking_workflow import BookingWorkflow
from app.pipeline.llm import LLMClient, LLMToolkit, TOOLKIT_SYSTEM_PROMPT
from app.pipeline.parameter_collector import load_parameters
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
    from app.models.gathered_parameter import GatheredParameter

    seq = [0]
    config = session.config
    assistant_cfg = config.get("assistant", {})

    # ── Pipeline initialisation ───────────────────────────────────────────────
    # Parameters to collect are loaded entirely from the control-plane config.
    # Nothing in the data plane hardcodes which fields to ask for.
    collection_state = load_parameters(config)
    session.collection_state = collection_state

    # LLMToolkit uses a minimal extraction-focused system prompt, NOT the
    # conversational booking prompt. A conversational system prompt would
    # cause the LLM to produce full sentences instead of bare extracted values.
    llm_client = LLMClient(system_prompt=TOOLKIT_SYSTEM_PROMPT)
    llm_toolkit = LLMToolkit(llm_client)
    llm_toolkit.on_llm_response = lambda purpose, latency_ms, tokens: \
        session.record_analytics(f"llm_response.{purpose}", "llm", latency_ms)

    workflow = BookingWorkflow(collection_state, llm_toolkit, config)

    tts = TTSClient(voice=assistant_cfg.get("persona_voice"))

    utterance_lock = asyncio.Lock()

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

    # ── Core: process a caller utterance through the workflow ─────────────────

    async def _process_utterance(caller_text: str) -> None:
        if session.state_machine.is_terminal():
            return

        session.add_user_turn(caller_text)
        await safe_send_text("server.thinking", {})

        # Run workflow (sync LLM calls) in a thread executor
        t0 = time.monotonic()
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: workflow.process_utterance(caller_text),
            )
        except Exception as exc:
            logger.error("Workflow error: %s", exc, exc_info=True)
            await safe_send_text("server.error", {"code": "workflow_error", "message": "Processing failed", "fatal": False})
            return

        session.record_analytics("workflow_turn", "llm", (time.monotonic() - t0) * 1000)
        session.add_assistant_turn(result.speak)

        audio_id = f"resp-{seq[0] + 1}"
        await safe_send_text("server.response_text", {"text": result.speak, "audio_id": audio_id})

        # Persist confirmed field
        if result.field_confirmed:
            name, raw, normalized = result.field_confirmed
            remaining = sum(
                1 for p in collection_state.required_params
                if p.name not in collection_state.collected
            )
            await safe_send_text("server.parameter_collected", {
                "parameter_name": name,
                "value": normalized,
                "remaining_count": remaining,
            })
            db_session.add(GatheredParameter(
                call_id=session.call_id,
                parameter_name=name,
                raw_value=raw,
                normalized_value=normalized,
                validated=True,
            ))
            db_session.commit()

        # Persist derived field (e.g. case_type from case_description)
        if result.extra_field_confirmed:
            ename, eraw, enormalized = result.extra_field_confirmed
            await safe_send_text("server.parameter_collected", {
                "parameter_name": ename,
                "value": enormalized,
                "remaining_count": 0,
            })
            db_session.add(GatheredParameter(
                call_id=session.call_id,
                parameter_name=ename,
                raw_value=eraw,
                normalized_value=enormalized,
                validated=True,
            ))
            db_session.commit()

        # Speak the response
        await _send_tts(result.speak, audio_id)

        # Persist booking and trigger completion
        if result.booking_details:
            await safe_send_text("server.booking_confirmed", {"booking": result.booking_details})
            record = db_session.get(CallRecord, session.call_id)
            if record:
                record.booking_id     = result.booking_details.get("booking_id")
                record.booking_status = result.booking_details.get("status")
                db_session.commit()

        if result.call_complete:
            await asyncio.sleep(2)   # let TTS finish playing
            await _handle_completion(result.booking_details)

    # ── Completion ────────────────────────────────────────────────────────────

    async def _handle_completion(booking_details: dict | None = None) -> None:
        session.state_machine.transition(CallState.COMPLETING)
        await safe_send_text("server.call_completing", {
            "collected_parameters": collection_state.collected,
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
                    collection_state.collected if collection_state else {},
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
            collected=collection_state.collected,
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

        # Opening — ask the first collection question (or present slots if no
        # parameters are configured).  Runs in executor because get_opening()
        # makes a synchronous LLM call to generate the first question.
        loop = asyncio.get_event_loop()
        opening_text = await loop.run_in_executor(None, workflow.get_opening)
        session.add_assistant_turn(opening_text)
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
