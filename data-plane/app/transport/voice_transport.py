"""Voice transport — STT input (Deepgram) + TTS output (Deepgram) with barge-in."""
from __future__ import annotations

import asyncio
import logging
import struct
import time
from typing import Callable, Awaitable

from app.pipeline.stt import STTSession
from app.pipeline.tts import TTSClient
from app.transport.base import BaseTransport

logger = logging.getLogger(__name__)

_TTS_MERGE_BYTES = 8192


def _encode_audio_frame(audio_id: str, audio_bytes: bytes) -> bytes:
    id_bytes = audio_id.encode("utf-8")
    return struct.pack("<I", len(id_bytes)) + id_bytes + audio_bytes


class VoiceTransport(BaseTransport):
    """Voice I/O: streams PCM from client into STT, streams TTS back as PCM."""

    def __init__(
        self,
        safe_send_text: Callable[[str, dict], Awaitable[None]],
        safe_send_binary: Callable[[bytes], Awaitable[None]],
        session,          # CallSession — used for analytics and config
        assistant_cfg: dict,
    ) -> None:
        self._safe_send_text = safe_send_text
        self._safe_send_binary = safe_send_binary
        self._session = session
        self._assistant_cfg = assistant_cfg

        self._utterance_queue: asyncio.Queue = asyncio.Queue()
        self._tts = TTSClient(voice=assistant_cfg.get("persona_voice"))
        self._tts_playing = False
        self._tts_cancel = asyncio.Event()
        self._last_utterance_t: float = 0.0
        self._stt: STTSession | None = None

    @property
    def utterance_queue(self) -> asyncio.Queue:
        return self._utterance_queue

    async def start(self) -> None:
        """Open the Deepgram STT session and wire STT callbacks."""

        async def on_interim(text: str, confidence: float) -> None:
            if self._tts_playing and not self._tts_cancel.is_set():
                logger.info(
                    "Barge-in detected while TTS playing — cancelling stream | text=%r",
                    text[:60],
                )
                self._tts_cancel.set()
            await self._safe_send_text("server.transcript_interim", {
                "text": text, "confidence": confidence, "is_final": False,
            })

        async def on_final(text: str, confidence: float, elapsed_ms: float) -> None:
            self._last_utterance_t = time.monotonic()
            await self._safe_send_text("server.transcript_final", {
                "text": text, "confidence": confidence,
            })
            await self._utterance_queue.put(text)

        self._stt = STTSession(
            on_interim=on_interim,
            on_final=on_final,
            language=self._assistant_cfg.get("language", "en-US"),
            spell_rules=self._session.config.get("spell_rules", []),
        )
        latency_ms = await self._stt.start()
        self._session.record_analytics("stt_session_open", "stt", latency_ms)

    async def close(self) -> None:
        if self._stt:
            await self._stt.close()

    async def send_response(self, text: str, audio_id: str) -> None:
        """Stream TTS audio to the client with barge-in detection."""
        if not text or not text.strip():
            logger.warning(
                "send_response called with empty text — skipping (audio_id=%s)", audio_id
            )
            return
        self._tts_cancel.clear()
        self._tts_playing = True
        chunks_sent = 0
        try:
            await self._safe_send_text("server.tts_stream_start", {"audio_id": audio_id})
            pending = bytearray()
            async for chunk, first_chunk_latency in self._tts.stream(text):
                if self._tts_cancel.is_set():
                    logger.debug(
                        "TTS stream interrupted (barge-in) after %d chunks — audio_id=%s",
                        chunks_sent, audio_id,
                    )
                    await self._safe_send_text("server.tts_interrupted", {"audio_id": audio_id})
                    return
                if first_chunk_latency is not None:
                    self._session.record_analytics("tts_first_chunk", "tts", first_chunk_latency)
                pending.extend(chunk)
                if len(pending) >= _TTS_MERGE_BYTES:
                    if chunks_sent == 0 and self._last_utterance_t:
                        latency_ms = (time.monotonic() - self._last_utterance_t) * 1000
                        self._session.record_analytics("response_latency", "pipeline", latency_ms)
                        self._last_utterance_t = 0.0
                    await self._safe_send_binary(_encode_audio_frame(audio_id, bytes(pending)))
                    chunks_sent += 1
                    pending = bytearray()
            if pending:
                if chunks_sent == 0 and self._last_utterance_t:
                    latency_ms = (time.monotonic() - self._last_utterance_t) * 1000
                    self._session.record_analytics("response_latency", "pipeline", latency_ms)
                    self._last_utterance_t = 0.0
                await self._safe_send_binary(_encode_audio_frame(audio_id, bytes(pending)))
                chunks_sent += 1
            await self._safe_send_text("server.tts_stream_end", {"audio_id": audio_id})
        except Exception as exc:
            logger.error("TTS streaming error: %s", exc)
            await self._safe_send_text("server.error", {
                "code": "tts_error", "message": "Voice synthesis failed", "fatal": False,
            })
        finally:
            self._tts_playing = False

    async def handle_binary_frame(self, data: bytes) -> None:
        if self._stt:
            await self._stt.send_audio(data)

    async def handle_text_frame(self, msg: dict) -> bool:
        msg_type = msg.get("type", "")
        if msg_type == "client.audio_end":
            if self._stt:
                await self._stt.finish_utterance()
        elif msg_type == "client.barge_in":
            if self._tts_playing and not self._tts_cancel.is_set():
                logger.debug("Barge-in signal from client VAD — cancelling TTS")
                self._tts_cancel.set()
        elif msg_type == "client.hangup":
            return True
        elif msg_type == "client.ready":
            logger.debug("Client ready signal received")
        return False
