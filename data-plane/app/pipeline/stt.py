"""Deepgram streaming Speech-to-Text — Flux model (SDK v6, v2 endpoint).

Turn-end detection
==================
The v2 API uses a TurnInfo event stream instead of is_final/speech_final:

  TurnInfo.event == "Update"          → on_interim  (partial transcript)
  TurnInfo.event == "StartOfTurn"     → on_interim  (speech started)
  TurnInfo.event == "EagerEndOfTurn"  → on_interim  + store as pending
  TurnInfo.event == "TurnResumed"     → on_interim  (was EOT but continued)
  TurnInfo.event == "EndOfTurn"       → on_final    (definitive turn end)

If the session closes before "EndOfTurn" arrives, the last "EagerEndOfTurn"
text is flushed as a final so the pipeline never stalls.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Awaitable, Callable

from deepgram import AsyncDeepgramClient
from deepgram.core.events import EventType
from deepgram.listen.v2.socket_client import V2SocketClientResponse
from deepgram.listen.v2.types import ListenV2Connected, ListenV2TurnInfo, ListenV2FatalError

from app.config import settings

logger = logging.getLogger(__name__)


class STTSession:
    """Manages a single Deepgram Flux live transcription session (v2 API)."""

    def __init__(
        self,
        on_interim: Callable[[str, float], Awaitable[None]],
        on_final: Callable[[str, float, float], Awaitable[None]],
        language: str = "en-US",
        spell_rules: list[dict] | None = None,
    ) -> None:
        self._on_interim = on_interim
        self._on_final = on_final
        self._language = language
        self._spell_rules = spell_rules or []
        self._session_start = time.monotonic()
        self._connected = False

        # Held open for the session lifetime.
        self._ctx = None
        self._connection = None
        self._listen_task: asyncio.Task | None = None

        # Text from EagerEndOfTurn used as fallback if EndOfTurn never arrives.
        self._pending_text: str = ""
        self._pending_confidence: float = 0.0

        # Hold-off: delay on_final by eot_hold_ms after EndOfTurn so that a
        # TurnResumed arriving late can still cancel the final dispatch.
        self._eot_hold_task: asyncio.Task | None = None

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _apply_spell_rules(self, text: str) -> str:
        for rule in self._spell_rules:
            if rule.get("rule_type") == "substitution":
                text = text.replace(rule["wrong_form"], rule["correct_form"])
        return text

    def _word_confidence(self, words: list) -> float:
        if words:
            scores = [getattr(w, "confidence", None) for w in words if getattr(w, "confidence", None) is not None]
            if scores:
                return sum(scores) / len(scores)
        return 1.0

    # ── Event handlers ────────────────────────────────────────────────────────

    def _on_open(self, _) -> None:
        logger.info("STT WebSocket connection opened")

    def _on_close(self, _) -> None:
        logger.info("STT WebSocket connection closed")
        # Flush any pending eager EOT text as a final result.
        if self._pending_text:
            elapsed_ms = (time.monotonic() - self._session_start) * 1000
            logger.info(
                "STT session closed with pending text — scheduling flush | text=%r", self._pending_text
            )
            asyncio.create_task(
                self._on_final(self._pending_text, self._pending_confidence, elapsed_ms)
            )
            self._pending_text = ""
            self._pending_confidence = 0.0

    def _on_error(self, error) -> None:
        logger.error("STT error: %s", error)

    def _on_message(self, message: V2SocketClientResponse) -> None:
        elapsed_ms = (time.monotonic() - self._session_start) * 1000

        if isinstance(message, ListenV2Connected):
            logger.info("STT connected | request_id=%s", message.request_id)
            return

        if isinstance(message, ListenV2FatalError):
            logger.error("STT fatal error: %s", message)
            return

        if not isinstance(message, ListenV2TurnInfo):
            logger.debug("STT unhandled message: type=%s", type(message).__name__)
            return

        event = message.event
        raw_text = (message.transcript or "").strip()
        text = self._apply_spell_rules(raw_text)
        confidence = self._word_confidence(message.words)

        logger.info(
            "STT TurnInfo | event=%s | text=%r | conf=%.2f | eot_conf=%.2f | elapsed=%.0fms",
            event, text, confidence, message.end_of_turn_confidence, elapsed_ms,
        )

        if not text:
            logger.debug("STT empty transcript — skipping")
            return

        if event == "EndOfTurn":
            logger.info("STT end-of-turn | text=%r | conf=%.2f — starting %.0fms hold", text, confidence, settings.deepgram_eot_hold_ms)
            self._pending_text = ""
            self._pending_confidence = 0.0
            self._cancel_eot_hold()
            self._eot_hold_task = asyncio.create_task(
                self._dispatch_final_after_hold(text, confidence, elapsed_ms)
            )

        elif event == "EagerEndOfTurn":
            logger.info("STT eager end-of-turn — storing as pending | text=%r", text)
            self._pending_text = text
            self._pending_confidence = confidence
            asyncio.create_task(self._on_interim(text, confidence))

        elif event == "TurnResumed":
            logger.info("STT turn resumed (was eager EOT) — cancelling hold | text=%r", text)
            self._cancel_eot_hold()
            self._pending_text = ""
            self._pending_confidence = 0.0
            asyncio.create_task(self._on_interim(text, confidence))

        else:
            # "Update" or "StartOfTurn" — cancel any pending hold (user still speaking)
            if event == "StartOfTurn":
                self._cancel_eot_hold()
            asyncio.create_task(self._on_interim(text, confidence))

    def _cancel_eot_hold(self) -> None:
        if self._eot_hold_task and not self._eot_hold_task.done():
            self._eot_hold_task.cancel()
            logger.debug("STT EOT hold cancelled")
        self._eot_hold_task = None

    async def _dispatch_final_after_hold(self, text: str, confidence: float, elapsed_ms: float) -> None:
        """Wait eot_hold_ms then fire on_final — cancelled if user resumes speaking."""
        try:
            hold_sec = settings.deepgram_eot_hold_ms / 1000.0
            await asyncio.sleep(hold_sec)
            logger.info("STT EOT hold elapsed — dispatching final | text=%r", text)
            await self._on_final(text, confidence, elapsed_ms)
        except asyncio.CancelledError:
            logger.info("STT EOT hold cancelled — user resumed speaking | text=%r", text)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> float:
        """Open Deepgram Flux v2 streaming connection. Returns elapsed ms."""
        t0 = time.monotonic()

        client = AsyncDeepgramClient(api_key=settings.deepgram_api_key)

        self._ctx = client.listen.v2.connect(
            model=settings.deepgram_stt_model,
            encoding="linear16",
            sample_rate=str(16000),
            eot_threshold=str(settings.deepgram_eot_threshold),
            eot_timeout_ms=str(settings.deepgram_eot_timeout_ms),
        )
        self._connection = await self._ctx.__aenter__()

        self._connection.on(EventType.OPEN, self._on_open)
        self._connection.on(EventType.MESSAGE, self._on_message)
        self._connection.on(EventType.CLOSE, self._on_close)
        self._connection.on(EventType.ERROR, self._on_error)

        self._listen_task = asyncio.create_task(self._connection.start_listening())
        self._connected = True

        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.info(
            "STT session ready | model=%s | eot_threshold=%.2f | eot_timeout_ms=%d | opened_in=%.0fms",
            settings.deepgram_stt_model,
            settings.deepgram_eot_threshold,
            settings.deepgram_eot_timeout_ms,
            elapsed_ms,
        )
        return elapsed_ms

    async def send_audio(self, chunk: bytes) -> None:
        if self._connection and self._connected:
            logger.debug("STT sending audio | bytes=%d", len(chunk))
            await self._connection._send(chunk)
        else:
            logger.warning("STT send_audio called but connection not ready")

    async def finish_utterance(self) -> None:
        """In v2, turn detection is fully model-driven — no manual finalize needed."""
        logger.info("STT finish_utterance called (no-op in v2; EOT is model-driven)")

    async def close(self) -> None:
        self._cancel_eot_hold()
        self._connected = False
        if self._connection:
            try:
                await self._connection.send_close_stream()
            except Exception:
                pass
        if self._ctx:
            try:
                await self._ctx.__aexit__(None, None, None)
            except Exception:
                pass
        if self._listen_task:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
        logger.info("STT session closed")
