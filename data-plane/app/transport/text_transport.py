"""Text-only transport — no audio, typed messages only."""
from __future__ import annotations

import asyncio
import logging
from typing import Callable, Awaitable

from app.transport.base import BaseTransport

logger = logging.getLogger(__name__)


class TextTransport(BaseTransport):
    """Text-only transport: user sends typed messages, assistant replies with text."""

    def __init__(
        self,
        safe_send_text: Callable[[str, dict], Awaitable[None]],
    ) -> None:
        self._safe_send_text = safe_send_text
        self._utterance_queue: asyncio.Queue = asyncio.Queue()

    @property
    def utterance_queue(self) -> asyncio.Queue:
        return self._utterance_queue

    async def start(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def send_response(self, text: str, audio_id: str) -> None:
        if not text or not text.strip():
            return
        await self._safe_send_text("server.message", {"text": text, "audio_id": audio_id})

    async def handle_binary_frame(self, data: bytes) -> None:
        pass  # no audio in text mode

    async def handle_text_frame(self, msg: dict) -> bool:
        msg_type = msg.get("type", "")
        if msg_type == "client.message":
            text = ((msg.get("payload") or {}).get("text") or "").strip()
            if text:
                await self._utterance_queue.put(text)
        elif msg_type == "client.hangup":
            return True
        return False
