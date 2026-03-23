"""Abstract transport interface — decouples I/O from call orchestration logic."""
from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio


class BaseTransport(ABC):
    """Abstract transport layer for a call session.

    Decouples input delivery (STT vs typed text) and output delivery
    (TTS audio vs JSON text message) from the core agent orchestration logic.
    """

    @abstractmethod
    async def start(self) -> None:
        """Initialise the transport (e.g. open STT session). No-op for text mode."""

    @abstractmethod
    async def close(self) -> None:
        """Tear down the transport."""

    @abstractmethod
    async def send_response(self, text: str, audio_id: str) -> None:
        """Deliver a response to the user.

        Voice: stream TTS audio with barge-in detection.
        Text: send a server.message JSON frame.
        """

    @abstractmethod
    async def handle_binary_frame(self, data: bytes) -> None:
        """Process an incoming binary WebSocket frame (PCM audio in voice mode; no-op in text)."""

    @abstractmethod
    async def handle_text_frame(self, msg: dict) -> bool:
        """Process an incoming text WebSocket frame.

        Returns True when the client has signalled a hangup.
        """

    @property
    @abstractmethod
    def utterance_queue(self) -> asyncio.Queue:
        """Async queue populated with user utterances as they arrive."""
