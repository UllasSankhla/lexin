"""Deepgram Text-to-Speech integration (SDK v6)."""
from __future__ import annotations

import logging
import time
from typing import AsyncGenerator

from deepgram import AsyncDeepgramClient

from app.config import settings

logger = logging.getLogger(__name__)

_RIFF_MAGIC = b"RIFF"


def _strip_wav_header(data: bytes) -> bytes:
    """Strip the RIFF/WAV container header and return raw PCM bytes.

    Deepgram's TTS streaming API wraps each audio chunk in its own WAV
    container.  Playing those header bytes as PCM produces a loud click
    (the 'RIFF' magic and size fields map to high-amplitude samples) followed
    by what sounds like undersampling (the remaining 44 header bytes displace
    actual audio samples).

    Standard PCM WAV layout:
      'RIFF' (4) + file_size (4) + 'WAVE' (4)
      then one or more sub-chunks, each: id (4) + size (4) + payload
      The 'data' sub-chunk payload is the raw PCM samples.

    We scan sub-chunks until we find 'data', then return everything from
    that point.  If the magic bytes are absent the data is already raw PCM
    and is returned unchanged.
    """
    if len(data) < 12 or data[:4] != _RIFF_MAGIC:
        return data

    offset = 12  # skip 'RIFF' (4) + file_size (4) + 'WAVE' (4)
    while offset + 8 <= len(data):
        chunk_id = data[offset:offset + 4]
        chunk_size = int.from_bytes(data[offset + 4:offset + 8], "little")
        offset += 8
        if chunk_id == b"data":
            logger.debug(
                "Stripped WAV header (%d bytes stripped, %d bytes PCM remain)",
                offset, len(data) - offset,
            )
            return data[offset:]
        offset += chunk_size  # skip non-data sub-chunk

    logger.warning("WAV 'data' chunk not found — returning raw bytes unchanged")
    return data


class TTSClient:
    def __init__(self, voice: str | None = None):
        self._client = AsyncDeepgramClient(api_key=settings.deepgram_api_key)
        self._voice = voice or settings.deepgram_tts_model

    async def stream(self, text: str) -> AsyncGenerator[tuple[bytes, float | None], None]:
        """
        Stream TTS audio chunks as they arrive from Deepgram.

        Yields (chunk_bytes, first_chunk_latency_ms) for the first chunk and
        (chunk_bytes, None) for every subsequent chunk.
        Audio is linear16 PCM at 24 kHz mono.
        """
        logger.info(
            "TTS stream request | model=%s encoding=linear16 sample_rate=24000 "
            "text_len=%d preview=%r",
            self._voice, len(text), text[:80],
        )
        t0 = time.monotonic()
        chunk_num = 0
        total_bytes = 0

        async for raw_chunk in self._client.speak.v1.audio.generate(
            text=text,
            model=self._voice,
            encoding="linear16",
            sample_rate=24000,
        ):
            chunk = _strip_wav_header(raw_chunk)
            if not chunk:
                logger.debug("TTS chunk was header-only after stripping — skipping")
                continue

            chunk_num += 1
            total_bytes += len(chunk)
            elapsed_ms = (time.monotonic() - t0) * 1000

            if chunk_num == 1:
                logger.info(
                    "TTS first chunk | chunk=1 raw_bytes=%d pcm_bytes=%d elapsed=%.0fms",
                    len(raw_chunk), len(chunk), elapsed_ms,
                )
                yield chunk, elapsed_ms
            else:
                logger.debug(
                    "TTS chunk | chunk=%d raw_bytes=%d pcm_bytes=%d "
                    "cumulative_bytes=%d elapsed=%.0fms",
                    chunk_num, len(raw_chunk), len(chunk), total_bytes, elapsed_ms,
                )
                yield chunk, None

        total_ms = (time.monotonic() - t0) * 1000
        logger.info(
            "TTS stream complete | chunks=%d total_bytes=%d total_ms=%.0f "
            "avg_chunk_bytes=%d",
            chunk_num, total_bytes, total_ms,
            total_bytes // chunk_num if chunk_num else 0,
        )
