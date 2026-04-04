"""Tests for empathy filler machinery.

Covers:
  - fillers.py: phrase pools, filler_sequence(), MAX_FILLERS
  - transport: VoiceTransport.interrupted, TextTransport.interrupted (always False)
  - dispatch logic: 500ms threshold, filler loop, barge-in, text-mode skip, disabled flag

The dispatch tests use a self-contained async helper that mirrors the filler
block in handler._process_utterance so they can run without spinning up a
full WebSocket session.

Run:
    cd data-plane
    PYTHONPATH=. python3 -m pytest tests/test_empathy_fillers.py -v
"""
from __future__ import annotations

import asyncio
import pytest

from app.pipeline.fillers import (
    MAX_FILLERS,
    _CONTINUATION,
    _PRIMARY,
    filler_sequence,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

class _MockTransport:
    """Minimal transport stub that records send_response calls.

    interrupt_after: set _interrupted=True after this many sends (0 = never).
    """

    def __init__(self, *, interrupt_after: int = 0) -> None:
        self.sent: list[tuple[str, str]] = []   # (phrase, audio_id)
        self._interrupt_after = interrupt_after
        self._interrupted = False

    @property
    def interrupted(self) -> bool:
        return self._interrupted

    async def send_response(self, text: str, audio_id: str) -> None:
        self.sent.append((text, audio_id))
        if self._interrupt_after and len(self.sent) >= self._interrupt_after:
            self._interrupted = True


async def _dispatch(
    compute_coro,
    transport: _MockTransport,
    *,
    enable_fillers: bool = True,
    threshold_s: float = 0.5,
    mode: str = "voice",
) -> tuple[object, int, bool]:
    """Mirror of the filler dispatch block in handler._process_utterance.

    Returns (compute_result, fillers_sent_count, barged_in).
    """
    compute_task = asyncio.create_task(compute_coro)
    filler_count = 0
    barged_in = False

    if enable_fillers and mode == "voice":
        try:
            await asyncio.wait_for(asyncio.shield(compute_task), timeout=threshold_s)
        except asyncio.TimeoutError:
            for phrase in filler_sequence():
                if filler_count >= MAX_FILLERS:
                    break
                await transport.send_response(phrase, f"filler-1-{filler_count + 1}")
                filler_count += 1
                if compute_task.done():
                    break
                if transport.interrupted:
                    await compute_task
                    barged_in = True
                    break

    result = await compute_task
    return result, filler_count, barged_in


# ── filler_sequence() unit tests ──────────────────────────────────────────────

class TestFillerSequence:

    def test_first_phrase_is_from_primary_pool(self):
        first = next(filler_sequence())
        assert first in _PRIMARY, f"Expected primary phrase, got: {first!r}"

    def test_continuation_phrases_are_from_continuation_pool(self):
        seq = filler_sequence()
        next(seq)  # skip primary
        for _ in range(len(_CONTINUATION)):
            phrase = next(seq)
            assert phrase in _CONTINUATION, f"Expected continuation phrase, got: {phrase!r}"

    def test_first_continuation_pass_has_no_repeats(self):
        seq = filler_sequence()
        next(seq)  # skip primary
        seen: list[str] = []
        for _ in range(len(_CONTINUATION)):
            seen.append(next(seq))
        assert len(seen) == len(set(seen)), f"Duplicate phrases in first continuation pass: {seen}"

    def test_sequence_is_infinite(self):
        seq = filler_sequence()
        # Should never raise StopIteration
        for _ in range(50):
            next(seq)

    def test_max_fillers_is_three(self):
        assert MAX_FILLERS == 3

    def test_primary_pool_is_non_empty(self):
        assert len(_PRIMARY) >= 2

    def test_continuation_pool_is_non_empty(self):
        assert len(_CONTINUATION) >= 2

    def test_all_phrases_are_non_empty_strings(self):
        for phrase in _PRIMARY + _CONTINUATION:
            assert isinstance(phrase, str) and phrase.strip(), \
                f"Empty or non-string phrase: {phrase!r}"


# ── Transport interrupted property ────────────────────────────────────────────

class TestTransportInterrupted:

    def test_voice_transport_not_interrupted_initially(self):
        from unittest.mock import MagicMock, AsyncMock
        from app.transport.voice_transport import VoiceTransport
        transport = VoiceTransport(
            safe_send_text=AsyncMock(),
            safe_send_binary=AsyncMock(),
            session=MagicMock(config={"spell_rules": []}),
            assistant_cfg={"persona_voice": "aura-2-thalia-en", "language": "en-US"},
        )
        assert transport.interrupted is False

    def test_text_transport_interrupted_always_false(self):
        from unittest.mock import AsyncMock
        from app.transport.text_transport import TextTransport
        transport = TextTransport(safe_send_text=AsyncMock())
        assert transport.interrupted is False

    def test_base_transport_default_interrupted_is_false(self):
        """BaseTransport.interrupted default should be False (via TextTransport)."""
        from unittest.mock import AsyncMock
        from app.transport.text_transport import TextTransport
        transport = TextTransport(safe_send_text=AsyncMock())
        # Confirm it comes from base, not an override
        from app.transport.base import BaseTransport
        assert isinstance(transport, BaseTransport)
        assert transport.interrupted is False


# ── Filler dispatch logic ─────────────────────────────────────────────────────

class TestFillerDispatch:

    def test_no_filler_when_flag_disabled(self):
        """enable_empathy_fillers=False → no fillers, even on slow compute."""
        transport = _MockTransport()

        async def slow():
            await asyncio.sleep(0.5)
            return "response"

        result, count, barged = asyncio.run(
            _dispatch(slow(), transport, enable_fillers=False, threshold_s=0.01)
        )
        assert result == "response"
        assert count == 0
        assert transport.sent == []

    def test_no_filler_when_compute_is_fast(self):
        """Compute finishes before threshold → no filler sent."""
        transport = _MockTransport()

        async def fast():
            await asyncio.sleep(0.01)
            return "quick response"

        result, count, barged = asyncio.run(
            _dispatch(fast(), transport, enable_fillers=True, threshold_s=0.5)
        )
        assert result == "quick response"
        assert count == 0
        assert transport.sent == []

    def test_filler_fires_on_slow_response(self):
        """Compute exceeds threshold → at least one filler is sent."""
        transport = _MockTransport()

        async def slow():
            await asyncio.sleep(0.5)
            return "real response"

        result, count, barged = asyncio.run(
            _dispatch(slow(), transport, enable_fillers=True, threshold_s=0.01)
        )
        assert result == "real response"
        assert count >= 1
        assert len(transport.sent) >= 1
        assert transport.sent[0][0] in _PRIMARY, \
            f"First filler should be from primary pool, got {transport.sent[0][0]!r}"

    def test_first_filler_is_primary_phrase(self):
        """The first phrase sent is always from the primary pool."""
        transport = _MockTransport()

        async def slow():
            await asyncio.sleep(0.5)
            return "result"

        asyncio.run(_dispatch(slow(), transport, enable_fillers=True, threshold_s=0.01))
        assert len(transport.sent) >= 1
        assert transport.sent[0][0] in _PRIMARY

    def test_max_fillers_cap_is_respected(self):
        """No more than MAX_FILLERS phrases are sent regardless of compute time."""
        transport = _MockTransport()

        async def very_slow():
            await asyncio.sleep(0.5)  # shorter than it sounds — mock send is instant
            return "late"

        result, count, barged = asyncio.run(
            _dispatch(very_slow(), transport, enable_fillers=True, threshold_s=0.01)
        )
        assert count <= MAX_FILLERS
        assert len(transport.sent) <= MAX_FILLERS

    def test_filler_not_sent_in_text_mode(self):
        """mode='text' → filler block is skipped entirely."""
        transport = _MockTransport()

        async def slow():
            await asyncio.sleep(0.5)
            return "response"

        result, count, barged = asyncio.run(
            _dispatch(slow(), transport, enable_fillers=True, threshold_s=0.01, mode="text")
        )
        assert result == "response"
        assert count == 0
        assert transport.sent == []

    def test_compute_result_is_always_returned(self):
        """The real response is returned regardless of how many fillers fired."""
        transport = _MockTransport()
        expected = "the real answer"

        async def slow():
            await asyncio.sleep(0.5)
            return expected

        result, _, _ = asyncio.run(
            _dispatch(slow(), transport, enable_fillers=True, threshold_s=0.01)
        )
        assert result == expected

    def test_barge_in_stops_filler_loop(self):
        """Barge-in mid-filler: loop stops after current filler, returns barged_in=True."""
        transport = _MockTransport(interrupt_after=1)  # interrupt after first filler

        async def slow():
            await asyncio.sleep(0.3)
            return "response"

        result, count, barged = asyncio.run(
            _dispatch(slow(), transport, enable_fillers=True, threshold_s=0.01)
        )
        assert barged is True
        assert count == 1
        assert len(transport.sent) == 1

    def test_barge_in_still_awaits_compute(self):
        """Even on barge-in, compute is awaited so graph state is consistent."""
        completed = []

        async def slow():
            await asyncio.sleep(0.1)
            completed.append(True)
            return "response"

        transport = _MockTransport(interrupt_after=1)
        asyncio.run(_dispatch(slow(), transport, enable_fillers=True, threshold_s=0.01))
        assert completed == [True], "Compute must complete even on barge-in"

    def test_filler_ids_are_unique_per_turn(self):
        """Each filler audio_id is distinct within a single turn."""
        transport = _MockTransport()

        async def slow():
            await asyncio.sleep(0.5)
            return "done"

        asyncio.run(_dispatch(slow(), transport, enable_fillers=True, threshold_s=0.01))
        ids = [audio_id for _, audio_id in transport.sent]
        assert len(ids) == len(set(ids)), f"Duplicate filler IDs: {ids}"
