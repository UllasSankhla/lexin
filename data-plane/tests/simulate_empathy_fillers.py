"""Simulation: empathy fillers trigger on real LLM calls (Cerebras).

Tests that the filler dispatch fires on turns where the planner +
agent pipeline exceeds 500 ms.  Uses the same dispatch logic as the
handler but drives real Cerebras LLM calls as the compute workload
instead of asyncio.sleep.

No WebSocket, no audio, no Deepgram — pure LLM + filler machinery.

Requires: CEREBRAS_API_KEY in environment.

Run:
    cd data-plane
    PYTHONPATH=. python3 -m pytest tests/simulate_empathy_fillers.py -v -s
"""
from __future__ import annotations

import asyncio
import time
import pytest

from app.agents.graph_config import APPOINTMENT_BOOKING
from app.agents.workflow import WorkflowGraph
from app.agents.planner import Planner
from app.agents.registry import build_registry
from app.pipeline.fillers import MAX_FILLERS, _CONTINUATION, _PRIMARY, filler_sequence


# ── Mock transport ────────────────────────────────────────────────────────────

class _MockTransport:
    """Records send_response calls; optionally simulates barge-in."""

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

    @property
    def phrases(self) -> list[str]:
        return [p for p, _ in self.sent]


# ── Dispatch mirror ───────────────────────────────────────────────────────────

async def _dispatch(
    compute_coro,
    transport: _MockTransport,
    *,
    threshold_s: float = 0.5,
    mode: str = "voice",
) -> tuple[str, int, bool]:
    """Run compute_coro with the empathy-filler dispatch logic.

    Mirrors the filler block in handler._process_utterance.
    Returns (speak_text, fillers_sent, barged_in).
    """
    compute_task = asyncio.create_task(compute_coro)
    filler_count = 0
    barged_in = False

    if mode == "voice":
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


# ── Real LLM compute ──────────────────────────────────────────────────────────

def _make_config(extra: dict | None = None) -> dict:
    base = {
        "assistant": {"persona_name": "Aria"},
        "parameters": [
            {"name": "first_name",    "display_label": "First Name",    "data_type": "name",  "required": True,  "collection_order": 1, "extraction_hints": []},
            {"name": "last_name",     "display_label": "Last Name",     "data_type": "name",  "required": True,  "collection_order": 2, "extraction_hints": []},
            {"name": "email_address", "display_label": "Email Address", "data_type": "email", "required": True,  "collection_order": 3, "extraction_hints": []},
        ],
        "faqs": [], "context_files": [], "practice_areas": [], "global_policy_documents": [],
        "_collected": {}, "_booking": {}, "_notes": "", "_tool_results": {},
    }
    if extra:
        base.update(extra)
    return base


async def _llm_compute(utterance: str, config: dict, history: list[dict]) -> str:
    """Run planner + agents exactly as handler._compute() does, return speak text."""
    loop = asyncio.get_running_loop()

    graph    = WorkflowGraph(APPOINTMENT_BOOKING)
    planner  = Planner(graph)
    registry = build_registry("sim-filler", list(history))

    config = {**config, "_workflow_stages": graph.primary_goal_summary()}

    steps = await loop.run_in_executor(
        None, lambda: planner.plan(utterance, history)
    )

    invoke_steps = [s for s in steps if s.action == "invoke" and s.agent_id]
    if not invoke_steps:
        return ""

    results = await asyncio.gather(*[
        loop.run_in_executor(
            None,
            lambda s=s: registry[s.agent_id].process(
                "" if s.use_empty_utterance else utterance,
                {},
                config,
                history,
            ),
        )
        for s in invoke_steps
    ])

    speaks = []
    for step, res in zip(invoke_steps, results):
        if not isinstance(res, Exception) and res.speak:
            speaks.append((res.speak, step.agent_id, res.confidence))

    return planner.combine_speaks(speaks)


# ── Simulations ───────────────────────────────────────────────────────────────

class TestEmpathyFillersTriggerOnRealLLM:

    def _run(self, utterance: str, config: dict, history: list, *, threshold_s: float = 0.5):
        transport = _MockTransport()
        t0 = time.monotonic()
        result, count, barged = asyncio.run(
            _dispatch(
                _llm_compute(utterance, config, history),
                transport,
                threshold_s=threshold_s,
            )
        )
        elapsed_ms = (time.monotonic() - t0) * 1000

        print(f"\n  utterance : {utterance!r}")
        print(f"  threshold : {threshold_s * 1000:.0f}ms")
        print(f"  elapsed   : {elapsed_ms:.0f}ms")
        print(f"  fillers   : {count}  → {transport.phrases}")
        print(f"  response  : {result!r}")
        return result, count, transport

    def test_data_collection_opening_triggers_filler(self):
        """Planner + dc_extract on a name utterance usually exceeds 500ms."""
        config = _make_config()
        result, count, transport = self._run(
            "My name is John Smith.",
            config,
            history=[],
            threshold_s=0.5,
        )
        assert result, "Expected a non-empty agent response"
        assert count >= 1, (
            "Expected at least one filler — if this fails the LLM responded unusually fast. "
            "Lower threshold_s or use a longer utterance."
        )
        assert transport.phrases[0] in _PRIMARY, \
            f"First filler should be from primary pool, got {transport.phrases[0]!r}"

    def test_narrative_opening_triggers_filler(self):
        """Long narrative description almost always exceeds 500ms."""
        config = _make_config({
            "assistant": {"persona_name": "Aria", "narrative_topic": "your legal matter"},
        })
        history = [
            {"role": "assistant", "content": "Could you briefly describe what happened?"},
        ]
        result, count, transport = self._run(
            "I was in a car accident on January 15th. A delivery truck ran a red light "
            "at the intersection of Main and 5th and hit my car on the driver's side. "
            "I was taken to the hospital with a broken collarbone and whiplash.",
            config,
            history=history,
            threshold_s=0.5,
        )
        assert result, "Expected a non-empty agent response"
        assert count >= 1, "Expected at least one filler on long narrative turn"
        assert transport.phrases[0] in _PRIMARY

    def test_continuation_filler_fires_on_very_slow_turn(self):
        """Force >1 filler by using a low threshold (100ms) — guarantees
        the filler loop runs through at least primary + one continuation."""
        config = _make_config()
        result, count, transport = self._run(
            "My name is Sarah Johnson and my email is sarah dot johnson at gmail dot com.",
            config,
            history=[],
            threshold_s=0.1,   # fires almost immediately → more fillers
        )
        assert result, "Expected a non-empty agent response"
        assert count >= 2, (
            f"Expected ≥2 fillers at 100ms threshold, got {count}. "
            f"Phrases: {transport.phrases}"
        )
        assert transport.phrases[0] in _PRIMARY
        for phrase in transport.phrases[1:]:
            assert phrase in _CONTINUATION, \
                f"Non-primary filler after first should be continuation, got {phrase!r}"

    def test_filler_count_does_not_exceed_max(self):
        """Filler count must never exceed MAX_FILLERS regardless of LLM speed."""
        config = _make_config()
        _, count, transport = self._run(
            "I was involved in a wrongful termination dispute at my previous employer. "
            "They fired me after I reported safety violations to OSHA. I have documented "
            "evidence including emails and witness statements.",
            config,
            history=[],
            threshold_s=0.05,  # very low — maximises fillers fired
        )
        assert count <= MAX_FILLERS, \
            f"Filler count {count} exceeded MAX_FILLERS={MAX_FILLERS}"
        assert len(transport.sent) <= MAX_FILLERS

    def test_real_response_always_returned_despite_fillers(self):
        """The real LLM response must always come back even when fillers fired."""
        config = _make_config()
        result, count, _ = self._run(
            "My name is Michael Chen.",
            config,
            history=[],
            threshold_s=0.1,
        )
        assert isinstance(result, str) and result.strip(), \
            "Real response must be a non-empty string"
        # Response should be agent-appropriate (ask for more info or confirm)
        assert len(result) > 10, f"Response suspiciously short: {result!r}"

    def test_barge_in_mid_filler_discards_response(self):
        """Barge-in after first filler: loop stops, compute completes, result discarded."""
        config = _make_config()
        transport = _MockTransport(interrupt_after=1)

        async def run():
            return await _dispatch(
                _llm_compute("My name is David Park.", config, []),
                transport,
                threshold_s=0.1,
            )

        result, count, barged = asyncio.run(run())
        print(f"\n  barged_in : {barged}")
        print(f"  fillers   : {count}  → {transport.phrases}")
        print(f"  result    : {result!r}")

        assert barged is True
        assert count == 1
        assert len(transport.sent) == 1
        # Compute still ran (result is populated even though handler discards it)
        assert isinstance(result, str)
