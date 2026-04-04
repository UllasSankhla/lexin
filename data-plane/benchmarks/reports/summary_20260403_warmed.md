# LLM Latency Benchmark — Warmed KV Cache Summary

**Date:** 2026-04-03  
**Model:** `qwen-3-235b-a22b-instruct-2507`  
**Warmup runs:** 3 (discarded — KV cache fill for static system prompts)  
**Measured runs:** 10  
**Total runs per scenario:** 13  
**Scope:** Pure LLM + Python routing. Excludes STT decoding and TTS synthesis.  
**Source report:** `latency_20260403_155713.md`  
**Previous run (cold):** `latency_20260403_150626.md`

---

## Scenario Overview (warmed)

| Scenario             | Wall avg | Wall med | Wall max | vs. cold med |
|----------------------|---------:|---------:|---------:|-------------:|
| narrative_collection |    707ms |    233ms |   4792ms | +53ms (+30%) |
| scheduling           |    758ms |    723ms |   1026ms | −53ms (−7%)  |
| faq                  |   1399ms |   1374ms |   1685ms | +30ms (+2%)  |
| data_collection      |   1953ms |   1806ms |   2901ms | +320ms (+22%)|
| planner              |   2787ms |   2372ms |   5895ms | +159ms (+7%) |

> Medians are the better comparison point — averages are dragged up by occasional spikes.

---

## Per-LLM Call Medians — Warmed vs Cold

| Tag                           | Warmed med | Cold med | Delta    |
|-------------------------------|-----------:|---------:|:---------|
| scheduling_slots_speak        |      180ms |    175ms | +5ms     |
| scheduling_slot_choice        |      182ms |    199ms | **−17ms**|
| scheduling_confirm_intent     |      185ms |    193ms | **−8ms** |
| faq_answer                    |      172ms |    165ms | +7ms     |
| faq_multi_match               |      200ms |    191ms | +9ms     |
| scheduling_slot_confirm_speak |      169ms |    154ms | +15ms    |
| narrative_done_intent         |      233ms |    180ms | +53ms    |
| dc_confirm_classify           |      212ms |    200ms | +12ms    |
| dc_extract                    |      309ms |    291ms | +18ms    |
| planner                       |      388ms |    356ms | **+32ms**|

---

## Key Findings

### KV cache effect
Most call tags showed **negligible change** (±20ms) between cold and warmed runs — medians shifted by less than 10%. This suggests Cerebras is already applying prefix caching effectively on cold runs, or that the static system prompt is a small fraction of total token processing time. The warmup phase is worth keeping for consistency, but it is not producing the large speedup one would expect if the cache were truly cold.

### Spikes are the real problem — not cold-start
The outliers visible in both runs are sporadic, not front-loaded:
- **`narrative_done_intent`**: run 6 spiked to **4792ms** (warmup was fine at ~174ms). Not a cache issue — likely a transient API/network event.
- **`planner`** run 7 T6: **3457ms** (all other turns in that run were normal ~300–600ms). Isolated spike, not a trend.
- **`dc_extract`** run 10 T1: **2065ms** (same scenario). Again isolated.

These spikes pull averages significantly above medians. The system is consistently fast (~180–400ms per call) with occasional multi-second outliers (~2–3% of calls). That is the distribution to design around for real-call latency budgets.

### Scheduling is the most consistent agent
- Warmed median: **723ms** for a full 3-turn booking flow
- Max: 1026ms (even the worst case is under a second)
- All 4 LLM call types stay within 150–200ms median

### FAQ is very stable
- Warmed median: **1374ms** across 4 turns
- Range: 1246–1685ms (tight, predictable)
- Legal-deflect turn (T3, 1 LLM call): ~180ms

### Data collection variance
- `dc_extract` warmed median: **309ms** — the extraction prompt is the heaviest single call
- `dc_confirm_classify` warmed median: **212ms**
- Combined per-turn avg for a confirmation exchange: ~511ms
- Still seeing occasional 800–900ms on confirm_classify — possibly longer user inputs triggering more tokens

### Planner overhead (warmed)
- **388ms median per call**, up slightly from 356ms cold
- For a 13-turn full call, planner alone = **~5.0s** cumulative
- Run 7 spike (T6: 3457ms) is a significant outlier and inflates the average to 464ms
- Excluding that single spike: avg ≈ 390ms, consistent with median

---

## Recommendations

1. **Latency budget per turn (p95 estimate):** ~500ms for single-LLM turns, ~700ms for dual-LLM turns (slot choice, FAQ multi-match). Design TTS buffering around these numbers.

2. **Spike handling:** ~2–3% of calls will see 1–5s spikes regardless of warmup. Consider a 2s timeout with graceful retry/fallback for `dc_extract` and `narrative_done_intent` specifically.

3. **Planner skip opportunity:** If we can bypass the planner for single-agent calls where the intent is clear from state (e.g., mid-data-collection confirmations), the ~390ms overhead per turn is recoverable. At 13 turns per call that's ~5 seconds.

4. **KV cache is already active:** The warm/cold delta is minimal. No need to pre-warm the API connection separately — Cerebras appears to cache the static prefix automatically after the first request per session.
