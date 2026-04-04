# Model Performance Comparison: qwen-3-235b vs gpt-oss-120b

**Date:** 2026-04-03  
**Benchmark:** LLM latency benchmark — 3 warmup + 10 measured runs per scenario  
**Scope:** Pure LLM + Python routing. Excludes STT and TTS.  
**Method:** Warmup runs discarded to allow Cerebras KV cache to fill on static system prompts.

Source reports:
- qwen-3-235b:  `latency_20260403_155713.md`
- gpt-oss-120b run 1: `latency_20260403_161140.md`
- gpt-oss-120b run 2: `latency_20260403_161537.md`

---

## Scenario Totals — Median Wall Time (ms)

| Scenario             | qwen-3-235b | gpt-oss-120b r1 | gpt-oss-120b r2 | Delta (qwen vs gpt avg) |
|----------------------|------------:|----------------:|----------------:|:-----------------------|
| narrative_collection |        233  |             452 |             216 | noisy — no clear winner |
| scheduling           |        723  |            1245 |            1156 | **+64% slower on gpt** |
| data_collection      |       1806  |            1480 |            1615 | −10% (gpt slightly faster) |
| faq                  |       1374  |            2001 |            2030 | **+47% slower on gpt** |
| planner              |       2372  |            3275 |            3471 | **+42% slower on gpt** |

---

## Per-LLM Call Medians (ms)

| Tag                           | qwen-3-235b | gpt-oss r1 | gpt-oss r2 | Verdict                     |
|-------------------------------|------------:|-----------:|-----------:|:----------------------------|
| dc_confirm_classify           |        212  |        178 |        199 | ~equal                      |
| scheduling_confirm_intent     |        185  |        203 |        173 | ~equal                      |
| scheduling_slot_choice        |        182  |        226 |        200 | slight gpt penalty          |
| faq_answer                    |        172  |        209 |        239 | gpt consistently slower     |
| narrative_done_intent         |        233  |        452 |        216 | high variance — inconclusive|
| faq_multi_match               |        200  |        265 |        262 | gpt consistently slower     |
| dc_extract                    |        309  |        430 |        430 | gpt consistently slower (+40%)|
| scheduling_slot_confirm_speak |        169  |        204 |        214 | gpt slower                  |
| scheduling_slots_speak        |        180  |        507 |        552 | **gpt 3x slower, consistent**|
| planner                       |        388  |        441 |        508 | gpt consistently slower (+30%)|

---

## Spike Behaviour (max observed, ms)

| Tag                | qwen max | gpt-oss r1 max | gpt-oss r2 max |
|--------------------|:--------:|:--------------:|:--------------:|
| dc_confirm_classify|    896   |      2041      |      1781      |
| dc_extract         |   2065   |       680      |      1101      |
| faq_multi_match    |    391   |      3644      |      1109      |
| narrative_done_int |   4792   |      1362      |      1239      |
| planner            |    830   |      2852      |      2268      |
| scheduling_slots   |    214   |      1358      |      1023      |

---

## Verdict

**qwen-3-235b is the better model for this workload.**

Reproducible advantages over gpt-oss-120b:
- `scheduling_slots_speak`: ~3x faster (180ms vs 530ms median) — directly impacts how quickly slot options are read to the caller
- `planner`: ~30% faster (388ms vs ~475ms median) — compounds across every turn in a call
- `faq_multi_match` + `faq_answer`: ~30–40% faster
- `dc_extract`: ~40% faster (309ms vs 430ms median)

gpt-oss-120b's only edge:
- `dc_confirm_classify`: ~10ms faster median — not meaningful in practice

The one scenario where gpt-oss looked competitive on scenario totals (`data_collection`) is explained by the benchmark completing in fewer turns on some runs, not by faster individual calls.

**Recommendation: keep qwen-3-235b as the production model.**
