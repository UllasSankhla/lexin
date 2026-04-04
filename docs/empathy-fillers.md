# Empathy Fillers

## Problem

Every caller utterance triggers a planner LLM call (~390 ms median) followed by one or more agent LLM calls (~200–400 ms each). On most turns the combined latency lands between 600–1000 ms. During that window the caller hears dead silence, which feels broken or laggy on a voice call.

The empathy filler system masks this silence by sending a short, generic phrase to TTS as soon as processing exceeds a configurable threshold, giving the caller something to hear while the real response is still being computed.

---

## How It Works

```
STT final arrives
        │
        ├──► _compute() starts as asyncio.Task  (planner + agents, real LLM work)
        │
        └──► wait 500 ms for _compute() to finish
                    │
             ┌──────┴──────┐
             │ done ≤500ms │   No filler — real response plays immediately
             └─────────────┘
                    │
             ┌──────┴──────────────────┐
             │ still running at 500ms  │   Filler loop fires
             └─────────────────────────┘
                    │
                    ▼
             filler_1 → TTS  ("Sure, one moment.")          ~1–1.5s audio
                    │
             _compute() done? ──YES──► real response queued, plays next
                    │ NO
                    ▼
             filler_2 → TTS  ("Thanks for your patience.")
                    │
             _compute() done? ──YES──► real response queued, plays next
                    │ NO
                    ▼
             filler_3 → TTS  ("Almost there.")              [MAX_FILLERS cap]
                    │
             await _compute() regardless ──► real response queued, plays next
```

Fillers and the real response are sent through the same `transport.send_response` path and share the same audio output queue on the client — they play back-to-back with no gap.

---

## Configuration

Controlled by a single boolean on `AssistantConfig`:

| Field | Type | Default | Description |
|---|---|---|---|
| `enable_empathy_fillers` | `bool` | `false` | Enable the filler dispatch for this assistant |

Set via the control-plane API (PATCH `/api/v1/assistant/`). The value is exported in the config payload consumed by the data plane and read per-call from `assistant_cfg.get("enable_empathy_fillers", False)`.

Fillers only fire in **voice mode**. Text-mode sessions skip the filler block entirely.

---

## Phrase Pools

Two tiers, drawn in order:

**Primary** (first phrase — always from this pool):

- "Sure, one moment."
- "Let me check that for you."
- "Of course, just a second."
- "Got it, one moment."

**Continuation** (subsequent phrases if processing is still running):

- "Thanks for your patience."
- "Almost there."
- "Just a moment longer."
- "Still on it, thank you."

Phrases are sampled randomly. Continuations are shuffled on each call so they don't repeat consecutively within a turn. The cap is `MAX_FILLERS = 3` total phrases per turn (primary + up to 2 continuations).

**Design rule:** No phrase carries semantic content that could be wrong. Nothing like "I found your slot" or "Let me book that." Pure process language only.

---

## Threshold and Timing

The threshold is hardcoded at **500 ms** in `handler._process_utterance`. This was chosen because:

- Scheduling turns (slot pick, confirmation) complete in ~180–250 ms — well under threshold, no filler fired.
- Data collection + planner combined runs ~600–1000 ms — reliably over threshold.
- 500 ms is the point at which silence becomes perceptibly awkward on a voice call.

Filler TTS audio durations (approximate, Deepgram aura-2):

| Phrase | Speaking time |
|---|---|
| "Sure, one moment." | ~0.9s |
| "Got it, one moment." | ~0.9s |
| "Let me check that for you." | ~1.1s |
| "Thanks for your patience." | ~1.0s |

Since the mock transport's `send_response` is instant in tests but real TTS takes ~1s per phrase, the real-world behavior is: filler_1 plays for ~1s, if compute is still running at that point filler_2 starts, and so on — matching the 600–1000 ms real compute window closely.

---

## Barge-In Handling

If the caller speaks while a filler is playing:

1. `VoiceTransport` sets `_tts_cancel` — the filler's `send_response` returns early.
2. `transport.interrupted` returns `True`.
3. The filler loop stops. No more fillers are queued.
4. `_compute()` is still awaited — it must complete to keep graph state consistent.
5. The assembled response is discarded (not sent to TTS) and control returns to the utterance processor, which will pick up the caller's new utterance.

This is safe because `_utterance_processor` holds the `utterance_lock` for the duration of `_process_utterance`. The new utterance sits in the queue and is processed as soon as the lock is released.

---

## Logging

All filler activity is logged at `INFO` level with a `FILLER |` prefix:

```
FILLER | turn=3 | threshold exceeded (500 ms) — starting filler loop
FILLER | turn=3 | sending filler 1/3 | id=filler-2-1 | text='Sure, one moment.'
FILLER | turn=3 | sending filler 2/3 | id=filler-2-2 | text='Thanks for your patience.'
FILLER | turn=3 | compute finished after 2 filler(s) | elapsed=1843ms
```

Exit conditions:

| Log line | Meaning |
|---|---|
| `compute finished after N filler(s) \| elapsed=Xms` | Normal exit — compute done between fillers |
| `max fillers (3) reached — awaiting compute` | All 3 fillers sent, still awaiting compute |
| `barge-in after N filler(s) — discarding response` | Caller spoke; response discarded |

---

## Implementation

| File | Role |
|---|---|
| `data-plane/app/pipeline/fillers.py` | Phrase pools, `filler_sequence()` generator, `MAX_FILLERS` constant |
| `data-plane/app/transport/base.py` | `interrupted: bool` property (default `False`) |
| `data-plane/app/transport/voice_transport.py` | Overrides `interrupted` to return `_tts_cancel.is_set()` |
| `data-plane/app/websocket/handler.py` | Filler dispatch inside `_process_utterance._compute` task |
| `control-plane/app/models/assistant.py` | `enable_empathy_fillers` DB column |
| `control-plane/app/schemas/assistant.py` | Field in `AssistantConfigBase` and `AssistantConfigUpdate` |
| `control-plane/app/services/config_export.py` | Included in config export payload |

---

## Tests

**`tests/test_empathy_fillers.py`** — unit and dispatch tests (no LLM, fast):

| Class | What it tests |
|---|---|
| `TestFillerSequence` | `filler_sequence()` correctness, pool membership, no consecutive repeats, infinite iteration |
| `TestTransportInterrupted` | `VoiceTransport.interrupted` is False initially; `TextTransport.interrupted` always False |
| `TestFillerDispatch` | Dispatch logic via async helper: flag disabled, fast compute, slow compute, primary-first, MAX_FILLERS cap, text mode skip, barge-in stops loop, compute always awaited on barge-in, unique filler IDs |

**`tests/simulate_empathy_fillers.py`** — LLM-level simulation tests (requires `CEREBRAS_API_KEY`):

| Test | What it proves |
|---|---|
| `test_data_collection_opening_triggers_filler` | Real 500ms threshold fires on a name utterance (~977ms elapsed) |
| `test_narrative_opening_triggers_filler` | Long narrative turn triggers fillers (~820ms elapsed) |
| `test_continuation_filler_fires_on_very_slow_turn` | At 100ms threshold, continuations drawn from `_CONTINUATION` pool |
| `test_filler_count_does_not_exceed_max` | Hard cap at `MAX_FILLERS=3` regardless of LLM time |
| `test_real_response_always_returned_despite_fillers` | LLM response always returned — fillers don't swallow it |
| `test_barge_in_mid_filler_discards_response` | Barge-in stops at 1 filler, compute still awaited, response discarded |

Run:

```bash
cd data-plane

# Unit + dispatch tests (no API key needed)
PYTHONPATH=. python3 -m pytest tests/test_empathy_fillers.py -v

# LLM simulation (requires CEREBRAS_API_KEY)
PYTHONPATH=. python3 -m pytest tests/simulate_empathy_fillers.py -v -s
```

---

## DB Migration

For existing control-plane databases, the new column must be added manually:

```sql
ALTER TABLE assistant_config
ADD COLUMN enable_empathy_fillers BOOLEAN NOT NULL DEFAULT 0;
```

New databases created via `create_all` will include the column automatically.
