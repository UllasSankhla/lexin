# Codelexin — AI Voice Booking Assistant

## Project Overview

Three-component system:
- `control-plane/` (port 8000) — FastAPI config management service
- `data-plane/` (port 8001) — FastAPI voice call handler (WebSocket + STT/LLM/TTS)
- `frontend/` — Vanilla JS embeddable widget (Shadow DOM, no dependencies)
- `evaluator/` — Offline transcript replay, scoring, and improvement suggestion engine

## Running the Services

```bash
# Control plane
cd control-plane && ./run.sh
# Seed with test data
python3 seed_data.py

# Data plane
cd data-plane && ./run.sh

# Frontend dev server
cd frontend && python3 -m http.server 5500
```

Required environment variables: `DEEPGRAM_API_KEY`, `CEREBRAS_API_KEY`, `CONTROL_PLANE_API_KEY`

---

## Tests

### Unit / integration tests (pytest)

```bash
cd data-plane
PYTHONPATH=. python3 -m pytest tests/test_scheduling_flow.py -v
PYTHONPATH=. python3 -m pytest tests/test_narrative_collection.py -v
PYTHONPATH=. python3 -m pytest tests/test_intake_qualification.py -v
PYTHONPATH=. python3 -m pytest tests/test_tools.py -v
```

Run all tests at once:
```bash
cd data-plane
PYTHONPATH=. python3 -m pytest tests/ -v
```

Tests use lightweight fakes for all external dependencies (Calendly, Cerebras, Deepgram) — no API keys needed.

### Connectivity / live API tests (require keys)

```bash
cd data-plane
PYTHONPATH=. python3 tests/test_stt_connection.py    # Deepgram STT connection
PYTHONPATH=. python3 tests/test_calendly.py          # Calendly API round-trip
```

---

## Simulations

Simulations run the full agent/workflow graph with scripted caller utterances — no WebSocket, no real audio, no API calls. Use these to verify agent routing, state machine transitions, and data collection logic end-to-end.

### Scenario simulations (pytest)

Each file in `data-plane/tests/simulate_*.py` is a named real-world scenario:

```bash
cd data-plane
PYTHONPATH=. python3 -m pytest tests/simulate_kavita_sharma.py -v -s
PYTHONPATH=. python3 -m pytest tests/simulate_abhay_sank.py -v -s
PYTHONPATH=. python3 -m pytest tests/simulate_data_collection.py -v -s
PYTHONPATH=. python3 -m pytest tests/simulate_fallback.py -v -s
PYTHONPATH=. python3 -m pytest tests/simulate_faq_interrupt.py -v -s
PYTHONPATH=. python3 -m pytest tests/simulate_farewell.py -v -s
PYTHONPATH=. python3 -m pytest tests/simulate_intake_qualification.py -v -s
PYTHONPATH=. python3 -m pytest tests/simulate_narrative_embedded_questions.py -v -s
```

Run all simulations at once:
```bash
cd data-plane
PYTHONPATH=. python3 -m pytest tests/simulate_*.py -v -s
```

### Workflow graph simulation (multi-scenario harness)

```bash
cd data-plane
PYTHONPATH=. python3 simulate.py
```

Runs 12 built-in scenarios: happy path, FAQ interrupts, fallback chains, not-qualified paths, ambiguous intents, phonetic name spelling, etc.

---

## Evaluator (Transcript Replay + Scoring)

The evaluator ingests real call transcripts, replays them through the live agent pipeline, scores AI responses turn-by-turn, clusters findings, and produces improvement suggestions.

```bash
cd evaluator

# Run against default transcripts/ directory
python3 run.py

# Run against a specific directory
python3 run.py /path/to/transcripts/

# Use a custom mock config
python3 run.py --config config/mock_config.py

# Verbose output
python3 run.py --verbose
```

Transcripts can be `.txt` or `.csv`. Reports are written to `evaluator/reports/` as JSON, Markdown, and HTML.

Pipeline: **Ingest → Replay → Evaluate → Synthesise → Report**

---

## Development Practices

### Keep changes small and targeted

- One logical change per commit. A fix for a data collection bug should not also clean up unrelated logging or refactor adjacent code.
- Every change should be testable in isolation — if you can't write a simulate script or unit test for it, the change scope is probably too large.
- Before touching shared infrastructure (transport, router, session), confirm whether a targeted agent-level fix is sufficient.

### Python async — streaming application rules

- **Never use `time.sleep()` in async code.** Always use `asyncio.sleep()`. Blocking sleeps inside coroutines stall the entire event loop and introduce invisible latency spikes.
- Prefer `asyncio.Queue` for producer/consumer handoff between coroutines. Do not share mutable state across tasks without a lock.
- Use `asyncio.gather()` for concurrent independent tasks; use `asyncio.wait()` when you need to handle first-completed or first-failed semantics.
- Always set explicit timeouts on external calls (`asyncio.wait_for`). An unawaited external call that hangs silently will stall a live call.
- Async generators (`async for`) are the correct model for streaming TTS chunks. Do not buffer the full response before sending to the client.
- Context propagation: pass session/call state as explicit arguments. Do not rely on `contextvars` or thread-locals across coroutine boundaries.

### Streaming application rules

- **Stream first-chunk, not full response.** For TTS: detect sentence boundaries in the LLM output stream and pipe each sentence to TTS immediately. Do not wait for the full LLM response.
- Audio pipeline ordering matters: STT final → LLM → TTS is the latency-critical path. Every millisecond of buffering compounds.
- Barge-in must be handled at the transport layer before the utterance reaches the agent. Cancellation events must propagate cleanly — do not leave orphaned TTS streams.
- Backpressure: if the utterance queue grows (caller speaking faster than pipeline can respond), older turns should be discardable without crashing state.

### Human speech — avoid brittle pattern matching

This is the most important design principle for this codebase:

**Do not solve speech understanding problems with regex or rigid conditionals.**

Human callers are unpredictable. They:
- Give multiple pieces of information in one sentence
- Use filler words, restart sentences, or correct themselves mid-utterance
- Spell names phonetically (NATO alphabet, "S as in Sam")
- Speak in accents, use colloquialisms, abbreviate, or trail off
- Deviate from the expected flow at any turn

Regex and keyword-matching approaches will always have edge cases that break on real calls. Prefer:
- **LLM extraction** over regex: use a focused LLM call (via `LLMToolkit`) to extract a structured value from a messy utterance rather than writing pattern rules
- **Semantic routing** over keyword dispatch: route intents through the LLM-based router, not switch statements on keywords
- **Parameterized prompts** over hardcoded conditionals: push variability into the prompt, not into branching Python logic
- **Field-level confidence and re-prompting** over strict validation rules: if an extracted value looks uncertain, have the AI ask the caller to confirm rather than rejecting it with a hard error

When you feel the urge to add a regex, ask: "Will this break on a caller who phrases this differently?" If yes — use the LLM instead.

### Scaling for variety and variability

Code that works for 10 call scenarios should work for 10,000 without modification. Design for:
- **New parameter types** without new extraction code — extraction_hints in config are the extension point
- **New agent behaviors** without touching the router — the workflow graph is the extension point
- **New languages/accents** without new parsing logic — Deepgram handles language, the LLM handles meaning
- **New validation rules** without new conditionals — push them into the prompt or config, not into `if/elif` chains

Every time a real call exposes a new edge case, the fix should generalize (update the prompt, improve the agent logic) rather than add another special case.

### Handling Bugs

When I report bugs, dont start by trying to fix them. Start by trying to reproduce them by looking at the transcript and writing a sim or test that tries to reproduce that. If there is no transcript - just write a test for reproducing that first. And then look at the logs to make sure that we are capturing any more details. Only when you have reproduced the bug and examined the logs, write a potential fix and check before going all the way to commiting the fix. Prove that the fix works by running the test case and then - running the test suite (ask before running the entire suite).

### Testing new behavior

When adding or fixing agent behavior:
1. Write a `simulate_<scenario>.py` in `data-plane/tests/` that reproduces the real-world utterance sequence
2. Confirm it fails before your fix and passes after
3. Run the full simulation suite to check for regressions: `PYTHONPATH=. python3 -m pytest tests/simulate_*.py -v -s`
4. If the fix touches extraction or LLM prompts, also run through the evaluator against recent transcripts

Real transcripts are the ground truth. If a simulate script passes but the evaluator shows regression on real calls, the simulate script is missing context — extend it.
