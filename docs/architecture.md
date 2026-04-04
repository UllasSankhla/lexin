# Codelexin — Architecture

## System Overview

Codelexin is a real-time AI voice booking assistant. A caller speaks to an embeddable widget, their audio is transcribed, routed through a multi-agent workflow, and responses are synthesized back as speech. At the end of the call a booking is confirmed on the firm's calendar and a webhook fires with the collected data.

Three independently deployable services:

```
┌─────────────────────┐     REST (config export)     ┌─────────────────────┐
│   Control Plane     │ ◄──────────────────────────► │    Data Plane       │
│   (port 8000)       │                              │    (port 8001)      │
│                     │                              │                     │
│  FastAPI + SQLite   │                              │  FastAPI + SQLite   │
│  Config management  │                              │  Call handler       │
│  Admin API          │                              │  WebSocket + STT    │
└─────────────────────┘                              │  LLM + TTS          │
                                                     └────────┬────────────┘
                                                              │ WebSocket
                                                     ┌────────▼────────────┐
                                                     │    Frontend         │
                                                     │  (port 5500 dev)    │
                                                     │  Embeddable widget  │
                                                     │  Shadow DOM / JS    │
                                                     └─────────────────────┘
```

---

## Control Plane

**Purpose:** Manages all per-tenant configuration. The data plane fetches this config at call start and caches it.

**Key models (SQLite):**

| Model | Purpose |
|---|---|
| `AssistantConfig` | Persona name, voice, language, system prompt, `enable_empathy_fillers` |
| `CollectionParameter` | Fields to collect (name, type, required, hints, order) |
| `FAQ` | Curated question/answer pairs for the FAQ agent |
| `ContextFile` | Uploaded business documents for the ContextDocs agent |
| `SpellRule` | STT correction substitutions (e.g. "acme" → "ACME Corp") |
| `WebhookEndpoint` | POST targets for post-call data delivery |
| `CalendlyConfig` | API token + event type URI for scheduling |
| `PracticeArea` | Practice areas used by intake qualification |
| `PolicyDocument` | Policy docs injected into the data collection prompt |

**Config export:** `GET /api/v1/config/export` assembles all of the above into a single JSON payload consumed by the data plane. The data plane caches this per owner and re-fetches on each new call initiation.

---

## Data Plane

The core of the system. Handles the full real-time call lifecycle.

### Entry Points

```
POST /api/v1/calls/initiate          — Creates call record, fetches config, returns session_token
GET  /ws/call?token=<session_token>  — Upgrades to WebSocket, starts call
GET  /api/v1/calls/                  — List calls (admin)
GET  /api/v1/calls/{id}/transcript   — Retrieve saved transcript
```

### Call Lifecycle

```
Client                     Data Plane                    External
  │                            │                            │
  │  POST /calls/initiate       │                            │
  │ ──────────────────────────► │                            │
  │                            │── fetch config ──────────► │ Control Plane
  │  { session_token }         │ ◄─────────────────────────  │
  │ ◄────────────────────────  │                            │
  │                            │                            │
  │  WS /ws/call?token=...     │                            │
  │ ──────────────────────────► │                            │
  │                            │── open STT session ──────► │ Deepgram
  │  [binary PCM audio]        │                            │
  │ ──────────────────────────► │                            │
  │                            │── stream transcript ──────► │
  │  [binary PCM audio chunks] │ ◄── final utterance ───────  │
  │ ◄────────────────────────  │                            │
  │  [JSON events]             │── LLM call ─────────────► │ Cerebras
  │ ◄────────────────────────  │ ◄────────────────────────  │
  │                            │── TTS synthesis ────────► │ Deepgram
  │                            │ ◄────────────────────────  │
  │  [binary PCM audio]        │                            │
  │ ◄────────────────────────  │                            │
  │                            │                            │
  │  [call ends]               │── POST booking ──────────► │ Calendly
  │                            │── POST webhook ──────────► │ Customer endpoint
  │                            │── save transcript ────────► │ Local filesystem
```

### Transport Layer

Two transport modes behind a common `BaseTransport` interface:

- **VoiceTransport** — PCM audio in (Deepgram STT) + PCM audio out (Deepgram TTS). Handles barge-in: cancels TTS stream when interim transcripts arrive while audio is playing. Exposes `interrupted: bool` so the filler dispatcher can detect mid-filler barge-in.
- **TextTransport** — JSON text frames in/out. Used for testing and text-mode widget integrations. `interrupted` always returns `False` — fillers are skipped in text mode.

### WebSocket Frame Protocol

| Direction | Frame type | Content |
|---|---|---|
| Client → Server | Binary | Raw PCM audio (16-bit, 16 kHz) |
| Server → Client | Binary | `[4-byte id length][id bytes][PCM bytes]` — prefixed audio chunks |
| Server → Client | Text | JSON: `{ type, seq, ts, payload }` |

Key server→client message types: `server.thinking`, `server.transcript_interim`, `server.transcript_final`, `server.parameter_collected`, `server.booking_confirmed`.

---

## Agent System

The heart of the data plane. Every caller utterance is classified by the **Planner**, which produces an ordered execution plan. The **Handler** executes the plan steps, follows graph edges, and assembles the response.

### Planner (intent classification + routing)

```
Utterance
    │
    ▼
┌─────────────────────────────────────────────────┐
│  Planner.plan()                                 │
│                                                 │
│  LLM classifies utterance into ordered intents: │
│   FIELD_DATA | CONFIRMATION | CORRECTION        │
│   NARRATIVE  | FAQ_QUESTION | DATA_STATUS       │
│   CONTINUATION | FAREWELL                       │
│                                                 │
│  Python maps intents → PlanSteps deterministically │
│   (no LLM in routing decisions)                 │
│                                                 │
│  _validate_waiting_confirm: ensures any active  │
│   WAITING_CONFIRM node is always in the plan    │
└──────────────────┬──────────────────────────────┘
                   │ list[PlanStep]
                   ▼
            Handler executes steps
```

The Planner replaces the earlier single-agent Router. Key difference: multi-intent utterances (e.g. "My name is John and I was in an accident") produce multiple steps executed concurrently, with results joined in speech order.

### Workflow Graph

A static DAG of `ActivityNode`s defined in `graph_config.py`. Each node maps to an agent class and declares its own transition edges.

```
                    ┌─────────────┐
                    │data_collection│  goal_order=1
                    │  (PRIMARY)  │
                    └──────┬──────┘
                    COMPLETED │
                           ▼
                  ┌──────────────────┐
                  │narrative_collection│ goal_order=2
                  │    (PRIMARY)     │
                  └────────┬─────────┘
                  COMPLETED │
                           ▼
                 ┌────────────────────┐
                 │intake_qualification│  goal_order=3  [auto-run via edge]
                 │    (PRIMARY)       │
                 └────────┬───────────┘
               COMPLETED  │    FAILED
                  ┌───────┘        └──────────────────┐
                  ▼                                    ▼
           ┌──────────┐                         end(not_qualified)
           │scheduling│  goal_order=4
           │(PRIMARY) │
           └────┬─────┘
       COMPLETED│
                ▼
         end(completed)

Interrupt agents (can fire at any turn, resume stack returns to primary):
  farewell ──────────────────────────────► end(caller_farewell)
  faq ──► [on_complete] resume | [on_failed] context_docs
  context_docs ──► [on_complete] resume | [on_failed] fallback
  fallback ──► resume

One-shot agents (non-primary, non-interrupt):
  empathy ──► decider  (fires once on first NARRATIVE, then COMPLETED)
```

**Node statuses:** `NOT_STARTED → IN_PROGRESS → WAITING_CONFIRM ↔ IN_PROGRESS → COMPLETED / FAILED`

**Resume stack:** When an interrupt-eligible agent fires mid-primary-goal, the active primary is pushed onto the Planner's resume stack. When the interrupt completes (`on_complete → resume`), the primary is popped and re-invoked with an empty utterance to re-surface its pending state.

### Agents

| Agent | Class | Role |
|---|---|---|
| `data_collection` | `DataCollectionAgent` | Collects required/optional fields one at a time with LLM extraction and confirmation. Handles corrections, phonetic spelling, multi-field utterances. |
| `narrative_collection` | `NarrativeCollectionAgent` | Listens to the caller describe their legal matter across one or more turns. Signals COMPLETED when enough narrative is captured. |
| `intake_qualification` | `IntakeQualificationAgent` | LLM assesses whether the narrative falls within configured practice areas. Returns COMPLETED (qualified) or FAILED (not qualified). |
| `scheduling` | `SchedulingAgent` | Fetches slots from Calendly, presents them by voice, takes choice, confirms, books. Internal stage machine: `presenting → awaiting_choice → awaiting_confirm → done`. |
| `empathy` | `EmpathyAgent` | One-shot. Fires on first NARRATIVE, produces a brief contextually-specific acknowledgment. Always returns COMPLETED. |
| `faq` | `FAQAgent` | Matches caller question against curated FAQ pairs. Returns UNHANDLED if no match. |
| `context_docs` | `ContextDocsAgent` | Answers from uploaded business context files. Fallback for FAQ misses. |
| `fallback` | `FallbackAgent` | Acknowledges unanswerable questions, records a note. Last resort in the interrupt chain. |
| `farewell` | `FarewellAgent` | Detects goodbye, produces a polite closing, edges to `end(caller_farewell)`. |

All agents implement `AgentBase.process(utterance, internal_state, config, history) → SubagentResponse`. Agents are **stateless between calls** — all per-turn state is passed in via `internal_state` and returned via `SubagentResponse.internal_state`.

### Tools (non-conversational async work)

Tools run at lifecycle trigger points. They do not speak to the caller.

| Tool | Trigger | Mode |
|---|---|---|
| `CalendarPrefetchTool` | `AGENT_COMPLETE` on `narrative_collection` | `fire_and_forget=False`, awaited before `scheduling` — overlaps with `intake_qualification` LLM call |
| `SummarizationTool` | `CALL_END` | `fire_and_forget=True` — runs in background after call |
| `WebhookTool` | `CALL_END` | `fire_and_forget=True` — runs after SummarizationTool so it can read `ai_summary` |

### Empathy Filter

A post-processing step applied to every agent response before it reaches TTS. Strips openers like "Absolutely!" and "Of course!" that accumulate from LLM responses. Prevents the assistant from sounding formulaic across turns.

### Empathy Fillers

A perceived-latency reduction mechanism. When `enable_empathy_fillers` is enabled in the assistant config and a turn takes longer than **500 ms** to compute, a short generic phrase ("Sure, one moment.", "Thanks for your patience.") is streamed to TTS in parallel with the ongoing LLM work. The real response is queued immediately after the filler audio.

```
t=0ms:   _compute() starts (planner + agents)
t=0ms:   asyncio.wait_for(shield(compute_task), timeout=0.5s)
t=500ms: timeout → filler_1 sent to TTS
t=~1.5s: filler_1 drains → compute done? → real response queued
```

Up to `MAX_FILLERS = 3` phrases are sent. Barge-in during a filler cancels the filler stream; the compute task is still awaited for graph state consistency but the response is discarded. Fillers never fire in text mode. See `docs/empathy-fillers.md` for full detail.

---

## LLM Integration

All LLM calls go through Cerebras (`llama-4-scout-17b-16e-instruct`) via two shared utilities:

- **`llm_text_call`** — Returns raw text. Used for TTS-bound responses (slots, summaries, empathy).
- **`llm_structured_call`** — Returns a validated Pydantic model. Used for extraction, routing, and classification (planner intents, router decisions, confirmation signals, slot choices).

All calls use exponential backoff retry (6 attempts, 25 ms → 800 ms). Client errors (4xx) are not retried.

---

## Data Layer

Each service has its own SQLite database (WAL mode for concurrent reads).

**Control plane DB:**

```
assistant_config      — persona settings, system prompt, voice
collection_parameter  — fields to collect, types, hints, order
faq                   — question/answer pairs
context_file          — uploaded docs with content
policy_document       — policy docs injected into collection prompt
spell_rule            — STT correction substitutions
webhook_endpoint      — POST targets
calendly_config       — API token and base event type
calendly_event_type   — available event types
practice_area         — qualifying practice areas
customer_key          — API keys for widget authentication
```

**Data plane DB:**

```
call_record           — UUID PK, state, timing, booking_id, transcript_path
gathered_parameter    — per-call collected field values
call_analytics        — per-stage timing events (stt_open, llm_call, etc.)
```

Transcripts and context files are stored on the local filesystem.

---

## Frontend Widget

Self-contained embeddable JS (`booking-widget.js`), no dependencies.

- Mounted in Shadow DOM — styles are fully isolated from the host page.
- Initializes via a `<script>` tag with `data-api-url` and `data-customer-key` attributes.
- Manages the full call lifecycle: `POST /calls/initiate` → WebSocket connect → audio capture (Web Audio API) → PCM streaming → receive audio chunks → playback.
- Handles barge-in client-side: tracks which audio chunk is playing; discards queued chunks when a new agent response arrives.

---

## Call Flow — End to End

```
1. Widget loads on page
        │
        ▼
2. User clicks "Call"
        │
        ▼
3. POST /calls/initiate
   ├── Authenticates customer key
   ├── Fetches config from control plane (cached per owner)
   ├── Creates CallRecord in DB
   └── Returns { call_id, session_token }
        │
        ▼
4. WebSocket /ws/call?token=...
   ├── CallSession created (config, state_machine, graph, planner, registry)
   ├── STT session opened (Deepgram nova-2)
   ├── Greeting generated via LLM + sent as TTS
   └── Two concurrent tasks start:
       ├── _ws_read_loop    — feeds PCM frames to STT
       └── _utterance_processor — consumes final transcripts one at a time
        │
        ▼
5. Each caller utterance → _process_utterance()
   ├── _compute() launched as asyncio.Task
   │     ├── planner.plan(utterance)        — intent classification (LLM)
   │     ├── reset_fields steps applied     — clears fields for corrections
   │     ├── invoke steps run concurrently  — agents process utterance
   │     ├── WAITING_CONFIRM agent wins     — if any step triggers confirmation
   │     ├── speaks assembled in intent order via combine_speaks()
   │     └── empathy_filter applied         — strips formulaic openers
   ├── [if enable_empathy_fillers] wait 500ms — if _compute() still running, send fillers
   ├── TTS stream → PCM chunks → WebSocket  (filler phrases first, then real response)
   ├── collected fields persisted to DB
   └── graph edges followed (→ decider | → resume | → end | → next node)
        │
        ▼
6. Call ends (farewell / booking complete / not qualified / error)
   ├── CallRecord updated (state, duration, booking_id)
   ├── Transcript saved to filesystem
   ├── SummarizationTool runs (background) — generates ai_summary
   ├── WebhookTool fires (background)      — POST to configured endpoints
   └── WebSocket closed
```

---

## Key Design Decisions

**Planner over Router:** The original Router selected one agent per turn. The Planner classifies all intents in speech order and produces multi-step plans, enabling multi-intent utterances to be handled correctly in a single turn.

**LLM classifies, Python routes:** The planner LLM identifies *what* the caller meant. Python deterministically maps that to agent invocations. Routing logic does not live in the LLM — it lives in `_build_steps_from_intents`. This makes routing testable and auditable.

**Stateless agents:** Agents receive their prior state as a dict and return a new state dict. There is no shared mutable state between turns at the agent level. The `WorkflowGraph` holds all durable state.

**Tool/agent separation:** Non-conversational work (calendar fetch, summarization, webhook) is fully separated from conversational agents. Tools cannot speak, agents cannot make external API calls directly. The boundary is enforced by design.

**Two STT/TTS pairs per call:** Deepgram provides both STT (nova-2, streaming) and TTS (aura-2, per-sentence). The TTS stream starts as soon as the first sentence is available — the full LLM response is not buffered before synthesis begins.

**Empathy fillers mask perceived latency:** Rather than reducing actual LLM latency, the filler system attacks perceived latency — the silence window the caller experiences. A short phrase streamed at the 500 ms mark gives the caller immediate auditory feedback while the real response finishes computing. This is toggled per-assistant via `enable_empathy_fillers`.
