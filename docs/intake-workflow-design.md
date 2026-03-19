# Intake Workflow Design
## Voice AI Receptionist — Agent Architecture and Design Decisions

*Document version: 2026-03-19*

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Workflow Engine Architecture](#2-workflow-engine-architecture)
3. [The 4-Goal Intake Sequence](#3-the-4-goal-intake-sequence)
4. [Agent Reference](#4-agent-reference)
5. [DataCollectionAgent vs NarrativeCollectionAgent](#5-datacollectionagent-vs-narrativecollectionagent)
6. [Router and Invoker](#6-router-and-invoker)
7. [Key Design Decisions](#7-key-design-decisions)
8. [Simulation Coverage](#8-simulation-coverage)
9. [Open Items](#9-open-items)

---

## 1. System Overview

The voice AI receptionist is built as a three-tier service:

| Component | Port | Role |
|---|---|---|
| `control-plane/` | 8000 | FastAPI config management — assistants, parameters, FAQs, context files, webhooks |
| `data-plane/` | 8001 | FastAPI voice call handler — WebSocket, STT (Deepgram nova-2), LLM (Cerebras llama-4-scout), TTS (Deepgram aura-2) |
| `frontend/` | 5500 | Vanilla JS embeddable widget, Shadow DOM, no dependencies |

A caller connects over WebSocket. Audio frames stream in as PCM; the data-plane runs STT in real time. Each completed utterance is passed to the **workflow engine**, which selects the right agent, invokes it, follows declared edges, and streams a TTS response back to the caller.

---

## 2. Workflow Engine Architecture

### 2.1 Design Principle

The data-plane previously encoded all orchestration logic imperatively in `handler.py`: the fallback chain (`faq → context_docs → fallback`) was a hardcoded sequence, the resume stack was bespoke code, and goal continuation had two separate branches that could accidentally double-fire.

The core insight driving the redesign: **the invoker should follow edges mechanically; all workflow logic should live in a declarative `WorkflowDefinition`**. This means adding a new goal, changing the fallback order, or wiring a qualification step requires editing the graph definition — not `handler.py`.

### 2.2 Schema

```python
@dataclass
class Edge:
    target: str
    # "<node_id>" — invoke that node immediately (same turn, no decider)
    # "decider"   — speak response, wait for next utterance, consult decider
    # "resume"    — pop resume stack; invoke primary with empty utterance
    # "end"       — finalize the call
    reason: str = ""   # surfaced in finalization log and analytics


@dataclass
class ActivityNode:
    id: str
    agent_class: str              # maps to a registered AgentBase
    description: str              # injected into decider prompt
    is_primary_goal: bool = False # explicitly marks goal-critical nodes
    goal_order: int | None = None # execution order among primary goals (1, 2, ...)
    interrupt_eligible: bool = False
    auto_run: bool = False        # invoked automatically, never by decider
    depends_on: list[str] = []

    # Declared transitions
    on_complete:        Edge = Edge("decider")
    on_failed:          Edge = Edge("decider")
    on_continue:        Edge = Edge("decider")  # covers IN_PROGRESS
    on_waiting_confirm: Edge = Edge("decider")  # covers WAITING_CONFIRM


@dataclass
class GoalSpec:
    primary_agents: list[str]   # ordered node ids — all must COMPLETE for success
    description: str            # injected into decider prompt as the mission statement


@dataclass
class DeciderSpec:
    system_prompt: str          # routing rules; dynamic sections injected by invoker
    max_tokens: int = 1024


@dataclass
class InterruptPolicy:
    eligible_agents: list[str]
    resume_strategy: str = "stack"


@dataclass
class ErrorPolicy:
    max_consecutive_errors: int = 3
    transient_error_speak: str
    on_max_errors: Edge = Edge("end", "agent_error")


@dataclass
class WorkflowDefinition:
    id: str
    description: str
    goal: GoalSpec
    decider: DeciderSpec
    nodes: list[ActivityNode]
    interrupt_policy: InterruptPolicy
    error_policy: ErrorPolicy
```

`WorkflowDefinition.__post_init__` validates the entire graph at construction time: all `depends_on` references, all `on_complete/on_failed/on_continue` edge targets, all `goal.primary_agents`, all `interrupt_policy.eligible_agents`, and uniqueness of `goal_order` values. A misconfigured graph fails fast at startup.

### 2.3 Invoker Algorithm

The invoker (`_invoke_and_follow`) follows edges recursively within a single turn:

```
INVOKE(agent_id, utterance):
    response = agent.process(utterance, state, config, history)
    graph.update(agent_id, response)
    edge = on_complete | on_failed | on_continue | on_waiting_confirm
           (selected by response.status)

    FOLLOW(edge):
      "<node_id>" → INVOKE(node_id, "")          # chain immediately, same turn
      "resume"    → INVOKE(resume_stack.pop(), "")# re-invoke primary agent
      "decider"   → speak response, wait for next utterance
      "end"       → finalize(edge.reason), EXIT

ROUTE (per turn):
    if any node is WAITING_CONFIRM → enforce that agent (bypass LLM)
    else → LLM decider selects agent_id, interrupt flag
    if interrupt → push active_primary() onto resume_stack
    INVOKE(agent_id, utterance)
```

Key property: **the invoker never makes routing decisions**. It follows edges. The decider makes routing decisions. Workflow logic is expressed entirely in `WorkflowDefinition` data.

A `chain_depth` counter (max 8) guards against cycles in the edge graph.

### 2.4 WAITING_CONFIRM Enforcement

`WAITING_CONFIRM` is a distinct agent status meaning "I've read back a value and need a yes or no." It is not treated as `IN_PROGRESS` — the next utterance **must** go back to the same agent, regardless of what the caller says (a question like "what's your fee?" during a name confirmation should not route to FAQ).

The router enforces this as a hard pre-check **before** the LLM call. The `on_waiting_confirm` edge defaults to `Edge("decider")`, which means: speak the confirmation question, wait for the next utterance, and the hard enforcement at the top of the turn handles routing it back.

### 2.5 Re-invokable Interrupt Agents

FAQ, context_docs, and fallback are `interrupt_eligible=True`. A caller may ask multiple questions throughout the conversation — once about fees, once about location, once about cancellation policy. The `available_nodes()` filter allows these agents regardless of their COMPLETED or FAILED status, so they can be invoked multiple times. `AgentState.invocation_count` tracks how many times each interrupt agent has been called during the conversation.

---

## 3. The 4-Goal Intake Sequence

The appointment booking workflow defines four primary goals that must all complete in order:

```
data_collection
    │ on_complete
    ▼
narrative_collection
    │ on_complete
    ▼
intake_qualification  ──── on_failed ───► end("not_qualified")
    │ on_complete
    ▼
scheduling
    │ on_complete
    ▼
end("completed")
```

`intake_qualification` runs immediately via direct edge chain from `narrative_collection.on_complete` — no caller utterance is needed. It is not `auto_run=True` (which is reserved for post-processing like webhook); it runs as part of the same chain.

At any point during `data_collection` or `narrative_collection`, the caller may ask a question. The decider routes to an interrupt agent (faq → context_docs → fallback if each fails). The primary agent is pushed onto the resume stack. When the interrupt chain completes (`on_complete=Edge("resume")`), the primary is re-invoked with an empty utterance to re-present its pending question.

### 3.1 Full Node Definitions (current graph)

```
faq             interrupt, on_complete→resume, on_failed→context_docs
context_docs    interrupt, on_complete→resume, on_failed→fallback
fallback        interrupt, on_complete→resume, on_failed→resume

data_collection      primary goal 1, on_complete→narrative_collection, on_failed→end
narrative_collection primary goal 2, on_complete→intake_qualification, on_failed→end
intake_qualification primary goal 3, on_complete→scheduling, on_failed→end("not_qualified")
scheduling           primary goal 4, on_complete→end("completed"), on_failed→end

webhook         auto_run, depends_on=[scheduling], on_complete→end
```

---

## 4. Agent Reference

### 4.1 DataCollectionAgent

**Purpose:** Collect structured fields from the caller one at a time with confirmation.

**Internal state:**
| Key | Type | Description |
|---|---|---|
| `stage` | `"asking"` \| `"waiting_confirm"` | Current sub-state |
| `current_field` | `str` | Field currently being collected |
| `pending_value` | `str \| None` | Extracted value awaiting yes/no |
| `collected` | `dict` | All confirmed field values |
| `retry_count` | `int` | Extraction retries for current field |
| `field_index` | `int` | Position in parameter list |

**Per-field flow:**
1. **Ask** — LLM-generated voice question tailored to field type
2. **Extract** — LLM extracts value and `refused` flag in a single call
3. **Validate** — regex pattern (built-in or configured)
4. **Spell back** — LLM-generated read-back phrase → `WAITING_CONFIRM`
5. **Confirm** — LLM detects yes / no / unclear (with embedded correction on "no")
6. **Advance** — if more fields remain, repeat; else `COMPLETED`

**Optional field handling:**
- If `refused=true` → skip (LLM detects "I don't have one", "skip", "not applicable")
- If extraction fails after `MAX_RETRIES=3` → skip optional, fail required

**LLM calls per field:** 3–5 (ask, extract+refused, spell_back, confirm, optional correction detect)

**Parameters** are loaded from `config["parameters"]` — a list of dicts with `name`, `display_label`, `data_type`, `required`, `validation_regex`, `validation_message`. This list is assembled by the control-plane config export and includes firm-specific fields (name, email, phone) as well as domain fields like MetLife ID for insurance-adjacent matters.

---

### 4.2 NarrativeCollectionAgent

**Purpose:** Listen to the caller describe their legal matter in free-form speech across multiple turns.

**Internal state:**
| Key | Type | Description |
|---|---|---|
| `stage` | `"collecting"` \| `"asking_done"` | Current sub-state |
| `segments` | `list[str]` | All caller utterances accumulated — never reset |
| `summary` | `str \| None` | Populated on COMPLETED |
| `case_type` | `str \| None` | Populated on COMPLETED |

**Flow:**
1. **Open** (utterance="") → speak a prompt: "Please describe your legal matter in your own words."
2. **Collect** — each caller utterance appended to `segments`; respond with a filler phrase
3. **Check completion** — once at least `_MIN_SEGMENTS=2` segments with `≥4` words each exist, run an LLM completion check against the last two segments
4. **Ask done** — if the LLM says the narrative seems complete, ask "Is there anything else you'd like to add?"
5. **Detect intent** — LLM classifies the caller's response as done or not-done
   - Not done → append utterance, return to collecting
   - Done → `_complete()`
6. **Complete** — LLM summarises the full narrative in 2–3 sentences and identifies the best-matching practice area from `config["practice_areas"]`; returns `COMPLETED`

**Interrupt handling:** `narrative_collection` is NOT `interrupt_eligible`, so it IS `active_primary()` while running. When the router sees a question mid-narrative, it routes to the interrupt chain with `interrupt=True`, pushing `narrative_collection` onto the resume stack. After the interrupt chain resolves (`on_complete=Edge("resume")`), `narrative_collection` is re-invoked with `utterance=""` and re-prompts the caller to continue. Because `segments` lives in `internal_state` (which is persisted in `AgentState.internal_state` in the graph), all utterances from before and after interruptions are concatenated in the final output.

**LLM calls:**
| Call | Tokens | Frequency |
|---|---|---|
| Completion check | 64 | Once per segment after MIN_SEGMENTS |
| Done intent detect | 32 | Once when in `asking_done` stage |
| Narrative summarise + case_type | 512 | Once on COMPLETED |
| Speak generation (completion) | 256 | Once on COMPLETED |

**Collected output:**
```python
{
    "narrative_summary": "2-3 sentence summary for intake coordinator",
    "case_type":         "best-matching practice area or 'unknown'",
    "full_narrative":    "raw concatenation of all segments",
}
```

---

### 4.3 IntakeQualificationAgent

**Purpose:** Decide whether the caller's matter falls within the firm's practice areas. One-shot, no caller interaction required.

**Invocation:** Called immediately via edge chain (`narrative_collection.on_complete → Edge("intake_qualification")`). The `utterance` argument is always `""` — the agent ignores it and reads from `history` and `config` instead.

**Decision logic:**
| Decision | Condition | Status returned | Edge followed |
|---|---|---|---|
| `qualified` | Case type clearly matches a practice area | `COMPLETED` | `on_complete → scheduling` |
| `ambiguous` | Case type is `"unknown"` or loosely related | `COMPLETED` | `on_complete → scheduling` |
| `not_qualified` | Case type clearly outside all practice areas | `FAILED` | `on_failed → end("not_qualified")` |

Both `qualified` and `ambiguous` return `COMPLETED` and proceed to scheduling. The rationale: it is better to schedule and let a human lawyer assess a borderline case than to reject a potential client on an AI's guess. If the LLM call itself fails, the default is `ambiguous` — never silently rejects.

**Speak behaviour:**
- `qualified` → warm acknowledgement, proceeding to schedule
- `ambiguous` → warm acknowledgement, noting a specialist will assess at the consultation
- `not_qualified` → empathetic decline, suggests seeking a specialist firm (two sentences maximum, not dismissive)

**Collected output:**
```python
{
    "qualification_decision": "qualified" | "ambiguous" | "not_qualified",
    "qualification_reason":   "one sentence LLM rationale",
}
```

**Internal state:** `decision` and `reason` fields — idempotent on re-invocation.

---

### 4.4 SchedulingAgent

**Purpose:** Present available Calendly slots, accept the caller's choice, confirm, and book.

Presents slots verbally (e.g. "I have Monday at 2pm and Tuesday at 3pm — which works for you?"), extracts the choice, spell-backs the slot (`WAITING_CONFIRM`), and on confirmation fires the Calendly booking API. Returns `COMPLETED` with a `booking` dict on success.

---

### 4.5 Interrupt Agents (FAQ, ContextDocs, Fallback)

**FAQAgent:** Uses an LLM to match the caller's question against the firm's curated FAQ list. If a match is found, rephrases the answer in voice-appropriate language. Returns `COMPLETED` (answered) or `FAILED` (no match). `FAILED` edges to `context_docs`.

**ContextDocsAgent:** Answers general questions from uploaded business documents using RAG-style LLM synthesis. Returns `COMPLETED` or `FAILED`. `FAILED` edges to `fallback`.

**FallbackAgent:** Acknowledges that the question cannot be answered right now, promises follow-up, and appends a note to the call record. Always returns `COMPLETED` (even a note-taking failure returns `COMPLETED` to resume the primary goal). `COMPLETED` and `FAILED` both edge to `resume`.

---

### 4.6 WebhookAgent

**Purpose:** Fire configured POST endpoints after the call completes. `auto_run=True`, `depends_on=["scheduling"]`. Invoked by the invoker's auto-run scan after scheduling COMPLETES. Does not interact with the caller.

---

## 5. DataCollectionAgent vs NarrativeCollectionAgent

These two agents both collect information from the caller, but they represent fundamentally different interaction models.

| Dimension | DataCollectionAgent | NarrativeCollectionAgent |
|---|---|---|
| **Interaction model** | Question-and-answer — one field, one turn | Free-form listening — many turns, no questions |
| **Structure of input** | Discrete, typed values (name, email, date) | Unstructured prose describing a situation |
| **Agent posture** | Active (drives the conversation) | Passive (listens, encourages, accumulates) |
| **Turn budget** | 2–4 turns per field (ask, extract, confirm) | Unbounded — caller stops when they feel done |
| **LLM role in collection** | Extract a specific field + validate | Detect when narrative is "complete enough" |
| **Confirmation** | Explicit per-field spell-back + yes/no | None during collection; "Is there anything else?" gate at the end |
| **State** | `stage`, `current_field`, `pending_value`, `collected`, `retry_count` | `stage`, `segments` (growing list), `summary`, `case_type` |
| **Output** | Structured key-value dict (`{"name": "John Smith", "email": "..."}`) | Prose summary, detected case type, and raw full narrative |
| **Failure mode** | Field extraction fails → retry → skip optional | Completion check fails → keep collecting (conservative default) |
| **Interrupt behaviour** | Active primary — paused to stack on interrupt, resumed with re-ask | Active primary — paused to stack on interrupt, resumed with re-prompt |
| **interrupt_eligible** | `False` | `False` |
| **Completion trigger** | All fields collected (or skipped) | LLM says narrative is complete + caller confirms done |
| **Config dependency** | `config["parameters"]` list | `config["practice_areas"]` list |

### Why two agents instead of one?

Merging them into a single "collect everything" agent was considered. The case against it: the interaction models are incompatible. DataCollectionAgent is a structured interviewer — it knows exactly what to ask next and validates each answer. NarrativeCollectionAgent is a listener — it has no questions to ask, only fillers to encourage the caller to keep speaking. Combining them would require a complex state machine to toggle between modes, and the distinct edge chains (DC → NC → IQ) would be hidden inside a monolith rather than visible in the graph definition.

### Where MetLife ID lives

The MetLife ID (insurance member ID) was considered as a candidate for `narrative_collection` because callers may mention it as part of describing their matter. **Decision: it stays in `data_collection`.**

Rationale: MetLife ID is a structured, validatable value (member ID pattern). Collecting it in `data_collection` means it goes through extract → validate → spell-back → confirm, which is the right treatment for a value that will be used programmatically. If the caller volunteers it during the narrative, it will appear in `full_narrative` and the intake coordinator can note it, but the agent won't attempt extraction from free-form prose.

---

## 6. Router and Invoker

### 6.1 Router

`Router` wraps `WorkflowGraph` and `Router._resume_stack`. Its `select()` method:

1. **Hard enforce** — if any node is `WAITING_CONFIRM`, return that node immediately (no LLM)
2. **Build prompt** — assembles the LLM user message from:
   - `workflow.goal.description` (the mission statement)
   - Recent conversation history (last 8 turns)
   - `graph.status_summary()` — each node's status, current field/pending value, invocation count
   - `graph.available_summary()` — only nodes with deps met, not auto_run, not COMPLETED/FAILED (except interrupt agents)
   - The caller's utterance
3. **LLM call** — Cerebras llama-4-scout-17b; returns `{agent_id, interrupt, reasoning}`
4. **Validate** — if `agent_id` not in available set, fall back to priority list
5. **Stack** — if `interrupt=True`, push `active_primary()` onto `_resume_stack`

Fallback priority when LLM fails or returns unavailable agent: `data_collection → scheduling → faq → fallback`.

### 6.2 Invoker

`_invoke_and_follow` in `handler.py` (and mirrored in `simulate.py`):

- Calls `agent.process(utterance, internal_state, config, history)`
- Calls `graph.update(agent_id, response, turn)`
- Calls `graph.get_edge(agent_id, response.status)` → `Edge`
- Follows edge: `"<node_id>"` → recursive call with `utterance=""`; `"resume"` → pop stack, recursive; `"decider"` → return; `"end"` → finalize
- Concatenates `speak` across chained nodes: `dc_speak + " " + nc_speak + ...`

### 6.3 LLM Calls Per Turn (worst case)

A single caller turn can involve multiple LLM calls via edge chaining. In the scenario where DC completes, NC opens, IQ evaluates, and scheduling opens, all in one chain:

| Call | Agent | Tokens |
|---|---|---|
| Router decider | Router | 1024 |
| Extraction | DataCollectionAgent | 512 |
| Spell-back | DataCollectionAgent | 256 |
| Narrative completion check | NarrativeCollectionAgent | 64 |
| Narrative done intent | NarrativeCollectionAgent | 32 |
| Narrative summarise | NarrativeCollectionAgent | 512 |
| Qualification | IntakeQualificationAgent | 128 |
| Qualification speak | IntakeQualificationAgent | 256 |
| Scheduling speak | SchedulingAgent | varies |

Most turns involve only 1–3 LLM calls. Chaining turns produce the highest call counts but happen at most once per conversation.

---

## 7. Key Design Decisions

### D1 — Declarative edges replace imperative orchestration

**Context:** The original `handler.py` had ~150 lines of orchestration logic: hardcoded fallback chains, a bespoke resume stack with a `requires_router_resume` flag on `SubagentResponse`, and two separate goal-continuation branches that could double-fire.

**Decision:** Replace with a `WorkflowDefinition` schema where every transition is a declared edge on a node. The invoker follows edges mechanically — it contains no workflow-specific logic.

**Effect:** Adding a new goal, reordering the fallback chain, or inserting a qualification step requires editing `graph_config.py` only. The invoker (`_invoke_and_follow`) is completely general.

---

### D2 — `requires_router_resume` removed from SubagentResponse

**Context:** Interrupt agents (faq, context_docs, fallback) were setting `requires_router_resume=True` on their responses to signal that the primary should be resumed. This mixed orchestration intent into agent responses — agents are supposed to be pure.

**Decision:** Remove the flag entirely. Resume intent is expressed as `on_complete=Edge("resume")` on the node definition. Agents don't know about the resume stack.

**Effect:** A new interrupt agent added to the graph automatically gets resume behaviour if its `on_complete` edge says `"resume"`. No agent code needs to change.

---

### D3 — WAITING_CONFIRM enforcement is a hard pre-check, not a prompt rule

**Context:** The router prompt contained a rule "if an agent is WAITING_CONFIRM, route back to it." This is a soft instruction to an LLM — it can be ignored if the utterance looks like a question.

**Decision:** The router checks `graph.active_waiting_confirm()` **before** any LLM call. If a node is WAITING_CONFIRM, that node is returned immediately. The LLM is not consulted.

**Effect:** A caller saying "actually, what are your fees?" while their name is being confirmed will not be routed to FAQ. The confirmation must be resolved first.

**Bug this fixed:** An earlier implementation had `on_waiting_confirm=Edge("self")` which resolved to `Edge(node_id)` — the invoker treated this as a chain target and immediately re-invoked the agent with `utterance=""`. DataCollectionAgent received an empty utterance in `waiting_confirm` stage and responded "Sorry, I didn't catch that" — consuming two responses for one confirmation turn. Fixed by defaulting `on_waiting_confirm` to `Edge("decider")` (speak the confirmation, wait for next utterance, then the hard enforcement handles routing).

---

### D4 — Interrupt agents are re-invokable across a conversation

**Context:** After an interrupt agent reaches COMPLETED or FAILED, it was no longer in `available_nodes()`. This meant callers could only ask one FAQ question per conversation.

**Decision:** `available_nodes()` allows interrupt-eligible agents regardless of COMPLETED or FAILED status.

**Rationale:** Callers naturally ask multiple questions. "What are your hours?" early on and "What are your fees?" later are separate interactions, not retries of the same one. `invocation_count` in `AgentState` tracks usage for analytics.

---

### D5 — NarrativeCollectionAgent: content-based completion, not turn-based

**Context:** Early designs considered a turn-count threshold (e.g. "ask if done after N turns") or a fixed-duration silence detection from STT.

**Decision:** Completion is detected by a fast LLM call (64 tokens, temperature 0.2) that reads the last two segments and total word count and decides if the narrative is "reasonably complete." The check only fires after at least 2 segments with 4+ words each.

**Rationale:** Turn-based thresholds force awkward timing — ask after 3 turns and a terse caller gets interrupted; wait for 8 turns and a verbose caller gets an unnecessary filler. Content-based detection lets the agent respond to what was actually said. The minimum segment threshold prevents the "Is there anything else?" question from firing after a single opening sentence.

---

### D6 — NarrativeCollectionAgent: NOT interrupt_eligible

**Context:** It was considered making `narrative_collection` `interrupt_eligible=True`, meaning it could be invoked by the decider and wouldn't naturally be the `active_primary()`. The alternative was making it a non-interrupt primary, which means it IS `active_primary()`.

**Decision:** `interrupt_eligible=False`. `narrative_collection` is a primary goal agent.

**Effect:** Because it IS `active_primary()`, when the router detects an interrupt (e.g. FAQ question mid-narrative), the standard mechanism pushes `narrative_collection` onto the resume stack. After FAQ completes, `on_complete=Edge("resume")` pops it and re-invokes it with `utterance=""`. The agent re-prompts "Please go ahead, I'm still listening." The `segments` list in `internal_state` is preserved across all interruptions — the full narrative is always the concatenation of everything the caller said.

---

### D7 — NarrativeCollectionAgent: "Is there anything else?" gate

**Context:** One option was to have the agent complete immediately when the LLM says the narrative is complete. Another was to always ask the gate question regardless of completion.

**Decision:** When the LLM completion check returns `true`, transition to `asking_done` stage and ask "Is there anything else you'd like to add?" Wait for the caller's response before completing.

**Rationale:** The completion check is a heuristic — it can fire early if the caller says something that sounds conclusive. The gate gives the caller control. If they say "no, that's all" → complete. If they say "actually, I should mention..." → append the new utterance and return to collecting. This mirrors how a human receptionist would behave.

---

### D8 — MetLife ID stays in DataCollectionAgent

**Context:** A law firm intake might need a MetLife member ID (or similar insurance identifier) if the matter is insurance-related. Two options were considered:
- **Option A:** Collect it as a structured field in `data_collection` alongside name, email, phone
- **Option B:** Collect it as part of the free-form narrative (the caller may mention it naturally)

**Decision:** Option A — MetLife ID stays in `data_collection`.

**Rationale:** The ID is a structured, validatable value. It benefits from the extract → validate → spell-back → confirm pipeline. A member ID extracted from free-form prose has no validation or correction loop. If the caller mentions it in the narrative, it will appear in `full_narrative` for the intake coordinator to note — but the agent won't attempt extraction from prose.

---

### D9 — IntakeQualificationAgent: ambiguous defaults to COMPLETED

**Context:** When the LLM says the case type is `"unknown"` or only loosely related to the firm's practice areas, should the call proceed to scheduling or end?

**Decision:** `ambiguous` maps to `COMPLETED` and proceeds to scheduling. The speak is slightly different from `qualified` — it acknowledges that a specialist will assess at the consultation.

**Rationale:** False negatives (rejecting a valid client) are more costly than false positives (scheduling a consultation for a borderline case). A human lawyer makes the final determination. The AI should be conservative about rejections. Additionally, if the LLM call itself fails, the default is `ambiguous` — never a silent rejection.

---

### D10 — IntakeQualificationAgent: direct edge chain, not auto_run

**Context:** Two mechanisms could trigger `intake_qualification` without a caller utterance:
- `auto_run=True` + `depends_on=["narrative_collection"]`: invoker's auto-run scan would pick it up after the turn
- Direct edge: `narrative_collection.on_complete=Edge("intake_qualification")`

**Decision:** Direct edge chain.

**Rationale:** `auto_run=True` is appropriate for post-processing that happens after the conversation ends (webhook). `intake_qualification` needs to run within the same turn as `narrative_collection`'s completion — the caller should hear the qualification result and the scheduling prompt as one coherent response, not as a separate turn. The direct chain achieves this: NC completes, IQ runs immediately, IQ either ends the call (not_qualified) or chains to scheduling, all within the same turn.

---

### D11 — Single scheduling type; multi-type scheduling parked

**Context:** A future requirement was raised: supporting multiple consultation types (free initial call, paid 30-min consult, paid 60-min consult) as separate Calendly event types.

**Decision:** Park this for now. The current `SchedulingAgent` presents slots from a single Calendly event type.

**Rationale:** Multi-type scheduling requires either a pre-scheduling type-selection step (a 5th primary goal) or a modified scheduling agent that handles the choice within its own flow. Either approach is non-trivial. The 4-goal sequence is already a meaningful change; adding complexity before it is validated in production would be premature.

---

### D12 — IntakeQualificationAgent: separate agent, not merged into NarrativeCollectionAgent

**Context:** It was considered having `narrative_collection` detect the case type and make the qualification decision internally before returning COMPLETED.

**Decision:** Keep qualification as a separate agent.

**Rationale:** These are distinct responsibilities. `narrative_collection` is responsible for gathering information — it should not make a business decision about whether to proceed. `intake_qualification` reads the collected narrative and makes that decision. Having it as a separate node also means it appears explicitly in the graph, its decision is logged independently, and its `on_failed` edge is a first-class transition in the workflow rather than buried in the narrative agent's completion logic.

---

## 8. Simulation Coverage

The workflow engine is tested via `data-plane/simulate.py`, a deterministic harness that runs `WorkflowDefinition` edge logic without real LLM, STT, or TTS calls. All agents are replaced with `ScriptedAgent` (pre-scripted response queues) or `ErrorAgent` (always raises). The decider can be replaced with a `decider_mock` callable.

### Current Scenarios (all passing)

| # | Scenario | What is verified |
|---|---|---|
| 1 | Happy path (4-goal) | All 4 goals complete in order; booking confirmed |
| 2 | FAQ interrupt mid-collection | Resume stack fires; collection resumes at right field |
| 3 | Fallback chain (faq→docs→fallback) | `on_failed` edges drive the chain; resume fires at end |
| 4 | 3 consecutive agent errors | Error policy fires `end("agent_error")` |
| 5 | WAITING_CONFIRM enforcement | Bad decider overridden; faq call count = 0 |
| 6 | Multiple FAQs during collection | FAQ invoked 3×; collection and booking still complete |
| 7 | FAQs during scheduling | Scheduling re-presented after each FAQ; booking succeeds |
| 8 | Narrative happy path (multi-turn) | NC accumulates 3 segments; asks gate; COMPLETED; IQ → scheduling |
| 9 | FAQ interrupt mid-narrative | Resume re-invokes NC with ""; segments preserved across interrupt |
| 10 | Not-qualified path | IQ returns FAILED → `end("not_qualified")`; scheduling never invoked |
| 11 | Ambiguous qualification | IQ COMPLETED (ambiguous) → scheduling proceeds; booking succeeds |

### What Simulation Validates vs. Integration Testing

**Simulation** answers: does the workflow definition produce the correct agent sequence for a given conversation? Deterministic, fast, no API calls.

**Integration testing** (not yet implemented) answers: do the LLM calls return sensible values for real utterances? Should be run against the real Cerebras model and flag prompt regressions.

---

## 9. Open Items

### 9.1 Multi-type scheduling

Parked (see D11). When the Calendly integration supports multiple event types, this requires either:
- A type-selection step inserted between `intake_qualification` and `scheduling` (a 5th primary goal)
- A modified `SchedulingAgent` that receives the desired type from collected data and selects the appropriate Calendly event type internally

### 9.2 SchedulingAgent FAILED path

`SchedulingAgent` currently never returns `AgentStatus.FAILED`. If Calendly is unreachable or returns persistent errors, the agent stays `IN_PROGRESS` indefinitely. A `max_retries` counter inside the agent should return `FAILED` after exhausting retries. The `on_failed=Edge("end", "scheduling_failed")` edge is already declared.

### 9.3 Integration tests for LLM prompts

The simulation harness uses scripted agents. The extraction, completion detection, and qualification prompts are not regression-tested against real utterances. Prompt changes can silently degrade performance. A suite of integration tests hitting the real LLM with representative utterances would catch these regressions.

### 9.4 "Not recommended" referral speak

When `intake_qualification` returns `not_qualified`, the current speak is generic ("seek a firm that specialises in your area of need"). A better experience would reference the caller's detected case type: "For a criminal defence matter, we'd recommend reaching out to a specialist criminal law firm." This could use `case_type` from collected data in the speak generation prompt.

### 9.5 Narrative resumption after multiple interrupts

The current resume stack is a list (LIFO). If a caller triggers an FAQ interrupt during a narrative, then triggers another FAQ during the FAQ's response, both primary agents are pushed. On second resume, the most recent is popped first. This is the correct LIFO behaviour, but it has not been explicitly tested with more than one interrupt in flight. Scenario 9 covers a single interrupt; a two-level interrupt scenario should be added.

### 9.6 Filler phrase variety

The `_FILLERS` list in `NarrativeCollectionAgent` has 5 entries, chosen pseudo-randomly. For long narratives, the same filler may repeat. A more sophisticated filler selection could track which phrases have been used and avoid repetition.

---

*Document maintained in `docs/intake-workflow-design.md`*
*Code: `data-plane/app/agents/`*
*Simulation: `data-plane/simulate.py`*
*Graph definition: `data-plane/app/agents/graph_config.py`*
