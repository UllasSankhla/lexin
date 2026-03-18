# Workflow Design: LLM-Driven Agent Orchestration

## Table of Contents
1. [Current Architecture](#1-current-architecture)
2. [Proposed Architecture](#2-proposed-architecture)
3. [Gap Analysis](#3-gap-analysis)
4. [Simulation Approach](#4-simulation-approach)

---

## 1. Current Architecture

### 1.1 Overview

The current system is a goal-directed multi-agent pipeline where a single LLM
"router" selects which agent handles each caller utterance. Agents are stateful
and pure — each one receives an utterance, its own internal state, a config
snapshot, and recent history; it returns a `SubagentResponse`. The orchestration
logic (fallback chain, resume stack, goal continuation, error recovery) lives as
imperative code in `handler.py`.

### 1.2 Components

#### WorkflowNode (`agents/workflow.py`)
```python
@dataclass
class WorkflowNode:
    id: str
    description: str
    depends_on: list[str]       # dependency gate — node not available until deps COMPLETED
    interrupt_eligible: bool    # can interrupt a primary agent mid-turn
    auto_run: bool              # invoked by invoker automatically, not by router
    completion_criteria: str    # human-readable, informational only
```

There is no concept of a "primary goal node" — the distinction between
goal-critical agents (data_collection, scheduling) and support agents
(faq, context_docs, fallback, webhook) is implicit:
- Goal-critical nodes: `interrupt_eligible=False`, `auto_run=False`
- Support/interrupt nodes: `interrupt_eligible=True`
- Post-processing: `auto_run=True`

#### WorkflowGraph (`agents/workflow.py`)
- Holds `nodes: dict[str, WorkflowNode]` and `states: dict[str, AgentState]`
- `available_nodes()` — nodes whose deps are met and status is not COMPLETED/FAILED
- `active_primary()` — the non-interrupt, non-auto-run node currently IN_PROGRESS or WAITING_CONFIRM
- `next_primary_goal()` — next runnable incomplete primary node
- `is_goal_complete()` — all non-interrupt, non-auto-run nodes COMPLETED
- `is_goal_terminal()` — all non-interrupt, non-auto-run nodes COMPLETED or FAILED
- `auto_run_ready()` — auto_run nodes whose deps are met and status is NOT_STARTED

#### Router (`agents/router.py`)
- Single LLM call per turn (1024 tokens)
- Input: utterance, recent_history[-8], graph status block, available agents, goal string
- Output: `{ agent_id, interrupt, reasoning }`
- Side effect: if `interrupt=true`, pushes `active_primary()` onto `_resume_stack`
- Fallback heuristic if LLM fails: `data_collection → scheduling → faq → fallback`
- JSON repair if response truncated: `_try_repair_json()` via regex

#### Agents (pure functions in `agents/`)
| Agent | Type | Key behaviour |
|---|---|---|
| `data_collection` | primary | ask → extract+refusal → validate → spell_back → confirm, field by field |
| `scheduling` | primary | present slots → choice → confirm → book via Calendly |
| `faq` | interrupt | LLM-match utterance to FAQ list → rephrase answer |
| `context_docs` | interrupt | LLM-answer from uploaded business documents |
| `fallback` | interrupt | acknowledge + record note |
| `webhook` | auto_run | fire POST to configured endpoints after scheduling |

#### Handler Orchestration (`websocket/handler.py`)
The orchestration logic is imperative, encoded as a fixed sequence after each
agent invocation:

```
1. Router selects agent
2. Agent invoked
3. Fallback chain  — hardcoded: faq → context_docs → fallback
   (triggered when agent is interrupt_eligible AND FAILED/empty speak)
4. Resume stack    — hardcoded: if requires_router_resume, pop stack, invoke primary
5. Goal continuation — hardcoded:
     a. If interrupt agent ran (and no resume): invoke next_primary_goal()
     b. If primary agent just COMPLETED: invoke next_primary_goal()
6. Persist: collected fields, notes, booking
7. Auto-run: webhook when deps met
8. Goal check:
     is_goal_complete()  → finalize("completed")
     is_goal_terminal()  → finalize("goal_failed")
9. TTS speak
10. Error handler: consecutive_errors++; ≥3 → finalize("agent_error")
```

### 1.3 Current Graph Definition

```
faq          (interrupt_eligible, no deps)
context_docs (interrupt_eligible, no deps)
fallback     (interrupt_eligible, no deps)
data_collection (primary, no deps)
scheduling      (primary, depends_on: [data_collection])
webhook         (auto_run, depends_on: [scheduling])
```

Goal string passed to router:
> "Collect caller information and schedule an appointment."

### 1.4 What Works Well
- Agents are cleanly isolated — pure functions, no shared state
- Router uses graph state in its prompt (status block + available agents)
- Fallback chain prevents silent failures when interrupt agents have no answer
- Resume stack correctly restores primary context after an interruption
- Goal continuation chains data_collection → scheduling automatically
- Error policy is explicit (3 strikes, then finalize)

### 1.5 What Is Brittle
- Fallback chain order (`faq → context_docs → fallback`) is hardcoded in handler.py
- Resume stack logic is bespoke code, not driven by node configuration
- Goal continuation logic (`primary_just_completed`, `interrupt_needs_continuation`)
  is hardcoded, not driven by node edges
- Primary goal nodes have no explicit flag — inferred by eliminating interrupt/auto_run
- The router goal string is a free-text constant, not derived from the workflow definition
- Adding a new workflow requires editing handler.py, not just defining a new graph

---

## 2. Proposed Architecture

### 2.1 Core Principle

The decider (LLM router) works **backwards from explicit goal nodes**. The workflow
definition states the goal as an ordered list of required primary agents. All
other agents exist to serve those goals. The invoker follows **declared edges**
rather than hardcoded imperative logic.

### 2.2 Schema

```python
@dataclass
class Edge:
    target: str
    # Targets:
    #   "<node_id>"  — invoke that node immediately (same turn, no decider)
    #   "decider"    — speak response, wait for next utterance, consult decider
    #   "resume"     — pop resume stack; invoke primary with empty utterance
    #   "end"        — finalize the call
    reason: str = ""   # surfaced in finalization log / analytics


@dataclass
class ActivityNode:
    id: str
    agent_class: str              # maps to a registered AgentBase
    description: str              # injected into decider prompt
    is_primary_goal: bool = False # explicitly marks goal-critical nodes
    goal_order: int | None = None # execution order among primary goals (1, 2, ...)
    interrupt_eligible: bool = False
    auto_run: bool = False
    depends_on: list[str] = field(default_factory=list)

    # Declared transitions
    on_complete: Edge = Edge("decider")
    on_failed:   Edge = Edge("decider")
    on_continue: Edge = Edge("decider")
    # on_continue applies to both IN_PROGRESS and WAITING_CONFIRM:
    #   speak the response, wait for next caller utterance, consult decider


@dataclass
class GoalSpec:
    primary_agents: list[str]   # ordered node ids — must all be COMPLETED for success
    description: str            # injected into decider prompt as the mission statement
    # "primary_agents" replaces the implicit inference from interrupt_eligible/auto_run


@dataclass
class DeciderSpec:
    system_prompt: str          # full routing prompt; receives goal + graph state
    max_tokens: int = 1024


@dataclass
class InterruptPolicy:
    eligible_agents: list[str]   # node ids allowed to interrupt primary agents
    resume_strategy: str = "stack"


@dataclass
class ErrorPolicy:
    max_consecutive_errors: int = 3
    transient_error_speak: str = "I'm sorry, something went wrong. Could you repeat that?"
    on_max_errors: Edge = Edge("end", reason="agent_error")


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

### 2.3 Appointment Booking — Proposed Definition

```python
APPOINTMENT_BOOKING = WorkflowDefinition(

    id="appointment_booking",
    description="Collect caller information and schedule an appointment",

    goal=GoalSpec(
        primary_agents=["data_collection", "scheduling"],  # explicit, ordered
        description=(
            "You must complete two primary goals in order:\n"
            "1. data_collection — collect all required caller information\n"
            "2. scheduling — present available slots and confirm a booking\n"
            "Always work toward the next incomplete primary goal. "
            "Interrupt agents (faq, context_docs, fallback) exist only to "
            "answer side questions; return to the primary goal immediately after."
        ),
    ),

    decider=DeciderSpec(
        system_prompt="""\
You are the routing decider for a voice appointment booking assistant.
Your mission: {goal.description}

RULES:
1. If a primary agent is WAITING_CONFIRM, always route back to it — it needs a yes/no.
2. If the caller is answering a question (providing information), route to the
   active or next primary goal agent.
3. If the caller is asking a question (not providing information), route to an
   interrupt-eligible agent (faq, context_docs, fallback).
4. After any interrupt-eligible agent completes, the invoker will resume the
   primary goal — you do not need to re-select it.
5. Only select agents listed in AVAILABLE AGENTS.

Respond ONLY with valid JSON:
{"agent_id": "<id>", "interrupt": <true|false>, "reasoning": "<one line>"}
""",
        max_tokens=1024,
    ),

    nodes=[

        # ── Support: interrupt-eligible ──────────────────────────────────────

        ActivityNode(
            id="faq",
            agent_class="FAQAgent",
            description="Answers specific questions from the curated FAQ list.",
            interrupt_eligible=True,
            on_complete=Edge("resume"),       # answered → resume primary
            on_failed=Edge("context_docs"),   # no match → try docs (explicit edge)
            on_continue=Edge("decider"),
        ),

        ActivityNode(
            id="context_docs",
            agent_class="ContextDocsAgent",
            description="Answers general questions from uploaded business documents.",
            interrupt_eligible=True,
            on_complete=Edge("resume"),
            on_failed=Edge("fallback"),       # no docs / no answer → fallback (explicit edge)
            on_continue=Edge("decider"),
        ),

        ActivityNode(
            id="fallback",
            agent_class="FallbackAgent",
            description="Acknowledges unanswerable questions and records a note.",
            interrupt_eligible=True,
            on_complete=Edge("resume"),
            on_failed=Edge("resume"),         # even on error, return to primary
            on_continue=Edge("decider"),
        ),

        # ── Primary goal 1: data collection ──────────────────────────────────

        ActivityNode(
            id="data_collection",
            agent_class="DataCollectionAgent",
            description=(
                "PRIMARY GOAL 1: Collect all required caller information "
                "one field at a time with confirmation."
            ),
            is_primary_goal=True,
            goal_order=1,
            interrupt_eligible=False,
            depends_on=[],
            on_complete=Edge("scheduling"),   # completed → chain immediately to goal 2
            on_failed=Edge("end", reason="data_collection_failed"),
            on_continue=Edge("decider"),      # mid-field → speak, wait, re-route
        ),

        # ── Primary goal 2: scheduling ────────────────────────────────────────

        ActivityNode(
            id="scheduling",
            agent_class="SchedulingAgent",
            description=(
                "PRIMARY GOAL 2: Present available appointment slots, "
                "take the caller's choice, confirm and book."
            ),
            is_primary_goal=True,
            goal_order=2,
            interrupt_eligible=False,
            depends_on=["data_collection"],
            on_complete=Edge("end", reason="completed"),
            on_failed=Edge("end", reason="scheduling_failed"),
            on_continue=Edge("decider"),
        ),

        # ── Post-processing: auto-run ─────────────────────────────────────────

        ActivityNode(
            id="webhook",
            agent_class="WebhookAgent",
            description="Dispatches call summary and collected data to webhooks.",
            auto_run=True,
            depends_on=["scheduling"],
            on_complete=Edge("end"),
            on_failed=Edge("end", reason="webhook_failed"),
            on_continue=Edge("end"),
        ),
    ],

    interrupt_policy=InterruptPolicy(
        eligible_agents=["faq", "context_docs", "fallback"],
        resume_strategy="stack",
    ),

    error_policy=ErrorPolicy(
        max_consecutive_errors=3,
        transient_error_speak=(
            "I'm sorry, something went wrong on my end. "
            "Could you please repeat that?"
        ),
        on_max_errors=Edge("end", reason="agent_error"),
    ),
)
```

### 2.4 Standardized Invoker Algorithm

The invoker reads any `WorkflowDefinition` and follows edges mechanically.
No workflow-specific logic lives in the invoker.

```
INVOKER(workflow, utterance):

  PRE-TURN:
    run auto_run nodes whose deps just became satisfied
    (follows on_complete edge when done)

  ROUTE:
    decider.select(utterance, history, graph_state, workflow.goal)
    → { agent_id, interrupt }

    if interrupt=true:
      push active_primary_goal() onto resume_stack

  INVOKE(agent_id, utterance):
    response = agent.process(utterance, state, config, history)
    graph.update(agent_id, response)
    edge = node.on_complete | on_failed | on_continue  (based on response.status)
    FOLLOW(edge)

  FOLLOW(edge):
    match edge.target:

      "<node_id>":
        INVOKE(node_id, "")              # chain immediately, same turn
        (chains recursively until edge.target == "decider" or "end")

      "resume":
        primary_id = resume_stack.pop()
        INVOKE(primary_id, "")           # re-ask pending question
        compose speak: interrupt_answer + " " + primary_question

      "decider":
        speak response
        wait for next caller utterance
        → back to ROUTE

      "end":
        finalize_call(edge.reason)
        EXIT

  ERROR on any INVOKE:
    consecutive_errors++
    if consecutive_errors >= workflow.error_policy.max_consecutive_errors:
      FOLLOW(error_policy.on_max_errors)
    else:
      speak error_policy.transient_error_speak
      → back to ROUTE (edge="decider")
```

Key property: **the invoker never makes routing decisions**. It follows edges.
The decider makes routing decisions. Workflow logic is expressed entirely in
`WorkflowDefinition` data.

---

## 3. Gap Analysis

### 3.1 Gaps in the Current Architecture

#### G1 — Primary goal nodes have no explicit flag
**Problem:** Whether a node is goal-critical is inferred by exclusion
(`not interrupt_eligible and not auto_run`). The router goal string is a
free-text constant unrelated to the graph.

**Impact:** Adding a new primary goal (e.g. a payment step) requires updating
the goal string manually, not just adding a node.

**Proposed fix:** `is_primary_goal=True` + `goal_order: int` on the node.
`GoalSpec.primary_agents` is the authoritative ordered list. The decider prompt
is generated from `GoalSpec.description` rather than hardcoded.

---

#### G2 — Fallback chain is hardcoded imperative logic
**Problem:** `faq → context_docs → fallback` is a hardcoded sequence in
`handler.py`. The order and membership are not configurable.

**Impact:** Changing the fallback order or adding a new interrupt agent requires
editing handler.py, not the graph definition.

**Proposed fix:** `on_failed` edges on interrupt nodes declare the chain explicitly.
`faq.on_failed = Edge("context_docs")` and `context_docs.on_failed = Edge("fallback")`.
The invoker just follows edges.

---

#### G3 — Resume stack logic is bespoke code
**Problem:** Resume-after-interrupt is implemented as special-case handler logic
(`requires_router_resume` flag, `router._resume_stack`, `resumed_from_stack`
flag to prevent double-invocation). The `requires_router_resume` flag on
`SubagentResponse` is an agent-level signal that bleeds into orchestration
concerns.

**Impact:** Every new interrupt-eligible agent must remember to set
`requires_router_resume=True`. The double-invocation bug (fixed earlier)
arose from two code paths both triggering resume.

**Proposed fix:** `on_complete=Edge("resume")` on interrupt nodes. The invoker
pops the stack mechanically when it sees target="resume". Agents no longer need
`requires_router_resume` — the edge on the node definition is the contract.

---

#### G4 — Goal continuation is hardcoded with two special cases
**Problem:** handler.py has two separate branches for continuing the primary goal:
1. After an interrupt agent: `interrupt_needs_continuation and not resumed_from_stack`
2. After primary COMPLETED: `primary_just_completed`

Both invoke `next_primary_goal()` with bespoke conditions. This is why the
double-question bug occurred — both branches could fire.

**Impact:** Fragile conditional logic. The `resumed_from_stack` flag was added
specifically to prevent the double-fire, which means the two branches are not
fully independent.

**Proposed fix:** `on_complete=Edge("scheduling")` on `data_collection`. The invoker
calls FOLLOW(edge) immediately on COMPLETED — no special case, no flags.
"resume" edges on interrupt nodes handle the other case uniformly.

---

#### G5 — `WAITING_CONFIRM` is not a distinct edge
**Problem:** `WAITING_CONFIRM` is currently treated the same as `IN_PROGRESS`
by the invoker (both map to `on_continue=Edge("decider")`). But the correct
routing behaviour is different: a WAITING_CONFIRM agent *must* be routed back
to on the next turn regardless of what the caller says, because it is waiting
for a yes/no answer.

**Impact:** If the decider misroutes a WAITING_CONFIRM turn (e.g. sends it to
FAQ because the utterance sounds like a question), the confirmation loop breaks.
The router prompt has a rule for this ("if an agent is WAITING_CONFIRM, route
back to it") but it is only a soft instruction, not enforced at the invoker level.

**Proposed fix:** Add `on_waiting_confirm: Edge` as a separate transition.
Set it to `Edge("<self>")` for agents that must receive the next utterance.
The invoker enforces this before consulting the decider — if any node is
WAITING_CONFIRM, the next utterance routes there unconditionally.

---

#### G6 — Scheduling has no explicit FAILED path
**Problem:** `SchedulingAgent` never returns `AgentStatus.FAILED`. If Calendly
is unreachable or returns errors, the agent stays `IN_PROGRESS` indefinitely.
`is_goal_terminal()` never fires.

**Impact:** A Calendly outage leaves the call stuck until max_duration watchdog
fires.

**Proposed fix:** Add a `max_retries` to SchedulingAgent; after exhausting retries
with no slots or booking errors, return `FAILED`. The declared `on_failed` edge
then drives clean finalization.

---

#### G7 — Decider prompt is not derived from the workflow definition
**Problem:** The router system prompt is a string constant in `router.py`. The
goal description, available agent list, and rules are all manually authored and
must be kept in sync with the graph definition.

**Impact:** If a new node is added to WORKFLOW_NODES but the router prompt is
not updated, the decider won't know to route to it.

**Proposed fix:** The invoker generates the decider prompt from
`WorkflowDefinition`: goal description from `GoalSpec.description`,
available agents section from `graph.available_summary()`,
status block from `graph.status_summary()`. Only the routing rules are
static in `DeciderSpec.system_prompt`; the dynamic sections are injected
by the invoker.

---

#### G8 — `SubagentResponse.requires_router_resume` mixes concerns
**Problem:** This flag on the agent's response signals orchestration intent
(resume the primary agent), but agents are supposed to be pure — they should
not know about the resume stack.

**Impact:** FAQAgent sets `requires_router_resume=True` on COMPLETED. If we add
a new interrupt agent and forget this flag, the conversation gets stuck.

**Proposed fix:** Remove `requires_router_resume` from `SubagentResponse`.
The invoker derives resume intent from the node's `on_complete` edge
(`target="resume"`), not from the agent's response.

---

### 3.2 Gaps in the Proposed Architecture (vs. current working system)

#### G9 — Edge chaining has no cycle guard
**Problem:** If `node_A.on_complete = Edge("node_B")` and
`node_B.on_complete = Edge("node_A")`, the invoker loops infinitely within
a single turn.

**Proposed fix:** The invoker maintains a `same_turn_chain_depth` counter.
If it exceeds a threshold (e.g. 5), log a warning and fall back to `Edge("decider")`.

---

#### G10 — Multi-turn `on_continue` state is implicit
**Problem:** `on_continue=Edge("decider")` means "wait for next utterance then
ask decider". But the current state of *why* we're waiting (which field,
which slot) lives entirely inside the agent's `internal_state`, not in the edge.
The decider must infer from the status block that data_collection is mid-field.

**Impact:** Not a bug today, but means the decider needs to read internal_state
details from the status block. If the status block is incomplete, routing
can go wrong.

**Proposed fix:** Status summary should expose the agent's `stage` and
`current_field` explicitly (already done in `workflow.status_summary()`).
No structural change needed.

---

#### G11 — No `on_waiting_confirm` edge (already flagged as G5 above)
Currently both IN_PROGRESS and WAITING_CONFIRM use `on_continue`. This should
be a separate, enforced edge.

---

#### G12 — WorkflowDefinition has no version or validation
**Problem:** If `goal.primary_agents` references a node id not present in
`nodes`, this is a silent error until runtime.

**Proposed fix:** `WorkflowDefinition.__post_init__` should validate:
- All `depends_on` references exist
- All `goal.primary_agents` exist in nodes
- All `interrupt_policy.eligible_agents` exist and have `interrupt_eligible=True`
- No edge target "<node_id>" references a missing node
- `goal_order` values are unique among primary goal nodes

---

## 4. Simulation Approach

### 4.1 Goal

Run a `WorkflowDefinition` end-to-end against a scripted conversation
without a real phone call, real STT/TTS, or a real Calendly account.
Verify:
- The correct agents are selected in the correct order
- Collected fields match expected values
- Edge transitions fire as declared
- Exception paths (refusal, FAQ interrupt, scheduling retry) behave correctly

### 4.2 What Needs to be Mocked

| Component | Mock strategy |
|---|---|
| STT | Feed utterances directly as strings |
| TTS | Capture speak text as a list; no audio |
| Decider (LLM) | Scripted decision table OR a real LLM call in test mode |
| Agent LLM calls | Scripted responses keyed on (agent_id, utterance) |
| Calendly API | Fake slot list + fake booking response |
| DB | In-memory SQLite |
| WebSocket | In-process queue — send/receive as Python objects |

### 4.3 Test Harness Structure

```python
@dataclass
class SimUtterance:
    text: str
    # Optionally override decider decision for this turn:
    force_agent: str | None = None
    force_interrupt: bool = False


@dataclass
class AgentMock:
    agent_id: str
    # Keyed on utterance substring → response to return
    responses: dict[str, SubagentResponse]
    default: SubagentResponse  # used when no key matches


class WorkflowSimulator:
    def __init__(
        self,
        workflow: WorkflowDefinition,
        agent_mocks: dict[str, AgentMock],
        decider_mock: Callable[[str, list, dict], str] | None = None,
        # decider_mock(utterance, history, graph_state) → agent_id
        # If None, uses the real LLM decider
    ):
        ...

    def run(self, script: list[SimUtterance]) -> SimResult:
        ...

    # SimResult contains:
    #   - full turn trace: [(utterance, agent_id, response, edge_followed)]
    #   - final graph state per node
    #   - collected_all dict
    #   - booking_result dict
    #   - finalization reason
    #   - any unhandled exceptions
```

### 4.4 Example Test Script

```python
script = [
    # Normal happy path
    SimUtterance("John Smith"),               # → data_collection, extracts name
    SimUtterance("yes"),                      # → data_collection, confirms name
    SimUtterance("john@example.com"),         # → data_collection, extracts email
    SimUtterance("yes that's correct"),       # → data_collection, confirms email
    SimUtterance("I don't have a member ID"), # → data_collection, skips optional field
    # data_collection COMPLETED → edge fires to scheduling
    SimUtterance("the second one"),           # → scheduling, selects slot 2
    SimUtterance("yes"),                      # → scheduling, confirms booking
    # scheduling COMPLETED → goal_complete → finalize
]

agent_mocks = {
    "data_collection": AgentMock(
        agent_id="data_collection",
        responses={
            "John Smith": SubagentResponse(
                status=AgentStatus.WAITING_CONFIRM,
                speak="I have your name as John Smith. Is that correct?",
                pending_confirmation={"field": "name", "value": "John Smith"},
                internal_state={...},
            ),
            "yes": SubagentResponse(
                status=AgentStatus.IN_PROGRESS,
                speak="Got it. What's your email address?",
                collected={"name": "John Smith"},
                internal_state={...},
            ),
            ...
        },
        default=SubagentResponse(
            status=AgentStatus.IN_PROGRESS,
            speak="Sorry, could you repeat that?",
            internal_state={},
        ),
    ),
    "scheduling": AgentMock(...),
}

result = WorkflowSimulator(
    workflow=APPOINTMENT_BOOKING,
    agent_mocks=agent_mocks,
    # Use real decider LLM here to test routing decisions:
    decider_mock=None,
).run(script)

assert result.finalization_reason == "completed"
assert result.collected_all["name"] == "John Smith"
assert result.collected_all["email"] == "john@example.com"
assert result.booking_result["booking_id"] is not None
```

### 4.5 Exception Path Scripts

```python
# Test: FAQ interrupt mid-collection
script_faq_interrupt = [
    SimUtterance("John Smith"),                      # data_collection, name extracted
    SimUtterance("yes"),                             # confirms name
    SimUtterance("What are your office hours?",      # → faq (interrupt)
                 force_interrupt=True),
    SimUtterance("john@example.com"),                # resumed: data_collection, email
    ...
]

# Test: Optional field refusal
script_refusal = [
    SimUtterance("I don't have a member ID"),        # → data_collection, refused, skip
]

# Test: Scheduling failure after 3 retries
script_scheduling_fail = [
    # all data collected
    SimUtterance("none of these work"),              # scheduling retry 1
    SimUtterance("none of those either"),            # scheduling retry 2
    SimUtterance("nothing works for me"),            # scheduling → FAILED
    # → edge: on_failed → end("scheduling_failed")
]

# Test: 3 consecutive agent errors
script_errors = [
    SimUtterance("aaa", force_agent="__raise__"),    # error 1
    SimUtterance("bbb", force_agent="__raise__"),    # error 2
    SimUtterance("ccc", force_agent="__raise__"),    # error 3 → finalize("agent_error")
]
```

### 4.6 What Simulation Validates

| Scenario | What is checked |
|---|---|
| Happy path | All primary goals complete, fields collected, booking made |
| FAQ interrupt | Resume stack fires correctly; collection resumes at right field |
| Optional field refusal | LLM refused flag triggers skip, not retry |
| Required field max retry | Field stays in retry loop; call continues |
| Scheduling retry | Slot re-presentation fires; eventually confirms or fails |
| Scheduling FAILED | on_failed edge fires; call ends cleanly |
| 3 errors | apology spoken, finalize("agent_error") |
| WAITING_CONFIRM enforced | Next utterance routes back to same agent unconditionally |
| Fallback chain | faq.on_failed → context_docs; context_docs.on_failed → fallback |
| Decider prompt accuracy | Generated prompt contains all available agents |

### 4.7 Simulation vs. Integration Testing

Simulation (above) answers: *does the workflow definition produce the right
agent sequence for a given conversation?*

Integration tests answer: *do the LLM calls return sensible values for real
utterances?*

The two should be run separately. Simulation uses deterministic mocks and is
fast (no API calls). Integration tests run against the real LLM and flag
prompt regressions.

---

*Last updated: 2026-03-18*
