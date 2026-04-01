# State Space Analysis — Proposal

## Why this is worth doing

The system has **two layers of state** that interact in non-obvious ways, and both are LLM-influenced:

1. **Workflow graph layer** — `AgentStatus` per node, resume stack, `consecutive_errors`, tool task lifecycle
2. **Agent internal layer** — `DataCollectionAgent`'s `collected`, `pending_confirmation`, `current_field`; `NarrativeCollectionAgent`'s narrative buffer; `SchedulingAgent`'s slot/stage

The LLM drives transitions (planner classification, router decision, agent extraction) but the *graph* holds state. This means the same graph state can produce different transitions depending on what the LLM decides — and the LLM's decision quality degrades gracefully but doesn't fail hard. So pathological states don't crash; they produce a subtly wrong response and advance to a worse state.

That's exactly the failure mode state space analysis is designed to find.

---

## The State Dimensions

Before naming bugs, here's what actually constitutes "a state" in this system:

**Graph layer**
- Status per node (9 nodes × 6 statuses: `NOT_STARTED`, `IN_PROGRESS`, `WAITING_CONFIRM`, `COMPLETED`, `FAILED`, `UNHANDLED`)
- Resume stack contents and depth (which agents, in which order)
- `consecutive_errors` counter (0–3)
- Pending tool tasks (CalendarPrefetchTool: pending / complete / failed / abandoned)

**Agent internal layer**
- `collected` dict: any subset of the configured required + optional fields
- `pending_confirmation`: `{field, value}` or `None`
- `current_field` being asked
- `narrative_collection` buffer content (partial vs. complete)
- `scheduling` stage: which slots were presented, which were selected

**Call layer**
- `CallState`: CONNECTING → ACTIVE → COMPLETING → DONE / ERROR
- Turn counter
- Transcript history (fed into LLM context)

Full Cartesian product is enormous, but the *reachable* states form a much smaller set — and the interesting ones are the near-valid states one or two unexpected utterances away from the happy path.

---

## Problematic State Classes

### 1. Stuck / Deadlock States
The system reaches a state where no turn can make forward progress.

- `WAITING_CONFIRM` active on a node, but the utterance that would resolve it is mis-classified as `NARRATIVE` by the planner. The planner's `_validate_waiting_confirm` injects the waiting agent at the *end* of the plan with `use_empty_utterance=True`. The empty utterance hits `DataCollectionAgent`, which re-asks the confirmation. Caller answers again. Planner again classifies it as something else. Loop.
- Resume stack contains a node that has since moved to `COMPLETED` or `FAILED`. When the stack pops and handler invokes it with an empty utterance, the completed node returns `COMPLETED` again — but no edge follow-through was planned. Handler may silently advance without speaking.
- Calendar prefetch task is still pending when `scheduling` agent runs. Handler awaits it. If prefetch never resolves (hung external call, no timeout), the call is stuck in silence with TTS never triggered.

### 2. Livelock States
The system keeps looping, producing responses, but never advancing to the next primary goal.

- `data_collection` returns `WAITING_CONFIRM`. Caller says something unrelated ("what are your fees?"). Planner correctly handles FAQ first, then injects `data_collection` as `use_empty_utterance=True`. `data_collection` re-asks. Caller answers. Planner classifies as `FIELD_DATA` → `data_collection` runs. But the field extraction fails (bad confidence). Agent returns `WAITING_CONFIRM` again with the same unconfirmed value. Caller says yes but phrased oddly. Repeat.
- Router fallback prefers `data_collection` when LLM fails. If Cerebras is degraded and every planner call falls back, and `data_collection` is `COMPLETED`, the fallback preference order still puts `data_collection` first — not `narrative_collection`. Every turn re-invokes a completed agent that says "I have everything I need, how can I help?" The call circles indefinitely.
- `EmpathyAgent` is in `available_ids` (not yet `COMPLETED`) when caller gives a `NARRATIVE` intent. Planner prepends empathy to every narrative step. If `EmpathyAgent.on_complete` edges back to `decider` and the planner still sees it as `NOT_STARTED` (because `available_nodes()` includes it while not `COMPLETED` or `FAILED`), it will be invoked again on the next `NARRATIVE` turn — empathy fires twice, three times.

### 3. Data Loss States
Collected data silently disappears from the confirmed state.

- `reset_fields` is called on a `CORRECTION` intent. The `collected_all` dict in handler is cleared for that field. But if the caller's utterance was mis-classified as `CORRECTION` when it was actually a `CONFIRMATION` (ambiguous "no wait, my name is..." phrasing), a correctly confirmed field is erased and re-asked.
- When `WAITING_CONFIRM` is active and the planner classifies the utterance as `CONFIRMATION`, `data_collection` is invoked and confirms the field. But the `SubagentResponse.collected` dict carries the confirmed value. The handler adds it to `collected_all`. However, if the same turn also triggers a `reset_fields` step *after* the invoke (possible if a subsequent `CORRECTION` intent is in the same multi-intent plan), the just-confirmed field gets immediately cleared.
- `WorkflowGraph.update()` downgraded `WAITING_CONFIRM → IN_PROGRESS` when an agent returns `UNHANDLED`. The `pending_confirmation` stays in `internal_state`. Next invocation of `data_collection` sees a stale `pending_confirmation` for a field the caller already answered a different way. Agent re-asks for a value already confirmed.

### 4. Phantom Confirmation / Confirmation Loop
The system asks for confirmation but the confirmation never resolves, or resolves incorrectly.

- `data_collection` is `WAITING_CONFIRM`. An interrupt (FAQ) fires. The graph's `update()` for the interrupt agent doesn't touch `data_collection`'s status. `active_waiting_confirm()` still returns `data_collection`. Next turn, Router hard-routes back to `data_collection`. But the caller hasn't answered — they're still getting the FAQ response. The re-routing interrupts the expected flow.
- The planner's `PENDING CONFIRMATION` context is populated from `dc_state.internal_state.get("pending_confirmation")`. If `data_collection`'s `internal_state` was not fully written back to the graph state (partial update between async steps), the planner sees no pending confirmation even though WAITING_CONFIRM is set — so it doesn't classify the caller's "yes" as `CONFIRMATION`, routes it to the wrong agent.
- Caller says "yes" (confirming a name read-back) but Deepgram transcribes it as "yeah" or "yep" mid-word ("yeah right"). Planner correctly classifies `CONFIRMATION`, passes to `data_collection`. But inside the agent, if the confirmation-matching logic compares against a narrow expected set, "yeah right" might be treated as `REJECTION` (the "right" being interpreted as sarcasm), clearing the pending value and re-asking.

### 5. Wrong Terminal / Premature Termination
The call ends before it should, or fails to end when it should.

- `FarewellAgent` is interrupt-eligible. Planner maps `FAREWELL` to `farewell` and stops adding steps. But `_validate_waiting_confirm` runs *after* `_build_steps_from_intents`. It detects an active `WAITING_CONFIRM` and appends the waiting agent at the end. Handler now executes: farewell (→ edges to `end` + `caller_farewell`), then tries to execute the waiting agent. If the handler checks `finalization_reason` before executing remaining steps, the waiting agent is skipped. If it doesn't, the waiting agent fires post-farewell and speaks into a dead call.
- `consecutive_errors` increments on any exception in agent invocation. If a tool call inside an agent raises (e.g., calendar API timeout), that's counted as an agent error. Three API timeouts in a row (possible during Calendly degradation) terminate the call with `agent_error` — even though the conversation was going fine and the errors were purely infrastructure.
- `intake_qualification` returns `FAILED` on `not_qualified`. But the agent's determination of "not qualified" is an LLM call. If the narrative was very short (caller gave just one sentence before asking a FAQ), the qualification LLM may have too little signal and produce a false negative. Call ends with `not_qualified` for someone who actually qualified.

### 6. Orphaned Intent / Misrouted Utterance
The caller's actual intent is processed by the wrong agent or not at all.

- Multi-intent utterance: `[FIELD_DATA, NARRATIVE]`. Planner invokes `data_collection` first (extracts name), then `narrative_collection`. But `narrative_collection` receives the *full original utterance*, not the narrative portion. It may re-extract the name as part of the narrative, creating a duplicate entry or confusing the narrative content.
- Planner maps `CONTINUATION` to `active_primary()`. But `active_primary()` returns the first node that is `IN_PROGRESS` or `WAITING_CONFIRM` in graph iteration order — not necessarily the one that spoke last. If `data_collection` is `IN_PROGRESS` and `narrative_collection` is also `IN_PROGRESS`, and the caller's continuation is clearly about their legal situation, it still routes to `data_collection` (lower goal_order, earlier in iteration).
- `FAREWELL` intent classification has a conservative threshold in the planner prompt ("NOT farewell: 'Okay. Thank you.'"). But "Thank you so much, that's great" mid-call (after a FAQ answer) can tip into `FAREWELL` classification, especially combined with TTS delivering an enthusiastic response. Call terminates mid-booking.

### 7. Resume Stack Corruption
The resume mechanism diverges from the actual graph state.

- The resume stack lives on the `Planner` instance, not in the `WorkflowGraph`. If handler re-creates the planner (not the current case, but relevant for any refactor), the stack is lost. On resume, `pop_resume()` returns `None`, the handler defaults to the primary goal, but the interrupted primary was mid-`WAITING_CONFIRM` — now that agent gets an unexpected utterance without its confirmation context.
- Interrupt within interrupt: `data_collection` is active, `faq` fires (pushed to stack: `[data_collection]`), then during the FAQ response the caller asks another question. If handler pushes `faq` again: stack is `[data_collection, faq]`. When inner FAQ completes, it pops `faq` and invokes faq with empty utterance. `faq` likely returns `UNHANDLED` (nothing to handle). Handler follows `faq.on_continue → decider`. Now stack is `[data_collection]` still — but the caller's original question may have been lost.
- `push_resume` has a guard: `if agent_id not in self._resume_stack`. If `data_collection` is interrupted twice for two separate FAQs, the second interrupt push is silently dropped. Second FAQ completes, resumes to `data_collection`. First FAQ's resume to `data_collection` is already consumed. Net: two FAQs both resume correctly, but this guard could mask a deeper stack inconsistency.

### 8. Context Window Drift
The history fed into the LLM drifts from the actual call state, causing increasingly inaccurate routing and extraction.

- `recent_history` is capped at 6-8 turns in the planner/router. On long calls (frequent FAQs, complex narrative, multi-round scheduling), the data collection questions and answers fall out of the context window. The planner can no longer see what was collected and may re-classify already-answered questions as `FIELD_DATA`, routing back to `data_collection` which already has them confirmed.
- The `status_summary()` injected into the router shows `last_spoke` truncated at 60 characters. For long confirmations ("I have your full name as John Michael Smith — is that correct?"), the truncation removes the value being confirmed. The router sees `WAITING_CONFIRM` but not *what* is being confirmed, potentially misrouting.

---

## Approaches to State Space Exploration

### Approach A — Combinatorial State Injection (highest immediate value)

Directly set the `WorkflowGraph` and agent `internal_state` to specific combinations, then drive a single utterance through the full pipeline and assert on outputs.

The state space is structured enough to enumerate targeted combinations:
- For each `(node_status_vector, pending_confirmation_present, resume_stack_depth, utterance_type)` tuple
- Assert: which agent responds, what `speak` is produced, what the graph state is afterward

This is essentially "what does the system do when it *finds itself* in state X?" — orthogonal to how it got there. It's faster to write than full-path simulations and covers more of the reachable space cheaply.

Works well here because the `WorkflowGraph` and `AgentState` are plain Python dataclasses — you can instantiate them directly without running a full call.

### Approach B — Property-Based Testing (Hypothesis)

Define invariants that must hold for *any* sequence of utterances and *any* graph state, then use a generator to auto-explore the space.

Example invariants:
- "After any turn, there is at most one node in `WAITING_CONFIRM`"
- "If `data_collection` is `COMPLETED`, all required fields are present in `collected`"
- "If `resume_stack` is non-empty, the top element is a non-interrupt-eligible agent that is `IN_PROGRESS`"
- "A `FAREWELL` planner step is always the last invoke step — nothing follows it"
- "After `scheduling.on_complete` fires, `CallState` transitions to `COMPLETING` within the same turn"

Hypothesis generates utterance sequences (random strings, real-world utterance templates, adversarial inputs) and shrinks failing cases to the minimal reproduction.

This is the most scalable approach for discovering *unknown* bugs — you don't need to know the bug exists to write the invariant.

### Approach C — Transition Coverage Simulation

Build an exhaustive state graph where:
- Nodes = `(graph_status_vector, pending_confirmation, resume_stack_signature)`
- Edges = utterance + planner decision

Walk the graph with BFS/DFS, flagging:
- Nodes with no outgoing edges that aren't terminal (`DONE`/`ERROR`) — **stuck states**
- Strongly connected components that don't include a terminal — **livelocks**
- Nodes reachable only via unexpected LLM decisions — **fragile states**

The LLM layer makes this non-deterministic, but you can model it with a bounded sample: run N=50 utterances of each intent type from each state, record all observed next states. This gives you a stochastic transition matrix rather than a deterministic one, which is more honest about the system's real behavior.

This is more expensive to build but produces a *map* of the state space that you can visualize and reason about across code changes.

### Approach D — Mutation Testing on the Planner / Router

Take the existing simulate scripts (which are known-good paths) and systematically mutate:
- Swap utterance intent types ("yes" → "no", "my name is John" → "my name is John but actually wait")
- Inject unexpected utterances at each turn (farewell mid-collection, correction of uncollected field)
- Inject API failures at specific turns (LLM fail on turn 3, calendar fail on turn 7)
- Vary the order of multi-intent utterances

Assert that the system reaches an acceptable terminal state (not necessarily the happy path, but not stuck or looping). This is the cheapest approach and directly extends the existing test infrastructure.

### Approach E — Trace-Based Replay Fuzzing

Extend the evaluator to:
1. Take a real transcript
2. Replay it up to turn N
3. At turn N, inject a diverging utterance (FAQ, correction, ambiguous phrasing)
4. Continue the call and observe whether it recovers to the happy path

This tests recoverability — can the system get back on track after an unexpected detour? It uses real call structure as the starting point, so the initial states are realistic, and the divergence points are systematically explored.

---

## Recommended Starting Point

The state space is too large for exhaustive coverage. The highest-value starting point is a combination of **A** and **D**:

1. **State injection tests** for the known dangerous combinations identified above (especially `WAITING_CONFIRM` + interrupt, resume stack + completed node, and `reset_fields` + same-turn `collected` write)
2. **Mutation of existing simulate scripts** — each existing scenario is a known path; adding ±2 unexpected utterances at each turn dramatically expands coverage for minimal new infrastructure

**B (property-based testing)** is the highest long-term investment — defining invariants is valuable even independent of Hypothesis, because the invariants document what the system is supposed to guarantee.

**C (full transition graph)** is worth doing once the system stabilizes, as a regression safety net rather than a discovery tool.

**E (trace fuzzing)** becomes most valuable once you have enough real call transcripts to make the starting states representative.
