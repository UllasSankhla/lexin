# Steering Handlers

## What Is Steering?

Steering handlers are deterministic Python functions that intercept LLM output
at critical decision points and validate its structural correctness before it is
acted on. If an output violates a structural invariant, the handler either:

- **Rejects and retries**: injects a targeted correction into the prompt and
  re-runs the LLM call (one retry only — no infinite loops)
- **Blocks and resets**: prevents the action entirely and resets to a safe state

The key distinction from prompt engineering: a steering handler is a **guarantee**,
not advice. It fires regardless of what the LLM returns. Rules that are easy to
express in Python but hard to reliably express in natural language belong here.

### When to Use Steering

Use steering when:
- An LLM output drives an **irreversible action** (booking, confirming, recording)
- A structural invariant can be expressed as a simple Python condition
- The failure mode is expensive (wrong booking, data loss, caller confusion)
- The rule is being violated sporadically despite being in the prompt

Do not use steering for:
- Judgment calls (tone, phrasing, empathy) — these belong in the prompt
- Rules already enforced by the Python state machine (e.g. workflow graph edges)
- Agents that only generate free-form text with no structured output to validate

---

## Implemented Handlers

### 1. `DataCollectionAgent._pending_field_fence`

**File:** `data-plane/app/agents/data_collection.py`  
**Trigger:** After every `dc_extract` mega-prompt call when `pending_confirmation` is active  
**Invariant:** While a required field is awaiting yes/no confirmation, the LLM
must not set `pending_confirmation` for a *different* field without first
resolving the current one via `intent = confirm_yes | confirm_no | correction`

**What it catches:**
- Single-word STT artifacts ("new.", "now.") being extracted as optional field
  values (e.g. MetLife ID) while a required field confirmation is pending
- LLM jumping ahead to the next field's confirmation before the current yes/no
  is resolved

**What it does NOT block:**
- `result.extracted` — multi-field acquisition via `extraction_queue` is
  unaffected. Callers can give 3 fields in one utterance; those go to the queue
  and are confirmed one at a time in subsequent turns.

**Retry behaviour:** Injects a correction string and re-runs `dc_extract` once.
If the retry also violates, the result is applied as-is (the Python state machine
downstream prevents actual data corruption).

**Tests:** `tests/simulate_olas_new_as_no.py`, `tests/simulate_multifield_utterance.py`

---

### 2. `SchedulingAgent._slot_choice_bounds_guard`

**File:** `data-plane/app/agents/scheduling.py`  
**Trigger:** After `SlotChoice` LLM returns a slot index, before `slots_data[idx]` is accessed  
**Invariant:** The returned 1-based index must fall within `[1, len(available_slots)]`

**What it catches:**
- LLM returning slot index 4 when only 3 slots exist (common after narrow date
  searches reduce the available list)
- Negative indices
- Slot choice call on an empty slot list

**Retry behaviour:** Injects the valid range into the prompt and re-runs
`SlotChoice` once. If the retry index is still out of bounds, falls through to
the existing retry/re-present logic.

**Tests:** `tests/simulate_scheduling_steering.py`

---

### 3. `SchedulingAgent._booking_preflight`

**File:** `data-plane/app/agents/scheduling.py`  
**Trigger:** After `_detect_confirmation` returns `"confirm"`, before `book_time_slot()` is called  
**Invariant:** The slot about to be booked must be structurally complete and
must match what was presented to the caller for confirmation

**Checks (in order):**
1. `chosen_data` is not None — a slot is actually selected in state
2. `slot_id` is present and non-empty — Calendly API requires this URI
3. `start_time` is present and parses as a valid ISO datetime — prevents
   `datetime.fromisoformat` raising at booking time
4. `chosen_data["description"]` matches `pending_confirmation["slot"]` — the
   slot being booked is the one the caller confirmed, not a different slot at
   the same index position (catches LLM mis-indexing between choice and confirm)

**What it does NOT do:** This handler does not re-run the LLM. On any failure
it blocks `book_time_slot`, logs the reason at ERROR level, and resets
`stage = "awaiting_choice"` so the agent re-presents available slots to the
caller.

**Why no retry:** `book_time_slot` is the only irreversible action in the
pipeline. A retry that produces a different slot ID would be more dangerous than
resetting to a clean state.

**Tests:** `tests/simulate_scheduling_steering.py`

---

## Candidates Considered and Not Implemented

### `IntakeQualificationAgent` — thin-narrative guard

**Situation:** `IntakeQualificationAgent` is a one-shot auto-decision node. If
the caller provided very little narrative (< 15 words), the LLM may return
`decision = "not_qualified"` based on almost no information. This closes the
door on a caller who simply hadn't had a chance to explain their matter.

**Potential handler:** Before surfacing a `not_qualified` decision, check
`len(full_narrative.split()) < 15`. If so, override to `"ambiguous"` so the
caller is passed to scheduling rather than rejected.

**Why not implemented yet:** The narrative collection agent already uses
`_has_enough_content` (minimum segment count + word count) to gate completion.
A caller reaching qualification always has at least one meaningful segment.
The risk is low in practice. Worth revisiting if production transcripts show
thin-narrative rejections.

---

### `NarrativeCollectionAgent` — premature done-intent guard

**Situation:** `_detect_done_intent` returns True if the LLM thinks the caller
is finished. Combined with `_has_enough_content`, this gates narrative
completion. The LLM failing silently (exception) defaults to `False` — safe.

**Why not needed:** The Python `_has_enough_content` check already acts as a
deterministic guard on this path. The two conditions must both be met before
`COMPLETED` is returned. This is steering-in-place, already implemented.

---

### `Planner` — routing order guard

**Situation:** The planner decides which agent to invoke. A potential invariant:
never route to `scheduling` before `data_collection` is `COMPLETED`.

**Why not needed:** The workflow graph edges already enforce this. The graph
will not make `scheduling` reachable until `data_collection` completes.
Adding a steering check here would duplicate what the graph already guarantees.

---

## Design Principles

**One retry maximum.** Steering handlers that trigger a retry do so once only.
If the second result also violates, apply it and let the Python state machine
handle any corruption downstream. Infinite retry loops are worse than a single
bad LLM turn.

**Zero latency on the happy path.** Every handler is a pure Python function.
No LLM call, no I/O. On the happy path (invariant holds), the overhead is
effectively zero.

**Log violations at WARNING or ERROR.** Steering violations are signal — they
indicate either a prompt gap or an LLM reliability issue on specific input
patterns. Keep the log structured enough to aggregate in production.

**Don't steer text generation.** Handlers that validate structured outputs
(JSON fields, indices, stage values) are appropriate. Handlers that try to
validate free-form speak text (tone, length, question count) are fragile and
better addressed in the prompt.
