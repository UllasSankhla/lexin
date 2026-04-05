"""Call planner — replaces single-agent routing with a multi-step execution plan.

Instead of routing the caller's utterance to exactly one agent, the Planner:
1. Reasons about the full conversation state (available agents, booking stages,
   already-collected fields, conversation history).
2. Classifies the utterance type (direct answer, follow-up, correction, etc.).
3. Emits an ordered execution plan: reset_fields and/or invoke steps.
4. Provides smart speak combination when multiple agents respond.

Why this is better than the Router:
- Multi-intent utterances ("my name is John and I'm calling about a work injury")
  are handled by BOTH data_collection and narrative_collection in one turn.
- Field corrections ("actually my email is john@gmail.com not jane@gmail.com")
  use reset_fields to cleanly re-open the field before re-invoking data_collection,
  instead of fighting the existing confirmed-state logic.
- Utterance classification detects follow-ups and continuations so the planner
  can route appropriately (e.g. a follow-up to a narrative stays in narrative).
- Smart speak combination merges FAQ answers with data questions naturally.

Backward compatibility:
- Planner exposes pop_resume() matching Router's interface.
- Handler replaces router.select() with planner.plan() and iterates steps.
- All edge chains, tool triggers, and resume-stack logic remain in handler.py.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

from app.agents.base import AgentStatus
from app.agents.llm_utils import llm_structured_call
from app.agents.agent_schemas import IntentItem, MultiIntentLLMResponse
from app.agents.workflow import WorkflowGraph

logger = logging.getLogger(__name__)

_MAX_PLAN_STEPS = 3

# Agent IDs whose speaks are "filler" / acknowledgement rather than task-complete
_FILLER_AGENTS = {"narrative_collection"}
# Agents whose speak is an answer to a question (not a question themselves)
_ANSWER_AGENTS = {"faq", "context_docs", "fallback"}
# Agent IDs whose speaks are empathetic acknowledgments (no questions, no data)
_EMPATHY_AGENTS = {"empathy"}


# ── Multi-intent system prompt ────────────────────────────────────────────────

_MULTI_INTENT_SYSTEM = """\
You are an intent classifier for a voice appointment booking assistant at a law firm.

Given the caller's most recent utterance AND the conversation history, identify every
distinct communicative intent in the utterance, in the ORDER they appear in speech.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INTENT TYPES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FIELD_DATA    — Caller provides one or more contact or intake data values: name, email,
                phone, address, employer, member ID, job title, date, etc.
                Multiple field values in one utterance → still one FIELD_DATA intent.

CONFIRMATION  — Caller responds yes/no to a pending read-back or confirmation question.
                Only classify as CONFIRMATION when PENDING CONFIRMATION is shown in
                context. Without one, "yes" is CONTINUATION or FIELD_DATA.

CORRECTION    — Caller explicitly corrects a previously confirmed contact field.
                Must include "field": the exact key from COLLECTED STATE being corrected.

NARRATIVE     — Caller describes their legal situation or circumstances.
                Rhetorical questions appended to a narrative ("can you help?",
                "am I in the right place?") are NARRATIVE, not FAQ_QUESTION.

FAQ_QUESTION  — Caller asks a concrete standalone question about fees, process,
                location, hours, what the firm handles, or firm policy.
                NOT FAQ_QUESTION if the question is rhetorical or embedded in narrative.

DATA_STATUS   — Caller asks about the data collection state: what has been collected,
                whether a field was captured correctly, or requests a read-back.
                e.g. "What information do you have so far?", "Did you get my name right?"
                These are NOT FAQ_QUESTION — route to data_collection regardless of stage.

CONTINUATION  — Caller continues or adds detail to their previous statement in direct
                response to the AI's last acknowledgment or prompt.

FAREWELL      — Caller is unmistakably signing off with explicit goodbye words.
                NOT farewell: "Okay. Thank you.", "Sounds good", "Perfect".

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CLASSIFICATION RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. HISTORY FIRST — the same words mean different things depending on what the AI last
   asked. Always read the conversation history before classifying.

2. SPEECH ORDER — list intents in the order the caller expressed them.
   Most utterances have exactly 1 intent. Use 2 only when clearly distinct
   communicative parts exist in sequence. Use 3 only for three distinct acts.

3. FAREWELL terminates — if present, it must be the last intent in the list.

4. CORRECTION requires "field" — the exact key from COLLECTED STATE.
   If the corrected field cannot be identified with confidence, use FIELD_DATA.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXAMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AI: "Is that correct?"  Caller: "Yes"  (PENDING CONFIRMATION shown in context)
  → [CONFIRMATION]

AI: "Tell me more."  Caller: "Nothing formal yet, just hints from my manager."
  → [CONTINUATION]  (continuing in response to AI prompt — not a new topic)

AI: "What is your email?"  Caller: "john at gmail dot com. Also my phone is 415-555-0192."
  → [FIELD_DATA]  (two fields in one utterance — still one FIELD_DATA intent)

AI: "What brings you to us today?"
Caller: "I was in a car accident last month. What are your fees?"
  → [NARRATIVE, FAQ_QUESTION]  (narrative first, then concrete question)

Caller: "No wait, my email is john@gmail.com not jane@gmail.com"
  (email_address in COLLECTED STATE)
  → [CORRECTION field=email_address]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESPONSE FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Return ONLY valid JSON. No markdown. No explanation.

{
  "intents": [
    {"type": "NARRATIVE", "field": null, "reason": "caller described their situation"},
    {"type": "FAQ_QUESTION", "field": null, "reason": "asked about fees"}
  ]
}
"""


# ── Plan models ────────────────────────────────────────────────────────────────

@dataclass
class PlanStep:
    """A single step in the execution plan."""
    action: str                     # "invoke" | "reset_fields"
    agent_id: str | None = None     # target agent for "invoke"
    fields: list[str] | None = None # field keys to clear for "reset_fields"
    reason: str = ""                # planner's stated reasoning (logged)
    use_empty_utterance: bool = False  # invoke with "" instead of caller_text (re-surface pending)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _join(a: str, b: str) -> str:
    """Join two speak strings with natural sentence separation."""
    a = a.rstrip()
    b = b.strip()
    if not b:
        return a
    sep = " " if a.endswith((".", "?", "!")) else ". "
    return f"{a}{sep}{b}"


# ── Planner class ──────────────────────────────────────────────────────────────

class Planner:
    """
    Generates a multi-step execution plan for each caller utterance.

    Usage in handler.py (replaces router.select() + first _invoke_and_follow call):

        # Old:
        agent_id, interrupt = router.select(utterance, recent_history)
        speak, fr = await _invoke_and_follow(agent_id, utterance, ...)

        # New:
        steps = await loop.run_in_executor(None, lambda: planner.plan(utterance, recent_history))
        speaks: list[tuple[str, str]] = []
        for step in steps:
            if step.action == "reset_fields":
                planner.reset_fields(step.fields, collected_all)
            elif step.action == "invoke":
                speak, fr = await _invoke_and_follow(step.agent_id, utterance, ...)
                if speak:
                    speaks.append((speak, step.agent_id))
                if fr is not None:
                    break
        speak_text = planner.combine_speaks(speaks)
    """

    def __init__(self, graph: WorkflowGraph) -> None:
        self._graph = graph
        self._resume_stack: list[str] = []

    # ── Public API ─────────────────────────────────────────────────────────────

    def plan(
        self,
        utterance: str,
        recent_history: list[dict],
    ) -> list[PlanStep]:
        """
        Classify the caller's utterance into an ordered list of intents (speech order),
        then deterministically map each intent to agent invocations.

        The LLM identifies what the caller meant and in what order; Python builds
        the execution plan from that — keeping routing logic out of the LLM.

        Falls back to a single-step plan targeting the next primary goal if the
        LLM call fails.
        """
        graph = self._graph

        available = graph.available_nodes()
        if not available:
            logger.warning("Planner: no available agents — defaulting to data_collection")
            return [PlanStep(action="invoke", agent_id="data_collection", reason="no_available_agents")]

        available_ids = {n.id for n in available}

        # Build context blocks
        available_block = graph.available_summary()
        status_block = graph.status_summary()
        booking_block = graph.primary_goal_summary()

        next_goal_id = graph.next_primary_goal()
        next_goal_line = ""
        if next_goal_id:
            node = graph.nodes[next_goal_id]
            state = graph.states[next_goal_id]
            next_goal_line = (
                f"\nNEXT PRIMARY GOAL: {next_goal_id} [{state.status.value}]\n"
                f"  {node.description}\n"
            )

        # Collected fields — shown so the LLM can identify CORRECTION intents
        dc_state = graph.states.get("data_collection")
        collected_fields = dc_state.internal_state.get("collected", {}) if dc_state else {}
        collected_line = (
            "\nCOLLECTED STATE (confirmed contact fields — use exact keys for CORRECTION.field):\n"
            + (
                "\n".join(f"  {k}: {v!r}" for k, v in collected_fields.items())
                or "  (none yet)"
            )
            + "\n"
        )

        # Pending confirmation — critical for CONFIRMATION intent classification
        pending_line = ""
        if dc_state:
            pending = dc_state.internal_state.get("pending_confirmation")
            if pending:
                field_name = pending.get("field", "unknown")
                value = pending.get("value", "unknown")
                pending_line = (
                    f"\nPENDING CONFIRMATION: data_collection is awaiting caller's yes/no "
                    f"for field '{field_name}' = {value!r}. "
                    f"A yes/correct/right response is a CONFIRMATION intent.\n"
                )

        # History: last 6 turns, full content for 3 most recent
        history_lines = self._format_history(recent_history)

        # Last AI message — critical for disambiguating partial answers and continuations
        last_ai_line = ""
        for turn in reversed(recent_history):
            if turn.get("role") == "assistant":
                last_ai_line = f"\nLAST AI SAID:\n  {turn['content']}\n"
                break

        user_msg = (
            f"BOOKING STAGES:\n{booking_block}\n\n"
            f"AVAILABLE AGENTS:\n{available_block}\n\n"
            f"AGENT STATUS:\n{status_block}\n"
            f"{next_goal_line}"
            f"{collected_line}"
            f"{pending_line}"
            f"CONVERSATION HISTORY (most recent last):\n{history_lines}\n"
            f"{last_ai_line}\n"
            f'CALLER JUST SAID: "{utterance}"\n\n'
            "Identify intents in speech order. Reply JSON only."
        )

        try:
            result = llm_structured_call(
                _MULTI_INTENT_SYSTEM, user_msg, MultiIntentLLMResponse, max_tokens=256, tag="planner"
            )

            steps = self._build_steps_from_intents(result.intents, available_ids)

            if not any(s.action == "invoke" for s in steps):
                logger.warning("Intent mapping produced no invoke steps — using fallback")
                steps = self._fallback_plan(available_ids)

            steps = self._validate_waiting_confirm(steps)

            logger.info(
                "Planner | intents=%s | steps=%s | utterance=%r",
                [(i.type, i.field) for i in result.intents],
                [(s.action, s.agent_id or s.fields) for s in steps],
                utterance[:100],
            )
            return steps

        except Exception as exc:
            logger.error("Planner LLM call failed: %s — using fallback plan", exc)
            fallback = self._fallback_plan(available_ids)
            return self._validate_waiting_confirm(fallback)

    def combine_speaks(self, speaks: list[tuple[str, str, float]]) -> str:
        """
        Combine speaks from multiple agents in speech order into a single response.

        speaks: list of (speak_text, agent_id, confidence) triples in execution order
                (same order as the caller's intents).

        Rules:
        1. Single speak: return as-is.
        2. Empathy + anything: empathy text first, then other speak joined naturally.
           Exception: empathy + narrative_collection filler → empathy alone
           (the narrative agent has nothing to add that empathy hasn't covered).
        3. All other combinations: join in execution order with natural sentence
           joining. Agents speak in the same order the caller raised each topic.
        """
        if not speaks:
            return ""
        if len(speaks) == 1:
            return speaks[0][0]

        # Filter out empty speaks before combining
        non_empty = [(t, i, c) for t, i, c in speaks if t and t.strip()]
        if not non_empty:
            return ""
        if len(non_empty) == 1:
            return non_empty[0][0]

        # Empathy first — always keep, but collapse empathy + narrative filler
        if non_empty[0][1] in _EMPATHY_AGENTS and len(non_empty) == 2:
            a_text, _, _ = non_empty[0]
            b_text, b_id, _ = non_empty[1]
            if b_id in _FILLER_AGENTS:
                return a_text  # narrative filler after empathy adds nothing
            return _join(a_text, b_text)

        # General: join all speaks in speech order
        result = non_empty[0][0]
        for speak_text, _, _ in non_empty[1:]:
            result = _join(result, speak_text)
        return result

    def reset_fields(self, fields: list[str], collected: dict) -> None:
        """
        Clear fields from the shared collected dict AND from data_collection's
        internal state so data_collection re-asks for them cleanly.

        If data_collection was COMPLETED, resets it to IN_PROGRESS so it
        becomes available for re-invocation.
        """
        for field_name in fields:
            removed = collected.pop(field_name, None)
            if removed is not None:
                logger.info("Planner: reset_fields — cleared %r (was %r)", field_name, removed)

        dc_state = self._graph.states.get("data_collection")
        if not dc_state:
            return

        internal = dc_state.internal_state
        dc_collected = internal.get("collected", {})
        for field_name in fields:
            dc_collected.pop(field_name, None)
        internal["collected"] = dc_collected

        # Clear pending confirmation if it involves a reset field
        pending = internal.get("pending_confirmation")
        if pending and pending.get("field") in fields:
            internal["pending_confirmation"] = None
            logger.info("Planner: reset_fields — cleared pending_confirmation for %s", fields)

        # Re-open data_collection if it was COMPLETED
        if dc_state.status == AgentStatus.COMPLETED:
            dc_state.status = AgentStatus.IN_PROGRESS
            logger.info(
                "Planner: reset_fields — data_collection COMPLETED → IN_PROGRESS "
                "to allow re-collection of %s", fields,
            )

    def pop_resume(self) -> str | None:
        """Pop and return the top of the resume stack (Router-compatible interface)."""
        if self._resume_stack:
            agent_id = self._resume_stack.pop()
            logger.info("Planner: resuming %s from stack", agent_id)
            return agent_id
        return None

    def push_resume(self, agent_id: str) -> None:
        """Push an agent onto the resume stack (called by handler on interrupt)."""
        if agent_id not in self._resume_stack:
            self._resume_stack.append(agent_id)
            logger.info("Planner: pushed %s onto resume stack", agent_id)

    def active_primary_for_resume(self) -> str | None:
        """Return the current active primary goal agent, for interrupt push."""
        state = self._graph.active_primary()
        return state.node_id if state else None

    # ── Private ────────────────────────────────────────────────────────────────

    def _build_steps_from_intents(
        self,
        intents: list[IntentItem],
        available_ids: set[str],
    ) -> list[PlanStep]:
        """
        Deterministically map an ordered list of intents to PlanSteps.

        Each intent produces one or two steps (CORRECTION produces reset_fields +
        invoke; NARRATIVE may prepend empathy). Steps are ordered to match the
        caller's speech order. FAREWELL terminates — no further steps after it.
        Consecutive duplicate data_collection invocations are collapsed to one.
        """
        graph = self._graph
        steps: list[PlanStep] = []
        last_invoke_agent: str | None = None

        for intent in intents:
            t = intent.type

            if t in ("FIELD_DATA", "CONFIRMATION"):
                if "data_collection" in available_ids and last_invoke_agent != "data_collection":
                    steps.append(PlanStep(action="invoke", agent_id="data_collection", reason=t))
                    last_invoke_agent = "data_collection"

            elif t == "CORRECTION":
                field = intent.field
                if field:
                    steps.append(PlanStep(action="reset_fields", fields=[field], reason="CORRECTION"))
                # data_collection may be COMPLETED (excluded from available_ids) but the
                # reset_fields step above will reopen it — always invoke it for CORRECTION.
                if "data_collection" in available_ids or "data_collection" in self._graph.nodes:
                    steps.append(PlanStep(action="invoke", agent_id="data_collection", reason="CORRECTION"))
                    last_invoke_agent = "data_collection"

            elif t == "NARRATIVE":
                # Prepend empathy if it hasn't been used this call
                if "empathy" in available_ids:
                    steps.append(PlanStep(action="invoke", agent_id="empathy", reason="NARRATIVE"))
                    last_invoke_agent = "empathy"
                if "narrative_collection" in available_ids:
                    steps.append(PlanStep(action="invoke", agent_id="narrative_collection", reason="NARRATIVE"))
                    last_invoke_agent = "narrative_collection"

            elif t == "DATA_STATUS":
                if "data_collection" in available_ids and last_invoke_agent != "data_collection":
                    steps.append(PlanStep(action="invoke", agent_id="data_collection", reason="DATA_STATUS"))
                    last_invoke_agent = "data_collection"

            elif t == "FAQ_QUESTION":
                for faq_agent in ("faq", "context_docs", "fallback"):
                    if faq_agent in available_ids:
                        steps.append(PlanStep(action="invoke", agent_id=faq_agent, reason="FAQ_QUESTION"))
                        last_invoke_agent = faq_agent
                        break

            elif t == "CONTINUATION":
                # Route to the currently active primary agent, or narrative_collection
                active = graph.active_primary()
                target = active.node_id if active else None
                if not target or target not in available_ids:
                    target = next(
                        (a for a in ("narrative_collection", "data_collection") if a in available_ids),
                        None,
                    )
                if target and target != last_invoke_agent:
                    steps.append(PlanStep(action="invoke", agent_id=target, reason="CONTINUATION"))
                    last_invoke_agent = target

            elif t == "FAREWELL":
                if "farewell" in available_ids:
                    steps.append(PlanStep(action="invoke", agent_id="farewell", reason="FAREWELL"))
                break  # FAREWELL terminates — nothing after a sign-off

            if len([s for s in steps if s.action == "invoke"]) >= _MAX_PLAN_STEPS:
                break

        return steps

    def _format_history(self, recent_history: list[dict]) -> str:
        """
        Format conversation history for the planner context.
        - Last 3 turns: full content (no truncation)
        - Earlier turns (up to 6 total): truncated at 200 chars
        """
        if not recent_history:
            return "  (none)"

        turns = recent_history[-6:]  # at most 6 turns
        recent_cutoff = max(0, len(turns) - 3)  # last 3 are full
        lines = []
        for i, t in enumerate(turns):
            role = t.get("role", "unknown")
            content = t.get("content", "")
            if i >= recent_cutoff:
                # Full content for the 3 most recent turns
                lines.append(f"  {role}: {content}")
            else:
                # Truncated for earlier turns
                truncated = content[:200] + ("..." if len(content) > 200 else "")
                lines.append(f"  {role}: {truncated}")
        return "\n".join(lines)

    def _validate_waiting_confirm(self, steps: list[PlanStep]) -> list[PlanStep]:
        """
        Post-plan safety net: ensure WAITING_CONFIRM agent is in every plan.

        Three cases:
        1. Waiting agent is FIRST invoke step: caller is directly responding to the
           confirmation — pass through the utterance (use_empty_utterance=False).
        2. Waiting agent appears AFTER other invoke steps: those agents handle their
           intent first, then the waiting agent re-asks its pending question
           (use_empty_utterance=True).
        3. Waiting agent is missing entirely: inject it at the end as a re-surface
           (use_empty_utterance=True).
        """
        waiting_id = self._graph.active_waiting_confirm()
        if not waiting_id:
            return steps

        invoke_steps = [s for s in steps if s.action == "invoke"]

        # Never inject after a farewell — the call is ending, re-surfacing a pending
        # confirmation into a terminated call produces a speak with nowhere to go.
        if any(s.agent_id == "farewell" for s in invoke_steps):
            return steps

        waiting_step = next((s for s in invoke_steps if s.agent_id == waiting_id), None)

        if waiting_step is None:
            # Missing — inject at end so caller's confirmation is re-surfaced after other intents
            logger.info(
                "Planner validator: WAITING_CONFIRM (%s) missing from plan — injecting",
                waiting_id,
            )
            steps.append(PlanStep(
                action="invoke",
                agent_id=waiting_id,
                reason="waiting_confirm_reinjected",
                use_empty_utterance=True,
            ))
        elif invoke_steps[0].agent_id != waiting_id:
            # Waiting agent is not first — it's a re-surface after other intents are handled
            waiting_step.use_empty_utterance = True
        # else: waiting agent IS first → caller is responding directly, pass utterance through

        return steps

    def _fallback_plan(self, available_ids: set[str]) -> list[PlanStep]:
        """Fallback when the LLM call fails or produces no invoke steps."""
        for preferred in ("data_collection", "narrative_collection", "scheduling", "faq", "fallback"):
            if preferred in available_ids:
                return [PlanStep(action="invoke", agent_id=preferred, reason="fallback")]
        fallback_id = next(iter(available_ids))
        return [PlanStep(action="invoke", agent_id=fallback_id, reason="fallback")]
