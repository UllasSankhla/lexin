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
from app.agents.llm_utils import llm_json_call
from app.agents.workflow import WorkflowGraph

logger = logging.getLogger(__name__)

_MAX_PLAN_STEPS = 3

# Agent IDs whose speaks are "filler" / acknowledgement rather than task-complete
_FILLER_AGENTS = {"narrative_collection"}
# Agents whose speak is an answer to a question (not a question themselves)
_ANSWER_AGENTS = {"faq", "context_docs", "fallback"}
# Agent IDs whose speaks are empathetic acknowledgments (no questions, no data)
_EMPATHY_AGENTS = {"empathy"}


# ── Pre-routing utterance classification ──────────────────────────────────────

UtteranceClass = Literal["FIELD_DATA", "LEGAL_NARRATIVE", "BOTH", "CONTROL"]

_CLASSIFY_SYSTEM = """\
You are a pre-routing classifier for a voice intake assistant at a law firm.

Your sole job: given the caller's latest utterance and the last few turns of conversation,
classify the utterance into exactly ONE of these four categories:

FIELD_DATA     — The caller is providing structured contact or intake information:
                 name, phone number, email address, physical address, employer name,
                 job title, member ID, date, or any other specific data field.
                 Includes spelled-out values, partial values, and corrections to fields.
                 Examples: "My name is Sarah Chen", "It's 415-555-0192",
                 "sarah dot chen at gmail dot com", "VW3R9KJ2", "I work at Microsoft"

LEGAL_NARRATIVE — The caller is describing their legal situation, matter, or circumstances.
                 Includes describing what happened, their current legal status, concerns,
                 questions about their case, or any free-form account of their issue.
                 Examples: "I was in a car accident last month",
                 "I'm on an H1B and my employer is hinting at layoffs",
                 "Nothing formal yet, just verbal hints, I want to know my options"

BOTH           — The utterance contains BOTH structured field data AND legal narrative.
                 Examples: "My name is Kavita Sharma and I have an H1B visa issue",
                 "I need help with an employment matter, I'm Marcus Rivera at microsoft"

CONTROL        — A short yes/no/confirmation/correction/stop signal with no new information.
                 Examples: "Yes", "No", "That's correct", "Yes that's right",
                 "Actually no", "Stop", "Cancel", "Goodbye"

Rules:
- Use the conversation history to understand context. A short utterance like
  "Nothing formal yet" is LEGAL_NARRATIVE if the prior turns show the caller
  was describing a legal matter.
- When the caller provides any specific data value (ID, number, name, email) AND
  describes their situation in the same utterance, classify as BOTH.
- Default to FIELD_DATA when unsure — it is the safer fallback.

Respond with ONLY valid JSON: {"class": "FIELD_DATA"} (or LEGAL_NARRATIVE, BOTH, CONTROL)
No explanation, no other keys.
"""


def classify_utterance(utterance: str, recent_history: list[dict] | None = None) -> UtteranceClass:
    """
    LLM-based pre-routing classification with conversation history context.

    Uses the last 4 turns of history so ambiguous short utterances (e.g.
    "Nothing formal yet") are classified correctly based on what came before.

    Returns one of: FIELD_DATA | LEGAL_NARRATIVE | BOTH | CONTROL

    Falls back to FIELD_DATA on any error — safe default since data_collection
    handles unrecognised utterances via its CANNOT_PROCESS path.
    """
    # Build a compact history block (last 4 turns only)
    history_block = ""
    if recent_history:
        last_turns = recent_history[-4:]
        lines = []
        for t in last_turns:
            role = "Caller" if t.get("role") == "user" else "Assistant"
            content = (t.get("content") or "")[:200]
            lines.append(f"{role}: {content}")
        if lines:
            history_block = "RECENT CONVERSATION:\n" + "\n".join(lines) + "\n\n"

    user_msg = f'{history_block}CALLER JUST SAID: "{utterance}"\n\nClassify this utterance.'

    try:
        result = llm_json_call(_CLASSIFY_SYSTEM, user_msg, max_tokens=32)
        raw = result.get("class", "FIELD_DATA")
        if raw in ("FIELD_DATA", "LEGAL_NARRATIVE", "BOTH", "CONTROL"):
            return raw  # type: ignore[return-value]
        logger.warning("classify_utterance: unexpected class %r — defaulting to FIELD_DATA", raw)
        return "FIELD_DATA"
    except Exception as exc:
        logger.warning("classify_utterance LLM call failed: %s — defaulting to FIELD_DATA", exc)
        return "FIELD_DATA"


# ── Plan models ────────────────────────────────────────────────────────────────

@dataclass
class PlanStep:
    """A single step in the execution plan."""
    action: str                     # "invoke" | "reset_fields"
    agent_id: str | None = None     # target agent for "invoke"
    fields: list[str] | None = None # field keys to clear for "reset_fields"
    reason: str = ""                # planner's stated reasoning (logged)
    use_empty_utterance: bool = False  # invoke with "" instead of caller_text (re-surface pending)


# ── Planner system prompt ──────────────────────────────────────────────────────

_PLANNER_SYSTEM = """\
You are an execution planner for a voice appointment booking assistant.

Given the caller's most recent utterance and full conversation context, you will:
  STEP 1: Classify the utterance type
  STEP 2: Create an ordered EXECUTION PLAN

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1: CLASSIFY THE UTTERANCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Choose the single best type from:

DIRECT_ANSWER   — Caller is directly answering what the AI just asked.
                  e.g. AI asked "What is your name?" → caller says "John Smith"
                  e.g. AI asked "Is that correct?" → caller says "Yes" or "No"

FOLLOW_UP       — Caller adds to, clarifies, or references their previous message.
                  e.g. Caller said "John" before, now says "Actually it's John Smith"

CONTINUATION    — Caller continues a narrative or description they were giving.
                  e.g. AI acknowledged their story, caller is adding more details

CORRECTION      — Caller explicitly corrects a previously confirmed contact field.
                  e.g. "Wait, my email is john@gmail.com not jane@gmail.com"
                  e.g. "Actually spell my name as J-O-H-N"

NEW_TOPIC       — Caller changes subject or asks an unrelated question.
                  e.g. After providing info, caller asks "What are your fees?"

NARRATIVE       — Caller is describing their legal situation for the first time.
                  e.g. "I was in a car accident last month..."

FAREWELL        — Caller is clearly ending the call after all goals are complete.
                  e.g. "Goodbye!", "Thanks, have a great day!", "Talk to you then, bye!"
                  NOT farewell: "That would be great, thank you" (accepting a slot offer mid-call)
                  NOT farewell: "Sounds good" or "Perfect" mid-conversation
                  NOT farewell: any gratitude expressed while scheduling is still in progress.
                  Use FAREWELL only when no primary goals remain (scheduling complete or not needed)
                  AND the caller is clearly signing off.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AVAILABLE ACTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"invoke"       — Run an agent with the caller's utterance. Each agent
                 independently extracts what is relevant to it.

"reset_fields" — Clear one or more already-collected contact fields so
                 they can be corrected. Use ONLY when the caller explicitly
                 corrects a previously confirmed value. Must be immediately
                 followed by invoke("data_collection"). Use the exact field
                 key shown in COLLECTED STATE (e.g. "full_name", "email_address").

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 2: PLANNING RULES (check in order)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. FAREWELL — If utterance_type=FAREWELL AND no primary goals are still in progress
   (i.e. scheduling has COMPLETED or "scheduling" is not in AVAILABLE AGENTS), plan
   ONLY: [invoke("farewell")].
   If scheduling is still in progress (stage not done), treat the utterance as a
   DIRECT_ANSWER to the scheduling agent instead — do NOT invoke farewell mid-booking.

2. EMPATHY — If utterance_type=NARRATIVE and "empathy" is in AVAILABLE AGENTS
   (meaning it has not yet been used this call), prepend an empathy step:
   invoke("empathy") → invoke("narrative_collection") [or "data_collection" if
   narrative_collection is not yet available].
   Skip this rule if "empathy" is NOT listed in AVAILABLE AGENTS.

3. PENDING CONFIRM — If PENDING CONFIRMATION is shown and utterance_type=DIRECT_ANSWER
   (caller is saying yes/no/confirm/correct), plan: [invoke("data_collection")].
   The data_collection agent will handle the confirmation response.

4. CORRECTION — If utterance_type=CORRECTION and a contact field was corrected,
   plan: reset_fields([field_key]) → invoke("data_collection")

5. DIRECT_ANSWER to narrative — If utterance_type=DIRECT_ANSWER and the AI last
   asked about the caller's legal situation or story, plan: [invoke("narrative_collection")].

6. CONTINUATION — If utterance_type=CONTINUATION, route to the same agent the AI
   was last using. If that agent is unavailable, route to narrative_collection.

7. MULTI-INTENT — If the utterance contains BOTH contact data (name, phone, email)
   AND a description of a legal matter or situation, and "empathy" is NOT available:
   invoke("data_collection") → invoke("narrative_collection")
   If "empathy" IS available: invoke("empathy") → invoke("data_collection") → invoke("narrative_collection")

8. NEW_TOPIC / QUESTION — If utterance_type=NEW_TOPIC and caller is asking about
   fees, process, location, or firm policy: invoke("faq"), invoke("context_docs"),
   or invoke("fallback") as appropriate.

9. GOAL PURSUIT — Always advance toward the NEXT PRIMARY GOAL. If no interrupt-worthy
   question is present, route to the NEXT PRIMARY GOAL agent.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONSTRAINTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- NEVER invoke "intake_qualification" directly — it auto-runs via edge chain.
- NEVER plan more than 3 steps.
- Always include at least one invoke step.
- Only use agents listed in AVAILABLE AGENTS.
- reset_fields must always be immediately followed by invoke("data_collection").
- Use the UTTERANCE CLASS pre-routing signal as a strong prior:
  LEGAL_NARRATIVE → prefer narrative_collection over data_collection.
  FIELD_DATA → prefer data_collection.
  BOTH → invoke data_collection then narrative_collection (or empathy first if available).
  CONTROL → short yes/no — follow PENDING CONFIRM and DIRECT_ANSWER rules.

Respond ONLY with valid JSON:
{
  "thinking": "<reasoning about utterance type and plan choice>",
  "utterance_type": "<DIRECT_ANSWER|FOLLOW_UP|CONTINUATION|CORRECTION|NEW_TOPIC|NARRATIVE|FAREWELL>",
  "steps": [
    {"action": "invoke", "agent_id": "<id>", "fields": null, "reason": "<brief>"},
    {"action": "reset_fields", "agent_id": null, "fields": ["field_key"], "reason": "<brief>"}
  ]
}
"""


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
        self.last_utterance_class: UtteranceClass = "FIELD_DATA"

    # ── Public API ─────────────────────────────────────────────────────────────

    def plan(
        self,
        utterance: str,
        recent_history: list[dict],
    ) -> list[PlanStep]:
        """
        Generate an execution plan for this utterance.

        Returns an ordered list of PlanSteps. The handler executes each step
        using its own _invoke_and_follow, keeping all infrastructure concerns
        (tools, DB persist, TTS, edge chains) in the handler.

        Falls back to a single-step plan targeting the next primary goal if the
        LLM call fails.
        """
        graph = self._graph

        # Pre-routing classification — LLM call with last 4 turns of history.
        utt_class = classify_utterance(utterance, recent_history)
        self.last_utterance_class = utt_class

        # WAITING_CONFIRM enforcement — relaxed for LEGAL_NARRATIVE utterances.
        # For field data and control signals, keep the hard bypass.
        # For legal narratives, route to narrative_collection first so the caller
        # can share their story, then re-surface the pending confirmation.
        waiting_id = graph.active_waiting_confirm()
        if waiting_id:
            if utt_class == "LEGAL_NARRATIVE":
                available_ids = {n.id for n in graph.available_nodes()}
                steps: list[PlanStep] = []
                if "narrative_collection" in available_ids:
                    steps.append(PlanStep(
                        action="invoke",
                        agent_id="narrative_collection",
                        reason="narrative_before_confirm",
                    ))
                steps.append(PlanStep(
                    action="invoke",
                    agent_id=waiting_id,
                    reason="waiting_confirm_follow_up",
                    use_empty_utterance=True,
                ))
                logger.info(
                    "Planner: WAITING_CONFIRM active but LEGAL_NARRATIVE — "
                    "routing narrative_collection first, then %s", waiting_id,
                )
                return steps
            else:
                logger.info(
                    "Planner: WAITING_CONFIRM enforced → %s (class=%s, LLM bypassed)",
                    waiting_id, utt_class,
                )
                return [PlanStep(action="invoke", agent_id=waiting_id, reason="waiting_confirm")]

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

        # Collected fields for correction detection
        collected_fields = {}
        dc_state = graph.states.get("data_collection")
        if dc_state:
            collected_fields = dc_state.internal_state.get("collected", {})
        collected_line = (
            "\nCOLLECTED STATE (confirmed contact fields — use exact keys in reset_fields):\n"
            + (
                "\n".join(f"  {k}: {v!r}" for k, v in collected_fields.items())
                or "  (none yet)"
            )
            + "\n"
        )

        # Pending confirmation context
        pending_line = ""
        if dc_state:
            pending = dc_state.internal_state.get("pending_confirmation")
            if pending:
                field_name = pending.get("field", "unknown")
                value = pending.get("value", "unknown")
                pending_line = (
                    f"\nPENDING CONFIRMATION: data_collection is awaiting caller confirmation "
                    f"for field '{field_name}' = {value!r}. "
                    f"If the caller says yes/correct/right, invoke data_collection.\n"
                )

        # History: last 6 turns, full content for the 3 most recent
        history_lines = self._format_history(recent_history)

        # Last AI message — show explicitly for follow-up/continuation detection
        last_ai_line = ""
        for turn in reversed(recent_history):
            if turn.get("role") == "assistant":
                last_ai_content = turn["content"]
                last_ai_line = f"\nLAST AI SAID:\n  {last_ai_content}\n"
                break

        user_msg = (
            f"AVAILABLE AGENTS:\n{available_block}\n\n"
            f"AGENT STATUS:\n{status_block}\n"
            f"{collected_line}"
            f"{pending_line}"
            f"{next_goal_line}"
            f"BOOKING STAGES:\n{booking_block}\n\n"
            f"CONVERSATION HISTORY (most recent last):\n{history_lines}\n"
            f"{last_ai_line}\n"
            f"UTTERANCE CLASS (pre-routing signal): {utt_class}\n"
            f"  FIELD_DATA=structured contact data, LEGAL_NARRATIVE=legal situation description,\n"
            f"  BOTH=contains both, CONTROL=short yes/no/stop signal\n"
            f"  Use this as a strong prior when deciding between data_collection and narrative_collection.\n\n"
            f'CALLER JUST SAID: "{utterance}"\n\n'
            "Classify the utterance and create an execution plan. Reply JSON only."
        )

        try:
            result = llm_json_call(_PLANNER_SYSTEM, user_msg, max_tokens=2048)
            thinking = result.get("thinking", "")
            utterance_type = result.get("utterance_type", "UNKNOWN")
            steps_raw = result.get("steps", [])

            steps: list[PlanStep] = []
            for s in steps_raw[:_MAX_PLAN_STEPS]:
                action = s.get("action", "")
                if action == "invoke":
                    agent_id = s.get("agent_id")
                    if agent_id and agent_id in available_ids:
                        steps.append(PlanStep(
                            action="invoke",
                            agent_id=agent_id,
                            reason=s.get("reason", ""),
                        ))
                    elif agent_id:
                        logger.warning(
                            "Planner returned unavailable agent %r — skipping step", agent_id
                        )
                elif action == "reset_fields":
                    fields = s.get("fields") or []
                    if fields:
                        steps.append(PlanStep(
                            action="reset_fields",
                            fields=fields,
                            reason=s.get("reason", ""),
                        ))

            if not any(s.action == "invoke" for s in steps):
                logger.warning("Planner produced no invoke steps — using fallback")
                steps = self._fallback_plan(available_ids)

            logger.info(
                "Planner | type=%s | thinking=%r | steps=%s | utterance=%r",
                utterance_type,
                thinking,
                [(s.action, s.agent_id or s.fields) for s in steps],
                utterance[:100],
            )
            return steps

        except Exception as exc:
            logger.error("Planner LLM call failed: %s — using fallback plan", exc)
            return self._fallback_plan(available_ids)

    def combine_speaks(self, speaks: list[tuple[str, str, float]]) -> str:
        """
        Intelligently combine speaks from multiple agents into a single response.

        speaks: list of (speak_text, agent_id, confidence) triples in execution order.

        Combination rules (checked in order):
        1. Single speak: return as-is.
        2. Named agent-type rules (empathy, answer+data, filler+data, etc.).
        3. Confidence-based selection: if one agent has confidence > other by 0.3+,
           prefer the higher-confidence speak (for parallel BOTH invocations).
        4. General: join all non-empty speaks with a space.
        """
        if not speaks:
            return ""
        if len(speaks) == 1:
            return speaks[0][0]

        texts = [s[0] for s in speaks]
        ids = [s[1] for s in speaks]

        # Two speaks: apply smart rules
        if len(speaks) == 2:
            a_text, a_id, a_conf = speaks[0]
            b_text, b_id, b_conf = speaks[1]

            # Empathy acknowledgment + any agent: empathy first, then other speak
            if a_id in _EMPATHY_AGENTS:
                if b_id in _FILLER_AGENTS:
                    # narrative filler after empathy is redundant — empathy alone
                    return a_text
                a_clean = a_text.rstrip()
                b_clean = b_text.strip()
                sep = " " if a_clean.endswith((".", "?", "!")) else ". "
                return f"{a_clean}{sep}{b_clean}"

            # Answer agent + data_collection: answer first, then re-ask
            if a_id in _ANSWER_AGENTS and b_id == "data_collection":
                a_clean = a_text.rstrip()
                b_clean = b_text.strip()
                sep = " " if a_clean.endswith((".", "?", "!")) else ". "
                return f"{a_clean}{sep}{b_clean}"

            # Narrative filler + data_collection: skip filler, use data question
            if a_id in _FILLER_AGENTS and b_id == "data_collection":
                return b_text

            # data_collection + narrative filler: use data speak
            if a_id == "data_collection" and b_id in _FILLER_AGENTS:
                return a_text

            # Confidence tiebreaker for parallel invocations (e.g. BOTH utterances):
            # if one agent is clearly more confident, use its speak
            if abs(a_conf - b_conf) > 0.3:
                winner_text = a_text if a_conf > b_conf else b_text
                winner_id = a_id if a_conf > b_conf else b_id
                logger.debug(
                    "combine_speaks: confidence winner=%s (%.2f vs %.2f)",
                    winner_id, max(a_conf, b_conf), min(a_conf, b_conf),
                )
                return winner_text

        # General: join all non-empty speaks
        combined = " ".join(t.strip() for t in texts if t.strip())
        return combined

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

    def _fallback_plan(self, available_ids: set[str]) -> list[PlanStep]:
        """Fallback when the LLM call fails or produces no invoke steps."""
        for preferred in ("data_collection", "narrative_collection", "scheduling", "faq", "fallback"):
            if preferred in available_ids:
                return [PlanStep(action="invoke", agent_id=preferred, reason="fallback")]
        fallback_id = next(iter(available_ids))
        return [PlanStep(action="invoke", agent_id=fallback_id, reason="fallback")]
