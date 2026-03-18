"""LLM-based turn router. Decides which agent handles each caller utterance."""
from __future__ import annotations

import logging
from app.agents.base import AgentStatus
from app.agents.llm_utils import llm_json_call
from app.agents.workflow import WorkflowGraph

logger = logging.getLogger(__name__)

_ROUTER_SYSTEM = """\
You are a routing agent for a voice appointment booking system.
Your job: decide which agent should handle the caller's current utterance.

RULES:
1. If an agent has status WAITING_CONFIRM and the utterance is a yes/no/correction, \
route to that same agent — it is waiting for a confirmation answer.
2. If the active primary agent (data_collection or scheduling) is in progress and \
the caller asks a question (not an answer), route to faq, context_docs, or fallback.
3. If data_collection is COMPLETED and scheduling is available, route to scheduling.
4. Default to data_collection if it is still in progress.
5. Only route to agents listed under AVAILABLE AGENTS.
6. Never route to an agent not in AVAILABLE AGENTS.

Respond ONLY with valid JSON: {"agent_id": "<id>", "interrupt": <true|false>, "reasoning": "<one line>"}
"""


class Router:
    def __init__(self, graph: WorkflowGraph, goal: str) -> None:
        self._graph = graph
        self._goal = goal
        # Resume stack: list of agent_ids interrupted mid-task
        self._resume_stack: list[str] = []

    def select(self, utterance: str, recent_history: list[dict]) -> str:
        """
        Given the caller's utterance and recent conversation turns,
        return the agent_id that should handle this turn.
        """
        graph = self._graph
        available = graph.available_nodes()
        if not available:
            logger.warning("Router: no available agents — defaulting to data_collection")
            return "data_collection"

        # Build compact status block
        status_block = graph.status_summary()
        available_block = graph.available_summary()

        # Format recent history (last 8 turns)
        history_lines = []
        for turn in recent_history[-8:]:
            role = turn.get("role", "?")
            content = turn.get("content", "")[:120]
            history_lines.append(f"  {role}: {content}")
        history_text = "\n".join(history_lines) or "  (none)"

        user_msg = (
            f"GOAL: {self._goal}\n\n"
            f"RECENT CONVERSATION:\n{history_text}\n\n"
            f"AGENT STATUS:\n{status_block}\n\n"
            f"AVAILABLE AGENTS:\n{available_block}\n\n"
            f"CALLER JUST SAID: \"{utterance}\"\n\n"
            f"Which agent should handle this? Reply JSON only."
        )

        try:
            result = llm_json_call(_ROUTER_SYSTEM, user_msg, max_tokens=150)
            agent_id = result.get("agent_id", "")
            interrupt = result.get("interrupt", False)
            reasoning = result.get("reasoning", "")

            # Log full routing context at INFO level
            logger.info(
                "Router decision | agent=%s | interrupt=%s | reasoning=%r\n"
                "  utterance: %r\n"
                "  status block:\n%s\n"
                "  available:\n%s",
                agent_id, interrupt, reasoning,
                utterance[:120],
                "\n".join(f"    {l}" for l in status_block.splitlines()),
                "\n".join(f"    {l}" for l in available_block.splitlines()),
            )

            # Validate selection is actually available
            available_ids = {n.id for n in available}
            if agent_id not in available_ids:
                logger.warning(
                    "Router returned unavailable agent %r — falling back", agent_id
                )
                agent_id = self._fallback_agent(available_ids)

            # Manage resume stack
            if interrupt:
                active = graph.active_primary()
                if active and active.node_id not in self._resume_stack:
                    self._resume_stack.append(active.node_id)
                    logger.info("Router: pushed %s onto resume stack", active.node_id)

            return agent_id

        except Exception as exc:
            logger.error("Router LLM call failed: %s — using fallback", exc)
            available_ids = {n.id for n in available}
            return self._fallback_agent(available_ids)

    def pop_resume(self) -> str | None:
        """Pop and return the most recently interrupted primary agent, if any."""
        if self._resume_stack:
            agent_id = self._resume_stack.pop()
            logger.info("Router: resuming %s from stack", agent_id)
            return agent_id
        return None

    def _fallback_agent(self, available_ids: set[str]) -> str:
        """Pick the most sensible available agent without LLM."""
        for preferred in ("data_collection", "scheduling", "faq", "fallback"):
            if preferred in available_ids:
                return preferred
        return next(iter(available_ids))
