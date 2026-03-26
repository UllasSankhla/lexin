"""LLM-based turn router — enforces WAITING_CONFIRM, then consults decider LLM."""
from __future__ import annotations

import logging
from app.agents.llm_utils import llm_json_call
from app.agents.workflow import WorkflowGraph

logger = logging.getLogger(__name__)


class Router:
    def __init__(self, graph: WorkflowGraph) -> None:
        self._graph = graph
        self._resume_stack: list[str] = []

    def select(
        self,
        utterance: str,
        recent_history: list[dict],
        hint: str | None = None,
    ) -> tuple[str, bool]:
        """
        Return (agent_id, interrupt).
        Enforces WAITING_CONFIRM unconditionally before consulting the LLM.
        """
        graph = self._graph
        workflow = graph.workflow

        # Hard enforcement: if any node is WAITING_CONFIRM, route back to it
        waiting_id = graph.active_waiting_confirm()
        if waiting_id:
            logger.info(
                "Router: WAITING_CONFIRM enforced → %s (LLM bypassed)", waiting_id
            )
            return waiting_id, False

        available = graph.available_nodes()
        if not available:
            logger.warning("Router: no available agents — defaulting to data_collection")
            return "data_collection", False

        status_block = graph.status_summary()
        available_block = graph.available_summary()
        booking_stages_block = graph.primary_goal_summary()
        history_lines = "\n".join(
            f"  {t['role']}: {t['content'][:120]}" for t in recent_history[-8:]
        ) or "  (none)"

        next_goal_id = graph.next_primary_goal()
        next_goal_line = ""
        if next_goal_id:
            next_node = graph.nodes[next_goal_id]
            next_state = graph.states[next_goal_id]
            next_goal_line = (
                f"\nNEXT PRIMARY GOAL TO PURSUE: {next_goal_id} "
                f"[{next_state.status.value}]\n"
                f"  {next_node.description}\n"
            )

        hint_block = (
            f"\nCONTEXT HINT: {hint}\n"
            "(The previous agent could not process this utterance and has deferred "
            "to the router. Route to an interrupt-eligible agent if the caller is "
            "asking a question or wants assistance; route back to the primary goal "
            "agent only if the utterance is clearly an attempt to provide information.)\n"
            if hint else ""
        )

        user_msg = (
            f"MISSION:\n{workflow.goal.description}\n\n"
            f"{next_goal_line}"
            f"BOOKING STAGES:\n{booking_stages_block}\n\n"
            f"RECENT CONVERSATION:\n{history_lines}\n\n"
            f"AGENT STATUS:\n{status_block}\n\n"
            f"AVAILABLE AGENTS:\n{available_block}\n\n"
            f"CALLER JUST SAID: \"{utterance}\"\n"
            f"{hint_block}\n"
            "Which agent should handle this? Reply JSON only."
        )

        try:
            result = llm_json_call(
                workflow.decider.system_prompt,
                user_msg,
                workflow.decider.max_tokens,
            )
            agent_id = result.get("agent_id", "")
            interrupt = bool(result.get("interrupt", False))
            reasoning = result.get("reasoning", "")

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

            available_ids = {n.id for n in available}
            if agent_id not in available_ids:
                logger.warning(
                    "Router returned unavailable agent %r — falling back", agent_id
                )
                agent_id = self._fallback_agent(available_ids)
                interrupt = False

            if interrupt:
                active = graph.active_primary()
                if active and active.node_id not in self._resume_stack:
                    self._resume_stack.append(active.node_id)
                    logger.info("Router: pushed %s onto resume stack", active.node_id)

            return agent_id, interrupt

        except Exception as exc:
            logger.error("Router LLM call failed: %s — using fallback", exc)
            return self._fallback_agent({n.id for n in available}), False

    def pop_resume(self) -> str | None:
        if self._resume_stack:
            agent_id = self._resume_stack.pop()
            logger.info("Router: resuming %s from stack", agent_id)
            return agent_id
        return None

    def _fallback_agent(self, available_ids: set[str]) -> str:
        for preferred in ("data_collection", "scheduling", "faq", "fallback"):
            if preferred in available_ids:
                return preferred
        return next(iter(available_ids))
