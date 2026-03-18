"""Workflow graph — nodes, per-call state, and DAG logic."""
from __future__ import annotations
from dataclasses import dataclass, field
from app.agents.base import AgentStatus, SubagentResponse


@dataclass
class WorkflowNode:
    id: str
    description: str
    depends_on: list[str] = field(default_factory=list)
    interrupt_eligible: bool = False
    auto_run: bool = False
    completion_criteria: str = ""


@dataclass
class AgentState:
    node_id: str
    status: AgentStatus = AgentStatus.NOT_STARTED
    internal_state: dict = field(default_factory=dict)
    last_response: SubagentResponse | None = None
    last_spoke: str = ""
    turn_activated: int = 0


class WorkflowGraph:
    def __init__(self, nodes: list[WorkflowNode]) -> None:
        self.nodes: dict[str, WorkflowNode] = {n.id: n for n in nodes}
        self.states: dict[str, AgentState] = {
            n.id: AgentState(node_id=n.id) for n in nodes
        }
        self._validate()

    def _validate(self) -> None:
        for node in self.nodes.values():
            for dep in node.depends_on:
                if dep not in self.nodes:
                    raise ValueError(
                        f"Node '{node.id}' depends on unknown node '{dep}'"
                    )

    def deps_met(self, node_id: str) -> bool:
        return all(
            self.states[dep].status == AgentStatus.COMPLETED
            for dep in self.nodes[node_id].depends_on
        )

    def available_nodes(self) -> list[WorkflowNode]:
        return [
            node for node in self.nodes.values()
            if self.deps_met(node.id)
            and self.states[node.id].status
            not in (AgentStatus.COMPLETED, AgentStatus.FAILED)
            and not node.auto_run
        ]

    def active_primary(self) -> AgentState | None:
        """Return the primary (non-interrupt) agent currently IN_PROGRESS or WAITING_CONFIRM."""
        for node in self.nodes.values():
            if node.interrupt_eligible or node.auto_run:
                continue
            state = self.states[node.id]
            if state.status in (AgentStatus.IN_PROGRESS, AgentStatus.WAITING_CONFIRM):
                return state
        return None

    def update(self, agent_id: str, response: SubagentResponse, turn: int) -> None:
        state = self.states[agent_id]
        state.status = response.status
        state.internal_state = response.internal_state
        state.last_response = response
        if response.speak:
            state.last_spoke = response.speak
        state.turn_activated = turn

    def is_goal_complete(self) -> bool:
        """All primary nodes completed successfully."""
        return all(
            self.states[nid].status == AgentStatus.COMPLETED
            for nid, node in self.nodes.items()
            if not node.auto_run and not node.interrupt_eligible
        )

    def is_goal_terminal(self) -> bool:
        """All primary nodes are COMPLETED or FAILED — no more work possible."""
        return all(
            self.states[nid].status in (AgentStatus.COMPLETED, AgentStatus.FAILED)
            for nid, node in self.nodes.items()
            if not node.auto_run and not node.interrupt_eligible
        )

    def next_primary_goal(self) -> str | None:
        """Return the agent_id of the next runnable, incomplete primary agent."""
        for nid, node in self.nodes.items():
            if node.auto_run or node.interrupt_eligible:
                continue
            state = self.states[nid]
            if state.status not in (AgentStatus.COMPLETED, AgentStatus.FAILED) \
                    and self.deps_met(nid):
                return nid
        return None

    def auto_run_ready(self) -> list[WorkflowNode]:
        return [
            node for node in self.nodes.values()
            if node.auto_run
            and self.deps_met(node.id)
            and self.states[node.id].status == AgentStatus.NOT_STARTED
        ]

    def status_summary(self) -> str:
        lines = []
        for nid, state in self.states.items():
            spoke = (
                f' | last said: "{state.last_spoke[:60]}"'
                if state.last_spoke else ""
            )
            extra = ""
            if state.internal_state.get("current_field"):
                extra = f' | asking: {state.internal_state["current_field"]}'
            if state.status == AgentStatus.WAITING_CONFIRM:
                pv = state.internal_state.get("pending_value", "")
                extra = f' | WAITING YES/NO for: "{pv}"'
            lines.append(f"  {nid}: {state.status.value}{extra}{spoke}")
        return "\n".join(lines)

    def available_summary(self) -> str:
        lines = []
        for node in self.available_nodes():
            lines.append(f"  - {node.id}: {node.description}")
        return "\n".join(lines) or "  (none)"
