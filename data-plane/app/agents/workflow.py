"""Workflow graph — nodes, per-call state, DAG logic, and WorkflowDefinition schema."""
from __future__ import annotations
from dataclasses import dataclass, field
from app.agents.base import AgentStatus, SubagentResponse
from app.tools.base import ToolBinding  # noqa: F401 — re-exported for graph_config convenience


# ── WorkflowDefinition schema ─────────────────────────────────────────────────

@dataclass
class Edge:
    target: str
    # "<node_id>" — invoke that node immediately (same turn)
    # "decider"   — speak response, wait for next utterance, consult decider
    # "resume"    — pop resume stack; invoke primary with empty utterance
    # "end"       — finalize the call
    # "self"      — route back to the same node (shorthand for on_waiting_confirm)
    reason: str = ""


@dataclass
class ActivityNode:
    id: str
    agent_class: str           # maps to a registered AgentBase implementation
    description: str           # injected into decider prompt
    is_primary_goal: bool = False
    goal_order: int | None = None
    interrupt_eligible: bool = False
    auto_run: bool = False
    depends_on: list[str] = field(default_factory=list)

    # Declared transition edges
    on_complete:       Edge = field(default_factory=lambda: Edge("decider"))
    on_failed:         Edge = field(default_factory=lambda: Edge("decider"))
    on_continue:       Edge = field(default_factory=lambda: Edge("decider"))
    on_waiting_confirm: Edge = field(default_factory=lambda: Edge("decider"))


@dataclass
class GoalSpec:
    primary_agents: list[str]  # ordered node ids — all must COMPLETE for success
    description: str           # injected into decider prompt as the mission statement


@dataclass
class DeciderSpec:
    system_prompt: str
    max_tokens: int = 1024


@dataclass
class InterruptPolicy:
    eligible_agents: list[str]
    resume_strategy: str = "stack"


@dataclass
class ErrorPolicy:
    max_consecutive_errors: int = 3
    transient_error_speak: str = (
        "I'm sorry, something went wrong on my end. Could you please repeat that?"
    )
    on_max_errors: Edge = field(default_factory=lambda: Edge("end", "agent_error"))


@dataclass
class WorkflowDefinition:
    id: str
    description: str
    goal: GoalSpec
    decider: DeciderSpec
    nodes: list[ActivityNode]
    interrupt_policy: InterruptPolicy
    error_policy: ErrorPolicy
    # Tools declared here are invoked by the workflow at lifecycle trigger points.
    # They are non-conversational and distinct from agent nodes.
    tools: list[ToolBinding] = field(default_factory=list)

    def __post_init__(self) -> None:
        node_ids = {n.id for n in self.nodes}
        valid_targets = {"decider", "resume", "end", "self"} | node_ids
        for n in self.nodes:
            for dep in n.depends_on:
                if dep not in node_ids:
                    raise ValueError(f"Node '{n.id}' depends on unknown node '{dep}'")
            for attr in ("on_complete", "on_failed", "on_continue", "on_waiting_confirm"):
                t = getattr(n, attr).target
                if t not in valid_targets:
                    raise ValueError(
                        f"Node '{n.id}'.{attr} references unknown target '{t}'"
                    )
        for pa in self.goal.primary_agents:
            if pa not in node_ids:
                raise ValueError(f"GoalSpec.primary_agents references unknown node '{pa}'")
        for ea in self.interrupt_policy.eligible_agents:
            if ea not in node_ids:
                raise ValueError(f"InterruptPolicy references unknown node '{ea}'")
        orders = [n.goal_order for n in self.nodes if n.goal_order is not None]
        if len(orders) != len(set(orders)):
            raise ValueError("goal_order values must be unique among nodes")

    def node(self, node_id: str) -> ActivityNode:
        for n in self.nodes:
            if n.id == node_id:
                return n
        raise KeyError(node_id)


# ── Per-call state ────────────────────────────────────────────────────────────

@dataclass
class AgentState:
    node_id: str
    status: AgentStatus = AgentStatus.NOT_STARTED
    internal_state: dict = field(default_factory=dict)
    last_response: SubagentResponse | None = None
    last_spoke: str = ""
    turn_activated: int = 0
    invocation_count: int = 0


# ── WorkflowGraph ─────────────────────────────────────────────────────────────

class WorkflowGraph:
    def __init__(self, workflow: WorkflowDefinition) -> None:
        self.workflow = workflow
        self.nodes: dict[str, ActivityNode] = {n.id: n for n in workflow.nodes}
        self.states: dict[str, AgentState] = {
            n.id: AgentState(node_id=n.id) for n in workflow.nodes
        }

    # ── Edge resolution ───────────────────────────────────────────────────────

    def get_edge(self, node_id: str, status: AgentStatus) -> Edge:
        """Return the declared edge for the given node and agent status."""
        node = self.nodes[node_id]
        if status == AgentStatus.COMPLETED:
            return node.on_complete
        if status == AgentStatus.FAILED:
            return node.on_failed
        if status == AgentStatus.WAITING_CONFIRM:
            edge = node.on_waiting_confirm
            # "self" is shorthand — resolve to the node's own id
            return Edge(node_id, edge.reason) if edge.target == "self" else edge
        # IN_PROGRESS / NOT_STARTED
        return node.on_continue

    # ── Graph queries ─────────────────────────────────────────────────────────

    def deps_met(self, node_id: str) -> bool:
        return all(
            self.states[dep].status == AgentStatus.COMPLETED
            for dep in self.nodes[node_id].depends_on
        )

    def available_nodes(self) -> list[ActivityNode]:
        return [
            node for node in self.nodes.values()
            if self.deps_met(node.id)
            and not node.auto_run
            and (
                node.interrupt_eligible  # interrupt agents are always re-invokable
                or self.states[node.id].status
                not in (AgentStatus.COMPLETED, AgentStatus.FAILED)
            )
        ]

    def active_primary(self) -> AgentState | None:
        """Return the primary (non-interrupt, non-auto-run) agent currently IN_PROGRESS or WAITING_CONFIRM."""
        for node in self.nodes.values():
            if node.interrupt_eligible or node.auto_run:
                continue
            state = self.states[node.id]
            if state.status in (AgentStatus.IN_PROGRESS, AgentStatus.WAITING_CONFIRM):
                return state
        return None

    def active_waiting_confirm(self) -> str | None:
        """Return node_id if any non-auto-run node is WAITING_CONFIRM, else None."""
        for nid, state in self.states.items():
            if self.nodes[nid].auto_run:
                continue
            if state.status == AgentStatus.WAITING_CONFIRM:
                return nid
        return None

    def next_primary_goal(self) -> str | None:
        """Return the next runnable, incomplete primary goal agent in goal_order."""
        primary = sorted(
            [n for n in self.nodes.values() if n.is_primary_goal],
            key=lambda n: n.goal_order or 0,
        )
        for node in primary:
            state = self.states[node.id]
            if state.status not in (AgentStatus.COMPLETED, AgentStatus.FAILED) \
                    and self.deps_met(node.id):
                return node.id
        return None

    def is_goal_complete(self) -> bool:
        """All primary goal nodes COMPLETED."""
        return all(
            self.states[nid].status == AgentStatus.COMPLETED
            for nid in self.workflow.goal.primary_agents
        )

    def is_goal_terminal(self) -> bool:
        """All primary goal nodes COMPLETED or FAILED — no more work possible."""
        return all(
            self.states[nid].status in (AgentStatus.COMPLETED, AgentStatus.FAILED)
            for nid in self.workflow.goal.primary_agents
        )

    def auto_run_ready(self) -> list[ActivityNode]:
        return [
            node for node in self.nodes.values()
            if node.auto_run
            and self.deps_met(node.id)
            and self.states[node.id].status == AgentStatus.NOT_STARTED
        ]

    def update(self, agent_id: str, response: SubagentResponse, turn: int) -> None:
        state = self.states[agent_id]
        state.status = response.status
        state.internal_state = response.internal_state
        state.last_response = response
        if response.speak:
            state.last_spoke = response.speak
        state.turn_activated = turn
        state.invocation_count += 1

    # ── Prompt helpers ────────────────────────────────────────────────────────

    def status_summary(self) -> str:
        lines = []
        for nid, state in self.states.items():
            node = self.nodes[nid]
            spoke = f' | last said: "{state.last_spoke[:60]}"' if state.last_spoke else ""
            extra = ""
            if state.internal_state.get("current_field"):
                extra = f' | asking: {state.internal_state["current_field"]}'
            if state.status == AgentStatus.WAITING_CONFIRM:
                pv = state.internal_state.get("pending_value", "")
                extra = f' | WAITING YES/NO for: "{pv}"'
            invocations = (
                f" | invoked {state.invocation_count}x"
                if node.interrupt_eligible and state.invocation_count > 0
                else ""
            )
            lines.append(f"  {nid}: {state.status.value}{extra}{invocations}{spoke}")
        return "\n".join(lines)

    def available_summary(self) -> str:
        lines = [
            f"  - {node.id}: {node.description}"
            for node in self.available_nodes()
        ]
        return "\n".join(lines) or "  (none)"
