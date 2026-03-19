"""Tool abstraction — non-conversational async work units invoked by the workflow.

Tools are distinct from Agents:
  - They do not speak to the caller and have no internal conversational state.
  - They are triggered by the workflow at explicit lifecycle points (CALL_END,
    AGENT_COMPLETE) rather than by caller utterances.
  - They communicate via a shared mutable context dict rather than SubagentResponse.
  - They are always async-native (no run_in_executor boilerplate in the caller).

Lifecycle triggers
------------------
CALL_END        — fired when _finalize_call runs; tools execute sequentially in
                  declaration order inside a single background task so each can
                  read results written to ctx.shared by the previous one.

AGENT_COMPLETE  — fired immediately after a named agent reaches COMPLETED status,
                  before the workflow follows the edge.  fire_and_forget=False tools
                  are tracked as pending asyncio.Tasks; the workflow awaits them
                  before invoking the next agent so results are available via
                  config["_tool_results"].
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum


class ToolTrigger(Enum):
    CALL_END       = "call_end"
    AGENT_COMPLETE = "agent_complete"   # requires agent_id on the binding


@dataclass
class ToolBinding:
    """Declares one tool invocation inside a WorkflowDefinition."""
    tool_class:         str             # key in the tool registry
    trigger:            ToolTrigger
    agent_id:           str | None = None    # required when trigger == AGENT_COMPLETE
    fire_and_forget:    bool       = True    # False → workflow awaits result before proceeding
    await_before_agent: str | None = None    # if set, only await this task immediately before
                                             # this specific agent runs (not before every agent)


@dataclass
class ToolContext:
    """Immutable call snapshot plus a mutable shared store passed to every tool."""
    call_id:          str
    config:           dict
    collected:        dict
    transcript_lines: list[str]
    booking_result:   dict
    duration_sec:     float
    transcript_path:  str | None
    # Mutable: tools write results here; subsequent tools and agents read from it.
    shared:           dict = field(default_factory=dict)


@dataclass
class ToolResult:
    success: bool
    data:    dict          = field(default_factory=dict)
    error:   str | None   = None


class ToolBase(ABC):
    """Base class for all tools.  Subclasses implement a single async run()."""

    @abstractmethod
    async def run(self, ctx: ToolContext) -> ToolResult: ...
