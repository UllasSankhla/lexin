"""Base types shared by all agents."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum


class AgentStatus(str, Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    WAITING_CONFIRM = "waiting_confirm"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SubagentResponse:
    status: AgentStatus
    speak: str = ""
    # DataCollection: confirmed field values {name: value}
    collected: dict[str, str] = field(default_factory=dict)
    # DataCollection: field + extracted value awaiting yes/no {"field": "name", "value": "John"}
    pending_confirmation: dict | None = None
    # Fallback: running notes text appended each invocation
    notes: str | None = None
    # Scheduling: confirmed booking dict
    booking: dict | None = None
    # Agent persists its own state here; router stores and passes back next turn
    internal_state: dict = field(default_factory=dict)


class AgentBase(ABC):
    @abstractmethod
    def process(
        self,
        utterance: str,
        internal_state: dict,
        config: dict,
        history: list[dict],
    ) -> SubagentResponse:
        """Process one caller utterance. All args are read-only except internal_state."""
        ...
