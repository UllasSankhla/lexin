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
    UNHANDLED = "unhandled"  # agent cannot process utterance — re-routes to decider


@dataclass
class SubagentResponse:
    status: AgentStatus
    speak: str = ""
    # DataCollection: confirmed field values {name: value} — sent to UI as parameter_collected events
    collected: dict[str, str] = field(default_factory=dict)
    # Backend-only fields: accumulated in collected_all (for webhook/downstream) but NOT sent to UI
    hidden_collected: dict[str, str] = field(default_factory=dict)
    # DataCollection: field + extracted value awaiting yes/no {"field": "name", "value": "John"}
    pending_confirmation: dict | None = None
    # Fallback: running notes text appended each invocation
    notes: str | None = None
    # Scheduling: confirmed booking dict
    booking: dict | None = None
    # Agent persists its own state here; router stores and passes back next turn
    internal_state: dict = field(default_factory=dict)
    # Confidence score (0.0–1.0): how well this agent handled the utterance.
    # Used by handler to select the best response when multiple agents are invoked.
    # WAITING_CONFIRM responses always win regardless of confidence.
    confidence: float = 1.0


class AgentBase(ABC):
    # Set to True on primary interactive agents (DataCollection, NarrativeCollection,
    # Scheduling).  One-shot agents (Empathy, IntakeQualification, FAQ, Fallback,
    # Farewell) leave this False — the framework contract does not apply to them.
    is_primary_interactive: bool = False

    @abstractmethod
    def process(
        self,
        utterance: str,
        internal_state: dict,
        config: dict,
        history: list[dict],
    ) -> SubagentResponse:
        """Process one caller utterance.

        Primary interactive agent contract (is_primary_interactive=True):
        1. Domain gate — if utterance is outside this agent's domain, return
           SubagentResponse(status=UNHANDLED) with internal_state UNCHANGED.
        2. State immutability on UNHANDLED — never mutate internal_state before
           the domain gate decision.
        3. Resume speak — when utterance is "" (resume after interrupt), re-surface
           whatever the agent was waiting for (pending confirmation, next question,
           etc.) so the caller knows where to pick up.
        """
        ...
