"""Agent registry — maps agent_id to agent instance for a call."""
from __future__ import annotations

from app.agents.base import AgentBase
from app.agents.data_collection import DataCollectionAgent
from app.agents.narrative_collection import NarrativeCollectionAgent
from app.agents.intake_qualification import IntakeQualificationAgent
from app.agents.faq import FAQAgent
from app.agents.context_docs import ContextDocsAgent
from app.agents.fallback import FallbackAgent
from app.agents.farewell import FarewellAgent
from app.agents.scheduling import SchedulingAgent


def build_registry(call_id: str, transcript: list[dict]) -> dict[str, AgentBase]:
    """Build the agent registry for a single call."""
    return {
        "faq":                   FAQAgent(),
        "context_docs":          ContextDocsAgent(),
        "fallback":              FallbackAgent(),
        "farewell":              FarewellAgent(),
        "data_collection":       DataCollectionAgent(),
        "narrative_collection":  NarrativeCollectionAgent(),
        "intake_qualification":  IntakeQualificationAgent(),
        "scheduling":            SchedulingAgent(),
    }
