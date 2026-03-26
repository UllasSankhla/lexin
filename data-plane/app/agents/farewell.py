"""Farewell agent — detects caller goodbye and ends the call politely.

No LLM required: farewell intent is detected by the router before this agent
is invoked.  The agent simply responds with a warm closing line and returns
COMPLETED, which the graph resolves to Edge("end", "caller_farewell").
"""
from __future__ import annotations

import logging
import random

from app.agents.base import AgentBase, AgentStatus, SubagentResponse

logger = logging.getLogger(__name__)

_RESPONSES = [
    "Thank you for calling. Have a wonderful day, goodbye!",
    "It was a pleasure speaking with you. Have a great day, goodbye!",
    "Thank you for reaching out. We look forward to speaking with you soon. Goodbye!",
    "Thanks for calling. Take care, and have a great day!",
]


class FarewellAgent(AgentBase):
    """Responds with a polite closing line and signals the call to end."""

    def process(
        self,
        utterance: str,
        internal_state: dict,
        config: dict,
        history: list[dict],
    ) -> SubagentResponse:
        speak = random.choice(_RESPONSES)
        logger.info("FarewellAgent: caller farewell detected — ending call | utterance=%r", utterance[:80])
        return SubagentResponse(
            status=AgentStatus.COMPLETED,
            speak=speak,
            internal_state=internal_state,
        )
