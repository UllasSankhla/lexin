"""Webhook agent — fires after scheduling. Auto-run, no user interaction.

Note: dispatch_webhooks is an async coroutine. This agent records the payload
in internal_state so the handler can schedule the actual dispatch asynchronously
after the agent returns. The agent itself generates the summary synchronously.
"""
from __future__ import annotations

import logging
from app.agents.base import AgentBase, AgentStatus, SubagentResponse
from app.services.summary_generator import generate_call_summary

logger = logging.getLogger(__name__)


class WebhookAgent(AgentBase):
    def __init__(self, call_id: str, transcript: list[dict]) -> None:
        self._call_id = call_id
        self._transcript = transcript

    def process(
        self,
        utterance: str,
        internal_state: dict,
        config: dict,
        history: list[dict],
    ) -> SubagentResponse:
        collected = config.get("_collected", {})
        booking = config.get("_booking", {})

        # generate_call_summary takes transcript_lines (list[str]) and collected_params (dict)
        # The transcript stored in self._transcript is list[dict] with role/content keys.
        # Convert to the line format expected by the summary generator.
        transcript_lines = [
            f"{t.get('role', 'unknown').upper()}: {t.get('content', '')}"
            for t in self._transcript
        ]

        try:
            caller_name, summary = generate_call_summary(transcript_lines, collected)
        except Exception as exc:
            logger.warning("WebhookAgent summary generation failed: %s", exc)
            caller_name = collected.get("name") or collected.get("client_name") or "Unknown Caller"
            summary = "Call completed."

        # Store dispatch params in internal_state so the handler can fire dispatch_webhooks
        # asynchronously (dispatch_webhooks is an async coroutine).
        internal_state["_webhook_dispatch"] = {
            "call_id": self._call_id,
            "caller_name": caller_name,
            "ai_summary": summary,
            "collected": collected,
            "booking": booking,
        }

        logger.info(
            "WebhookAgent: prepared dispatch payload for call %s | caller=%s",
            self._call_id, caller_name,
        )

        return SubagentResponse(
            status=AgentStatus.COMPLETED,
            speak="",
            internal_state=internal_state,
        )
