"""WebhookTool — dispatches call.completed events to configured endpoints."""
from __future__ import annotations

import logging

from app.services.webhook_dispatcher import dispatch_webhooks
from app.tools.base import ToolBase, ToolContext, ToolResult

logger = logging.getLogger(__name__)


class WebhookTool(ToolBase):
    """
    Dispatches call.completed webhooks to all endpoints configured for this tenant.

    Reads from ctx.shared
    ---------------------
    caller_name  — written by SummarizationTool (falls back to "Unknown Caller")
    ai_summary   — written by SummarizationTool (falls back to "")

    When declared after SummarizationTool in the CALL_END group, tools run
    sequentially so these values are always available.
    """

    async def run(self, ctx: ToolContext) -> ToolResult:
        caller_name = ctx.shared.get("caller_name", "Unknown Caller")
        ai_summary  = ctx.shared.get("ai_summary", "")

        try:
            await dispatch_webhooks(
                config=ctx.config,
                call_id=ctx.call_id,
                duration_sec=ctx.duration_sec,
                collected=ctx.collected,
                transcript_path=ctx.transcript_path,
                caller_name=caller_name,
                ai_summary=ai_summary,
                booking_details=ctx.booking_result or None,
            )
            logger.info("WebhookTool: dispatched for call %s", ctx.call_id)
            return ToolResult(success=True, data={})
        except Exception as exc:
            logger.warning("WebhookTool: dispatch failed for call %s: %s", ctx.call_id, exc)
            return ToolResult(success=False, error=str(exc))
