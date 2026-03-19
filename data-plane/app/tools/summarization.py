"""SummarizationTool — generates the post-call AI summary and persists it."""
from __future__ import annotations

import asyncio
import logging

from app.database import SessionLocal
from app.models.call_record import CallRecord
from app.services.summary_generator import generate_call_summary
from app.tools.base import ToolBase, ToolContext, ToolResult

logger = logging.getLogger(__name__)


class SummarizationTool(ToolBase):
    """
    Runs generate_call_summary in a thread, writes caller_name and ai_summary
    to the call record, and publishes both to ctx.shared so that downstream
    tools (e.g. WebhookTool) can read them without a DB round-trip.

    Writes to ctx.shared
    --------------------
    caller_name  — str
    ai_summary   — str
    """

    async def run(self, ctx: ToolContext) -> ToolResult:
        loop = asyncio.get_event_loop()
        caller_name, ai_summary = "Unknown Caller", ""

        try:
            caller_name, ai_summary = await loop.run_in_executor(
                None,
                lambda: generate_call_summary(
                    ctx.transcript_lines, ctx.collected, ctx.config
                ),
            )
        except Exception as exc:
            logger.warning("SummarizationTool: summary generation failed: %s", exc)
            # Publish defaults so WebhookTool still has something to work with
            ctx.shared["caller_name"] = caller_name
            ctx.shared["ai_summary"]  = ai_summary
            return ToolResult(success=False, error=str(exc))

        ctx.shared["caller_name"] = caller_name
        ctx.shared["ai_summary"]  = ai_summary

        bg_db = SessionLocal()
        try:
            record = bg_db.get(CallRecord, ctx.call_id)
            if record:
                record.caller_name = caller_name
                record.ai_summary  = ai_summary
                bg_db.commit()
                logger.info(
                    "SummarizationTool: persisted for call %s — caller=%r",
                    ctx.call_id, caller_name,
                )
        except Exception as exc:
            logger.warning("SummarizationTool: DB update failed: %s", exc)
            return ToolResult(
                success=False,
                data={"caller_name": caller_name, "ai_summary": ai_summary},
                error=str(exc),
            )
        finally:
            bg_db.close()

        return ToolResult(
            success=True,
            data={"caller_name": caller_name, "ai_summary": ai_summary},
        )
