"""CalendarPrefetchTool — fetches available slots concurrently with the qualification LLM call."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone, timedelta

from app.services.calendar_service import list_available_slots
from app.tools.base import ToolBase, ToolContext, ToolResult

logger = logging.getLogger(__name__)


class CalendarPrefetchTool(ToolBase):
    """
    Pre-fetches available appointment slots for the next 7 days.

    Triggered when narrative_collection COMPLETES (fire_and_forget=False,
    await_before_agent="scheduling"), so the calendar API call runs concurrently
    with the intake_qualification LLM call.  The workflow only awaits this task
    immediately before invoking the scheduling agent, giving the Calendly API
    call the full intake_qualification processing window to complete.

    The scheduling agent reads the result from config["_tool_results"]["prefetched_slots"]
    and skips its own calendar API round-trip and event-type LLM match.

    Writes to ctx.shared
    --------------------
    prefetched_slots — list[TimeSlot]
    """

    async def run(self, ctx: ToolContext) -> ToolResult:
        collected = ctx.collected
        purpose = (
            collected.get("purpose") or collected.get("reason") or
            collected.get("service") or collected.get("appointment_type") or
            ctx.config.get("assistant", {}).get("persona_description", "appointment")
        )

        now = datetime.now(timezone.utc)
        search_start = (now + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        search_end = search_start + timedelta(days=7)

        loop = asyncio.get_running_loop()
        try:
            slots = await loop.run_in_executor(
                None,
                lambda: list_available_slots(
                    purpose, ctx.config, None,
                    search_start=search_start, search_end=search_end,
                ),
            )
        except Exception as exc:
            logger.warning(
                "CalendarPrefetchTool: fetch failed for call %s: %s",
                ctx.call_id, exc, exc_info=True,
            )
            return ToolResult(success=False, error=str(exc))

        ctx.shared["prefetched_slots"] = slots
        logger.info(
            "CalendarPrefetchTool: fetched %d slots for call %s",
            len(slots), ctx.call_id,
        )
        return ToolResult(success=True, data={"slot_count": len(slots)})
