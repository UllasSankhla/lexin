"""Persist call transcripts to the local filesystem."""
import logging
from datetime import datetime, timezone
from pathlib import Path

import aiofiles

from app.config import settings

logger = logging.getLogger(__name__)


async def save_transcript(session) -> str | None:
    """Write the call transcript file and return its path."""
    try:
        config = session.config
        assistant_name = config.get("assistant", {}).get("persona_name", "ASSISTANT")

        lines = [
            f"CALL ID: {session.call_id}",
            f"STARTED: {datetime.fromtimestamp(session.start_time, tz=timezone.utc).isoformat()}",
            f"ASSISTANT: {assistant_name}",
            "",
            "=" * 60,
            "TRANSCRIPT",
            "=" * 60,
            "",
        ]
        lines.extend(session.transcript_lines)

        if session.collection_state and session.collection_state.collected:
            lines.append("")
            lines.append("=" * 60)
            lines.append("GATHERED PARAMETERS")
            lines.append("=" * 60)
            for name, value in session.collection_state.collected.items():
                lines.append(f"  {name}: {value}")

        lines.append("")
        lines.append(f"COMPLETED: {datetime.now(timezone.utc).isoformat()}")
        lines.append(f"DURATION: {session.duration_sec():.1f}s")

        if session.analytics_events:
            lines.append("")
            lines.append("=" * 60)
            lines.append("ANALYTICS (stage timing)")
            lines.append("=" * 60)
            for evt in session.analytics_events:
                token_info = f" ({evt['token_count']} tokens)" if evt.get("token_count") else ""
                lines.append(f"  [{evt['stage']}] {evt['event_name']}: {evt['latency_ms']:.0f}ms{token_info}")

        content = "\n".join(lines)
        path = Path(settings.transcripts_path) / f"{session.call_id}.txt"
        async with aiofiles.open(path, "w", encoding="utf-8") as f:
            await f.write(content)

        logger.info("Saved transcript for call %s to %s", session.call_id, path)
        return str(path)
    except Exception as e:
        logger.error("Failed to save transcript for call %s: %s", session.call_id, e)
        return None
