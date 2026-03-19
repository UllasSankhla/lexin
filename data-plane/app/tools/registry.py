"""Tool registry — resolves tool class names declared in WorkflowDefinition."""
from __future__ import annotations

from app.tools.base import ToolBase
from app.tools.calendar_prefetch import CalendarPrefetchTool
from app.tools.summarization import SummarizationTool
from app.tools.webhook import WebhookTool

_REGISTRY: dict[str, type[ToolBase]] = {
    "SummarizationTool":    SummarizationTool,
    "WebhookTool":          WebhookTool,
    "CalendarPrefetchTool": CalendarPrefetchTool,
}


def resolve_tool(class_name: str) -> ToolBase:
    """Return a fresh instance of the named tool, or raise ValueError."""
    cls = _REGISTRY.get(class_name)
    if cls is None:
        raise ValueError(
            f"Unknown tool class: {class_name!r}. Registered: {sorted(_REGISTRY)}"
        )
    return cls()
