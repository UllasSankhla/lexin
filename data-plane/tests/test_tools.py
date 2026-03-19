"""Tests for the tool system introduced in app/tools/.

Run with:
    PYTHONPATH=. .venv/bin/python3 -m pytest tests/test_tools.py -v
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ctx(**overrides):
    """Return a minimal ToolContext suitable for unit tests."""
    from app.tools.base import ToolContext

    defaults = dict(
        call_id="test-call-id",
        config={},
        collected={},
        transcript_lines=[],
        booking_result={},
        duration_sec=60.0,
        transcript_path=None,
        shared={},
    )
    defaults.update(overrides)
    return ToolContext(**defaults)


# ===========================================================================
# 1. ToolBinding / ToolContext / ToolResult dataclasses
# ===========================================================================

class TestDataclasses:
    def test_tool_binding_fields_and_defaults(self):
        from app.tools.base import ToolBinding, ToolTrigger

        binding = ToolBinding(
            tool_class="SomeTool",
            trigger=ToolTrigger.CALL_END,
        )
        assert binding.tool_class == "SomeTool"
        assert binding.trigger == ToolTrigger.CALL_END
        assert binding.agent_id is None
        assert binding.fire_and_forget is True
        assert binding.await_before_agent is None

    def test_tool_binding_explicit_values(self):
        from app.tools.base import ToolBinding, ToolTrigger

        binding = ToolBinding(
            tool_class="CalendarPrefetchTool",
            trigger=ToolTrigger.AGENT_COMPLETE,
            agent_id="narrative_collection",
            fire_and_forget=False,
            await_before_agent="scheduling",
        )
        assert binding.agent_id == "narrative_collection"
        assert binding.fire_and_forget is False
        assert binding.await_before_agent == "scheduling"

    def test_tool_context_shared_defaults_to_empty_dict(self):
        ctx = _make_ctx()
        assert isinstance(ctx.shared, dict)
        assert ctx.shared == {}

    def test_tool_result_defaults(self):
        from app.tools.base import ToolResult

        result = ToolResult(success=True)
        assert result.data == {}
        assert result.error is None

    def test_tool_result_failure_with_error(self):
        from app.tools.base import ToolResult

        result = ToolResult(success=False, error="something broke")
        assert result.success is False
        assert result.error == "something broke"


# ===========================================================================
# 2. Tool registry
# ===========================================================================

class TestToolRegistry:
    def test_resolve_summarization_tool(self):
        from app.tools.registry import resolve_tool
        from app.tools.summarization import SummarizationTool

        instance = resolve_tool("SummarizationTool")
        assert isinstance(instance, SummarizationTool)

    def test_resolve_webhook_tool(self):
        from app.tools.registry import resolve_tool
        from app.tools.webhook import WebhookTool

        instance = resolve_tool("WebhookTool")
        assert isinstance(instance, WebhookTool)

    def test_resolve_calendar_prefetch_tool(self):
        from app.tools.registry import resolve_tool
        from app.tools.calendar_prefetch import CalendarPrefetchTool

        instance = resolve_tool("CalendarPrefetchTool")
        assert isinstance(instance, CalendarPrefetchTool)

    def test_resolve_unknown_raises_value_error(self):
        from app.tools.registry import resolve_tool

        with pytest.raises(ValueError, match="Unknown tool class"):
            resolve_tool("UnknownTool")

    def test_resolve_returns_fresh_instance_each_call(self):
        from app.tools.registry import resolve_tool

        a = resolve_tool("SummarizationTool")
        b = resolve_tool("SummarizationTool")
        assert a is not b


# ===========================================================================
# 3. graph_config tool bindings
# ===========================================================================

class TestGraphConfigToolBindings:
    def test_three_tools_declared(self):
        from app.agents.graph_config import APPOINTMENT_BOOKING

        assert len(APPOINTMENT_BOOKING.tools) == 3

    def test_calendar_prefetch_binding(self):
        from app.agents.graph_config import APPOINTMENT_BOOKING
        from app.tools.base import ToolTrigger

        bindings = {b.tool_class: b for b in APPOINTMENT_BOOKING.tools}
        b = bindings["CalendarPrefetchTool"]
        assert b.trigger == ToolTrigger.AGENT_COMPLETE
        assert b.agent_id == "narrative_collection"
        assert b.fire_and_forget is False
        assert b.await_before_agent == "scheduling"

    def test_summarization_tool_binding(self):
        from app.agents.graph_config import APPOINTMENT_BOOKING
        from app.tools.base import ToolTrigger

        bindings = {b.tool_class: b for b in APPOINTMENT_BOOKING.tools}
        b = bindings["SummarizationTool"]
        assert b.trigger == ToolTrigger.CALL_END
        assert b.fire_and_forget is True

    def test_webhook_tool_binding(self):
        from app.agents.graph_config import APPOINTMENT_BOOKING
        from app.tools.base import ToolTrigger

        bindings = {b.tool_class: b for b in APPOINTMENT_BOOKING.tools}
        b = bindings["WebhookTool"]
        assert b.trigger == ToolTrigger.CALL_END
        assert b.fire_and_forget is True

    def test_summarization_before_webhook_in_list(self):
        from app.agents.graph_config import APPOINTMENT_BOOKING

        class_names = [b.tool_class for b in APPOINTMENT_BOOKING.tools]
        assert class_names.index("SummarizationTool") < class_names.index("WebhookTool")


# ===========================================================================
# 4. SummarizationTool
# ===========================================================================

class TestSummarizationTool:
    def _make_db_mock(self, record=None):
        """Return a SessionLocal() mock whose context supports get/commit/close."""
        db_instance = MagicMock()
        db_instance.get.return_value = record or MagicMock()
        db_instance.commit.return_value = None
        db_instance.close.return_value = None

        session_local = MagicMock(return_value=db_instance)
        return session_local

    def test_success_path(self):
        from app.tools.summarization import SummarizationTool

        ctx = _make_ctx()
        session_mock = self._make_db_mock()

        with patch("app.tools.summarization.generate_call_summary",
                   return_value=("Jane Smith", "She called about a contract.")), \
             patch("app.tools.summarization.SessionLocal", session_mock):
            result = asyncio.run(SummarizationTool().run(ctx))

        assert result.success is True
        assert result.data["caller_name"] == "Jane Smith"
        assert result.data["ai_summary"] == "She called about a contract."
        assert ctx.shared["caller_name"] == "Jane Smith"
        assert ctx.shared["ai_summary"] == "She called about a contract."

    def test_failure_path_defaults_written_to_shared(self):
        from app.tools.summarization import SummarizationTool

        ctx = _make_ctx()

        with patch("app.tools.summarization.generate_call_summary",
                   side_effect=Exception("fail")):
            result = asyncio.run(SummarizationTool().run(ctx))

        assert result.success is False
        assert ctx.shared["caller_name"] == "Unknown Caller"


# ===========================================================================
# 5. WebhookTool
# ===========================================================================

class TestWebhookTool:
    def test_success_reads_from_shared_and_calls_dispatch(self):
        from app.tools.webhook import WebhookTool

        shared = {"caller_name": "Bob", "ai_summary": "Summary."}
        ctx = _make_ctx(shared=shared)

        dispatch_mock = AsyncMock(return_value=None)
        with patch("app.tools.webhook.dispatch_webhooks", dispatch_mock):
            result = asyncio.run(WebhookTool().run(ctx))

        assert result.success is True
        dispatch_mock.assert_called_once()
        _, kwargs = dispatch_mock.call_args
        assert kwargs["caller_name"] == "Bob"
        assert kwargs["ai_summary"] == "Summary."

    def test_failure_when_dispatch_raises(self):
        from app.tools.webhook import WebhookTool

        ctx = _make_ctx(shared={"caller_name": "Bob", "ai_summary": "Summary."})

        dispatch_mock = AsyncMock(side_effect=Exception("network error"))
        with patch("app.tools.webhook.dispatch_webhooks", dispatch_mock):
            result = asyncio.run(WebhookTool().run(ctx))

        assert result.success is False
        assert result.error is not None
        assert "network error" in result.error


# ===========================================================================
# 6. CalendarPrefetchTool
# ===========================================================================

class TestCalendarPrefetchTool:
    def _fake_slots(self, count=3):
        slots = []
        for i in range(count):
            s = MagicMock()
            s.slot_id = f"slot-{i}"
            slots.append(s)
        return slots

    def test_success_writes_slots_to_shared(self):
        from app.tools.calendar_prefetch import CalendarPrefetchTool

        ctx = _make_ctx(collected={"purpose": "consultation"})
        slots = self._fake_slots(3)

        with patch("app.tools.calendar_prefetch.list_available_slots",
                   return_value=slots):
            result = asyncio.run(CalendarPrefetchTool().run(ctx))

        assert result.success is True
        assert result.data["slot_count"] == 3
        assert ctx.shared["prefetched_slots"] is slots
        assert len(ctx.shared["prefetched_slots"]) == 3

    def test_failure_does_not_write_prefetched_slots(self):
        from app.tools.calendar_prefetch import CalendarPrefetchTool

        ctx = _make_ctx()

        with patch("app.tools.calendar_prefetch.list_available_slots",
                   side_effect=Exception("calendar API down")):
            result = asyncio.run(CalendarPrefetchTool().run(ctx))

        assert result.success is False
        assert "prefetched_slots" not in ctx.shared


# ===========================================================================
# 7. ctx.shared is shared by reference
# ===========================================================================

class TestSharedContextReference:
    def test_shared_dict_is_same_object_across_contexts(self):
        from app.tools.base import ToolContext

        shared = {}
        ctx1 = _make_ctx(shared=shared)
        ctx2 = _make_ctx(shared=shared)

        ctx1.shared["from_tool_1"] = "hello"
        assert ctx2.shared["from_tool_1"] == "hello"

    def test_tool_output_visible_to_subsequent_tool(self):
        """Simulate sequential CALL_END tool execution reading shared state."""
        from app.tools.base import ToolContext

        shared = {}
        ctx1 = _make_ctx(shared=shared)
        ctx2 = _make_ctx(shared=shared)

        # Tool 1 writes
        ctx1.shared["caller_name"] = "Alice"
        ctx1.shared["ai_summary"] = "Needs help with contract."

        # Tool 2 reads (as WebhookTool would)
        assert ctx2.shared.get("caller_name") == "Alice"
        assert ctx2.shared.get("ai_summary") == "Needs help with contract."


# ===========================================================================
# 8. WorkflowDefinition accepts tools list
# ===========================================================================

class TestWorkflowDefinitionToolsList:
    def test_appointment_booking_loads_without_error(self):
        from app.agents.graph_config import APPOINTMENT_BOOKING

        assert APPOINTMENT_BOOKING is not None

    def test_tools_is_a_list(self):
        from app.agents.graph_config import APPOINTMENT_BOOKING

        assert isinstance(APPOINTMENT_BOOKING.tools, list)

    def test_tools_is_not_none(self):
        from app.agents.graph_config import APPOINTMENT_BOOKING

        assert APPOINTMENT_BOOKING.tools is not None
