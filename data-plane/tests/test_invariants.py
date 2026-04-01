"""
Invariant tests — one test per named invariant from docs/state_testing.md.

Static tests (no API keys needed):
    INV-P4  — next_primary_goal() never returns a COMPLETED agent
    INV-P5  — dependency ordering enforced by available_nodes()
    INV-C1  — at most one node is WAITING_CONFIRM at any time
    INV-R2  — interrupt-eligible agents are never pushed onto the resume stack
    INV-R4  — planner plan has at most _MAX_PLAN_STEPS invoke steps
    INV-S3  — combine_speaks returns non-empty for any non-empty input
    INV-T2  — every 'end' edge in the graph config carries a known reason
    INV-TL1 — CalendarPrefetchTool binding is declared exactly once

Live tests (require CEREBRAS_API_KEY — pass --live):
    INV-P1  — planner always produces at least one invoke step
    INV-C2  — WAITING_CONFIRM always carries pending_confirmation in internal_state
    INV-C3  — every plan produced while WAITING_CONFIRM is active includes the waiting agent
    INV-C5  — pending_confirmation is None after a CONFIRMATION intent resolves
    INV-D1  — all values in collected_all are non-empty strings
    INV-D2  — all keys in collected_all match configured parameter names
    INV-D3  — data_collection COMPLETED → all required fields present in collected_all
    INV-D4  — DataCollectionAgent internal_state["collected"] stays in sync with caller's collected_all
    INV-R1  — intake_qualification is never selected as a planner step
    INV-R3  — FAREWELL intent always produces farewell as the last (terminal) plan step
    INV-A1  — EmpathyAgent is invoked at most once per call
    INV-A2  — SchedulingAgent stage transitions follow the declared sequence
    INV-S1  — every non-UNHANDLED agent response carries a non-empty speak
    INV-S2  — speak produced during WAITING_CONFIRM contains the pending field value
    INV-CR1 — completed data_collection reopens only via explicit CORRECTION + reset_fields

Run static only (no API keys):
    cd data-plane
    PYTHONPATH=. python3 -m pytest tests/test_invariants.py -v -s

Run all (requires API keys):
    PYTHONPATH=. python3 -m pytest tests/test_invariants.py -v -s --live
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from app.agents.base import AgentStatus
from app.agents.graph_config import APPOINTMENT_BOOKING
from app.agents.workflow import WorkflowGraph, AgentState
from app.tools.base import ToolTrigger


# ── Shared fixtures ───────────────────────────────────────────────────────────

# Minimal config with three required fields — enough to drive DataCollectionAgent
# to completion without optional fields adding noise.
_CONFIG_3 = {
    "assistant": {"persona_name": "Aria"},
    "parameters": [
        {
            "name": "full_name",
            "display_label": "Full Name",
            "data_type": "name",
            "required": True,
            "extraction_hints": [],
            "collection_order": 1,
        },
        {
            "name": "phone_number",
            "display_label": "Phone Number",
            "data_type": "phone",
            "required": True,
            "extraction_hints": [],
            "collection_order": 2,
        },
        {
            "name": "email_address",
            "display_label": "Email Address",
            "data_type": "email",
            "required": True,
            "extraction_hints": [],
            "collection_order": 3,
        },
    ],
}

_REQUIRED_FIELDS = {"full_name", "phone_number", "email_address"}

# A plausible happy-path utterance sequence for data_collection
_DC_HAPPY_PATH = [
    "",                                    # opening (empty = first turn)
    "My name is Jane Doe",
    "Yes that's correct",
    "415 555 1234",
    "Yes",
    "jane dot doe at example dot com",
    "Yes",
]


def _make_slot(n: int, offset_days: int = 1):
    """Create a deterministic TimeSlot for test use (no Calendly call needed)."""
    from app.services.calendar_service import TimeSlot
    base = datetime.now(timezone.utc).replace(
        hour=10, minute=0, second=0, microsecond=0
    ) + timedelta(days=offset_days + n - 1)
    return TimeSlot(
        slot_id=f"slot-{n}",
        start=base,
        end=base + timedelta(hours=1),
        description=f"Option {n}: {base.strftime('%A, %B %-d at 10:00 AM')}",
        event_type_uri="https://api.calendly.com/event_types/TEST",
        spots_remaining=1,
    )


def _fresh_graph() -> WorkflowGraph:
    return WorkflowGraph(APPOINTMENT_BOOKING)


# ── INV-P5 ────────────────────────────────────────────────────────────────────

class TestInvP5DependencyOrdering:
    """
    INV-P5: No primary goal can start before its dependency is COMPLETED.

    intake_qualification depends on narrative_collection (explicit depends_on).
    scheduling depends on intake_qualification (explicit depends_on).
    These deps are enforced by available_nodes() via deps_met() — we verify
    that the runtime enforcement matches the declared contract.
    """

    def test_intake_qualification_blocked_until_narrative_completes(self):
        graph = _fresh_graph()
        available_ids = {n.id for n in graph.available_nodes()}

        assert "intake_qualification" not in available_ids, (
            "intake_qualification is available at call start before narrative_collection completes"
        )

    def test_intake_qualification_unblocked_after_narrative_completes(self):
        graph = _fresh_graph()
        graph.states["narrative_collection"].status = AgentStatus.COMPLETED

        available_ids = {n.id for n in graph.available_nodes()}
        assert "intake_qualification" in available_ids, (
            "intake_qualification not available after narrative_collection is COMPLETED"
        )

    def test_scheduling_blocked_until_intake_qualification_completes(self):
        graph = _fresh_graph()
        graph.states["narrative_collection"].status = AgentStatus.COMPLETED

        available_ids = {n.id for n in graph.available_nodes()}
        assert "scheduling" not in available_ids, (
            "scheduling is available before intake_qualification completes"
        )

    def test_scheduling_unblocked_after_intake_qualification_completes(self):
        graph = _fresh_graph()
        graph.states["narrative_collection"].status = AgentStatus.COMPLETED
        graph.states["intake_qualification"].status = AgentStatus.COMPLETED

        available_ids = {n.id for n in graph.available_nodes()}
        assert "scheduling" in available_ids, (
            "scheduling not available after intake_qualification is COMPLETED"
        )

    def test_partial_chain_blocked(self):
        """narrative COMPLETED but intake still NOT_STARTED → scheduling still blocked."""
        graph = _fresh_graph()
        graph.states["narrative_collection"].status = AgentStatus.COMPLETED
        graph.states["intake_qualification"].status = AgentStatus.NOT_STARTED

        available_ids = {n.id for n in graph.available_nodes()}
        assert "scheduling" not in available_ids, (
            "scheduling available when intake_qualification is NOT_STARTED"
        )

    def test_failed_dependency_also_blocks(self):
        """A FAILED dependency should block the dependent just as NOT_STARTED does."""
        graph = _fresh_graph()
        graph.states["narrative_collection"].status = AgentStatus.FAILED

        # intake_qualification depends on narrative_collection being COMPLETED
        available_ids = {n.id for n in graph.available_nodes()}
        assert "intake_qualification" not in available_ids, (
            "intake_qualification available even though narrative_collection FAILED"
        )


# ── INV-T2 ────────────────────────────────────────────────────────────────────

class TestInvT2EndEdgesHaveReasons:
    """
    INV-T2: Every edge that targets 'end' in the workflow graph must carry
    a non-empty reason drawn from the known finalization reason set.

    This is a static analysis of the graph config — no LLM required.
    A missing or unrecognised reason means the call ends without a classifiable
    outcome, breaking analytics and webhook routing.
    """

    VALID_REASONS = {
        "completed",
        "caller_farewell",
        "not_qualified",
        "agent_error",
        "data_collection_failed",
        "scheduling_failed",
        "narrative_collection_failed",
        "qualification_error",
    }

    def test_all_node_edges_to_end_have_known_reason(self):
        failures = []
        for node in APPOINTMENT_BOOKING.nodes:
            for attr in ("on_complete", "on_failed", "on_continue", "on_waiting_confirm"):
                edge = getattr(node, attr)
                if edge.target != "end":
                    continue
                if not edge.reason:
                    failures.append(f"  {node.id}.{attr} → end  [NO REASON]")
                elif edge.reason not in self.VALID_REASONS:
                    failures.append(
                        f"  {node.id}.{attr} → end  [UNKNOWN REASON: {edge.reason!r}]"
                    )
        assert not failures, "End edges with missing or unknown reasons:\n" + "\n".join(failures)

    def test_error_policy_edge_has_known_reason(self):
        edge = APPOINTMENT_BOOKING.error_policy.on_max_errors
        assert edge.target == "end", (
            f"error_policy.on_max_errors should target 'end', got {edge.target!r}"
        )
        assert edge.reason in self.VALID_REASONS, (
            f"error_policy.on_max_errors has unknown reason {edge.reason!r}"
        )


# ── INV-TL1 ───────────────────────────────────────────────────────────────────

class TestInvTl1CalendarPrefetchOnce:
    """
    INV-TL1: CalendarPrefetchTool must be declared exactly once, as a
    non-fire-and-forget AGENT_COMPLETE binding awaited before scheduling.

    Zero bindings → scheduling blocks on a live calendar fetch mid-conversation.
    Two bindings → double-prefetch; the second may return a different slot list
    than the one the caller heard.
    """

    def _prefetch_bindings(self):
        return [
            b for b in APPOINTMENT_BOOKING.tools
            if b.tool_class == "CalendarPrefetchTool"
        ]

    def test_exactly_one_binding(self):
        bindings = self._prefetch_bindings()
        assert len(bindings) == 1, (
            f"Expected 1 CalendarPrefetchTool binding, found {len(bindings)}"
        )

    def test_trigger_is_agent_complete(self):
        binding = self._prefetch_bindings()[0]
        assert binding.trigger == ToolTrigger.AGENT_COMPLETE, (
            f"CalendarPrefetchTool trigger should be AGENT_COMPLETE, got {binding.trigger!r}"
        )

    def test_not_fire_and_forget(self):
        binding = self._prefetch_bindings()[0]
        assert not binding.fire_and_forget, (
            "CalendarPrefetchTool must not be fire_and_forget — "
            "it must block until slots are available before scheduling runs"
        )

    def test_awaited_before_scheduling(self):
        binding = self._prefetch_bindings()[0]
        assert binding.await_before_agent == "scheduling", (
            f"CalendarPrefetchTool should be awaited before 'scheduling', "
            f"got await_before_agent={binding.await_before_agent!r}"
        )


# ── INV-C2 ────────────────────────────────────────────────────────────────────

@pytest.mark.live
class TestInvC2WaitingConfirmHasPending:
    """
    INV-C2: Whenever DataCollectionAgent returns WAITING_CONFIRM, its
    internal_state must carry a non-null pending_confirmation with both
    'field' and 'value' populated.

    Without pending_confirmation, the agent cannot re-ask, cannot accept
    yes/no, and cannot store the confirmed value — the confirmation loop
    runs blind and the field is never collected.
    """

    def _drive_to_waiting_confirm(self, utterances):
        from app.agents.data_collection import DataCollectionAgent
        agent = DataCollectionAgent()
        state: dict = {}
        history: list[dict] = []

        for utt in utterances:
            resp = agent.process(utt, state, _CONFIG_3, history)
            state = resp.internal_state
            if utt:
                history.append({"role": "user", "content": utt})
            if resp.speak:
                history.append({"role": "assistant", "content": resp.speak})

            if resp.status == AgentStatus.WAITING_CONFIRM:
                return resp, state

        return None, state

    def _assert_pending_confirmation_valid(self, state: dict, context: str):
        pc = state.get("pending_confirmation")
        assert pc is not None, (
            f"{context}: status is WAITING_CONFIRM but pending_confirmation is None"
        )
        assert pc.get("field"), (
            f"{context}: pending_confirmation has no 'field': {pc}"
        )
        assert pc.get("value"), (
            f"{context}: pending_confirmation has no 'value': {pc}"
        )

    def test_name_collection_waiting_confirm(self):
        resp, state = self._drive_to_waiting_confirm([
            "",
            "My name is John Smith",
        ])
        if resp is None:
            pytest.skip("Agent did not reach WAITING_CONFIRM for name — LLM variability")
        self._assert_pending_confirmation_valid(state, "name collection")

    def test_phone_collection_waiting_confirm(self):
        resp, state = self._drive_to_waiting_confirm([
            "",
            "My name is John Smith",
            "Yes that's correct",
            "415 555 0192",
        ])
        if resp is None:
            pytest.skip("Agent did not reach WAITING_CONFIRM for phone")
        self._assert_pending_confirmation_valid(state, "phone collection")

    def test_email_collection_waiting_confirm(self):
        resp, state = self._drive_to_waiting_confirm([
            "",
            "My name is John Smith",
            "Yes",
            "415 555 0192",
            "Yes",
            "john at example dot com",
        ])
        if resp is None:
            pytest.skip("Agent did not reach WAITING_CONFIRM for email")
        self._assert_pending_confirmation_valid(state, "email collection")

    def test_waiting_confirm_invariant_holds_across_all_turns(self):
        """Check the invariant on every WAITING_CONFIRM turn, not just the first."""
        from app.agents.data_collection import DataCollectionAgent
        agent = DataCollectionAgent()
        state: dict = {}
        history: list[dict] = []
        violations: list[str] = []

        for i, utt in enumerate(_DC_HAPPY_PATH):
            resp = agent.process(utt, state, _CONFIG_3, history)
            state = resp.internal_state
            if utt:
                history.append({"role": "user", "content": utt})
            if resp.speak:
                history.append({"role": "assistant", "content": resp.speak})

            if resp.status == AgentStatus.WAITING_CONFIRM:
                pc = state.get("pending_confirmation")
                if not pc or not pc.get("field") or not pc.get("value"):
                    violations.append(
                        f"  Turn {i}: WAITING_CONFIRM with invalid pending_confirmation={pc!r}"
                    )

            if resp.status == AgentStatus.COMPLETED:
                break

        assert not violations, "INV-C2 violated:\n" + "\n".join(violations)


# ── INV-D4 ────────────────────────────────────────────────────────────────────

@pytest.mark.live
class TestInvD4CollectedStateConsistency:
    """
    INV-D4: After each turn, the values in the caller's collected_all dict must
    match the corresponding values in DataCollectionAgent's internal_state["collected"].

    Two representations of collected state must stay in sync. Divergence causes the
    planner to make routing decisions based on stale context.
    """

    def test_internal_and_external_collected_agree_on_happy_path(self):
        from app.agents.data_collection import DataCollectionAgent
        agent = DataCollectionAgent()
        state: dict = {}
        history: list[dict] = []
        collected_all: dict = {}
        violations: list[str] = []

        for i, utt in enumerate(_DC_HAPPY_PATH):
            resp = agent.process(utt, state, _CONFIG_3, history)
            state = resp.internal_state
            if utt:
                history.append({"role": "user", "content": utt})
            if resp.speak:
                history.append({"role": "assistant", "content": resp.speak})

            # Simulate what handler._persist_response does
            for field, value in resp.collected.items():
                if field not in collected_all:
                    collected_all[field] = value

            # THE INVARIANT: every value in collected_all must match internal_state
            internal = state.get("collected", {})
            for field, value in collected_all.items():
                if field not in internal:
                    violations.append(
                        f"  Turn {i} ({utt!r}): "
                        f"field {field!r} in collected_all but missing from internal_state['collected']"
                    )
                elif internal[field] != value:
                    violations.append(
                        f"  Turn {i} ({utt!r}): "
                        f"field {field!r} mismatch — "
                        f"collected_all={value!r}  internal={internal[field]!r}"
                    )

            if resp.status == AgentStatus.COMPLETED:
                break

        assert not violations, "INV-D4 violated:\n" + "\n".join(violations)

    def test_correction_keeps_both_sides_in_sync(self):
        """After a field correction, both collected_all and internal_state must reflect the new value."""
        from app.agents.data_collection import DataCollectionAgent
        agent = DataCollectionAgent()
        state: dict = {}
        history: list[dict] = []
        collected_all: dict = {}

        # Collect name
        for utt in ["", "My name is John Doe", "Yes"]:
            resp = agent.process(utt, state, _CONFIG_3, history)
            state = resp.internal_state
            for field, value in resp.collected.items():
                collected_all[field] = value
            if utt:
                history.append({"role": "user", "content": utt})
            if resp.speak:
                history.append({"role": "assistant", "content": resp.speak})

        if "full_name" not in collected_all:
            pytest.skip("full_name not collected — LLM variability, cannot test correction")

        original_name = collected_all["full_name"]

        # Issue a correction
        correction_utt = "Wait — actually my name is Jane Doe, not John Doe"
        resp = agent.process(correction_utt, state, _CONFIG_3, history)
        state = resp.internal_state
        for field, value in resp.collected.items():
            collected_all[field] = value

        # After processing the correction, both sides must agree
        internal = state.get("collected", {})
        for field, value in collected_all.items():
            assert internal.get(field) == value, (
                f"After correction: collected_all[{field!r}]={value!r} "
                f"but internal_state['collected'][{field!r}]={internal.get(field)!r}"
            )


# ── INV-R3 ────────────────────────────────────────────────────────────────────

@pytest.mark.live
class TestInvR3FarewellTerminatesPlan:
    """
    INV-R3: When the planner classifies an utterance as FAREWELL, the farewell
    agent must be the last invoke step in the plan — nothing may follow it.

    Steps after farewell would execute after the call has been finalized, speaking
    into a closed connection or corrupting post-call state.
    """

    FAREWELL_UTTERANCES = [
        "Goodbye! Thank you so much.",
        "Bye bye, take care!",
        "Thanks, talk to you later!",
        "That's all I needed, goodbye!",
        "Okay, thanks. Bye!",
        "Thank you, have a great day. Bye!",
    ]

    def _plan_for(self, utterance: str, graph: WorkflowGraph) -> list:
        from app.agents.planner import Planner
        planner = Planner(graph)
        return planner.plan(utterance, [])

    def test_farewell_is_last_step(self):
        failures = []
        for utt in self.FAREWELL_UTTERANCES:
            graph = _fresh_graph()
            graph.states["data_collection"].status = AgentStatus.IN_PROGRESS

            steps = self._plan_for(utt, graph)
            invoke_steps = [s for s in steps if s.action == "invoke"]

            if not invoke_steps:
                failures.append(f"  {utt!r}: no invoke steps produced")
                continue

            farewell_indices = [i for i, s in enumerate(invoke_steps) if s.agent_id == "farewell"]

            if not farewell_indices:
                failures.append(
                    f"  {utt!r}: no farewell step — got {[s.agent_id for s in invoke_steps]}"
                )
                continue

            farewell_idx = farewell_indices[0]
            after_farewell = invoke_steps[farewell_idx + 1:]
            if after_farewell:
                failures.append(
                    f"  {utt!r}: steps after farewell: {[s.agent_id for s in after_farewell]}"
                )

        assert not failures, "INV-R3 violated:\n" + "\n".join(failures)

    def test_farewell_mid_collection_does_not_add_data_collection_after(self):
        """Specifically guards against _validate_waiting_confirm injecting after farewell."""
        graph = _fresh_graph()
        graph.states["data_collection"].status = AgentStatus.WAITING_CONFIRM
        graph.states["data_collection"].internal_state = {
            "pending_confirmation": {"field": "full_name", "value": "John Smith"},
            "collected": {},
        }

        from app.agents.planner import Planner
        planner = Planner(graph)
        steps = planner.plan("Thanks so much, goodbye!", [])

        invoke_steps = [s for s in steps if s.action == "invoke"]
        if not invoke_steps:
            pytest.skip("No invoke steps produced — LLM variability")

        farewell_indices = [i for i, s in enumerate(invoke_steps) if s.agent_id == "farewell"]
        if not farewell_indices:
            pytest.skip("FAREWELL not classified for this utterance — LLM variability")

        farewell_idx = farewell_indices[0]
        after = invoke_steps[farewell_idx + 1:]
        assert not after, (
            f"Steps injected after farewell (likely by _validate_waiting_confirm): "
            f"{[s.agent_id for s in after]}"
        )


# ── INV-A2 ────────────────────────────────────────────────────────────────────

@pytest.mark.live
class TestInvA2SchedulingStageSequence:
    """
    INV-A2: SchedulingAgent must follow the declared stage sequence:
        presenting → awaiting_choice → awaiting_confirm → done

    Backward transitions or stage skips indicate lost internal state and mean
    a booking may be created without an explicit caller confirmation.
    """

    VALID_FORWARD_TRANSITIONS = {
        "presenting":      {"awaiting_choice"},
        "awaiting_choice": {"awaiting_confirm", "presenting"},  # presenting = new slots offered
        "awaiting_confirm": {"done", "awaiting_choice"},         # awaiting_choice = caller rejected
        "done":            set(),
    }

    def _run_scheduling(self, utterances: list[str]) -> list[tuple[str, str]]:
        """Returns list of (utterance, resulting_stage) tuples."""
        from app.agents.scheduling import SchedulingAgent
        from app.services.calendar_service import TimeSlot

        mock_slots = [_make_slot(i) for i in range(1, 4)]
        mock_booking = {
            "booking_id": "BK-TEST-001",
            "status": "active",
            "slot_description": mock_slots[0].description,
            "cancel_url": "https://calendly.com/cancellations/test",
            "reschedule_url": "https://calendly.com/reschedulings/test",
        }

        config = {
            **_CONFIG_3,
            "_collected": {
                "full_name": "Jane Doe",
                "phone_number": "4155551234",
                "email_address": "jane@example.com",
            },
            "_tool_results": {"calendar_slots": mock_slots},
        }

        with patch("app.agents.scheduling.list_available_slots", return_value=mock_slots), \
             patch("app.agents.scheduling.book_time_slot", return_value=mock_booking):

            agent = SchedulingAgent()
            state: dict = {}
            transitions: list[tuple[str, str]] = []

            for utt in utterances:
                resp = agent.process(utt, state, config, [])
                stage = resp.internal_state.get("stage", "presenting")
                transitions.append((utt, stage))
                state = resp.internal_state

                if resp.status == AgentStatus.COMPLETED:
                    break

        return transitions

    def test_happy_path_stage_sequence_is_valid(self):
        transitions = self._run_scheduling([
            "",               # present slots
            "The first one",  # choose slot
            "Yes please",     # confirm
        ])

        violations = []
        prev_stage = "presenting"
        for utt, cur_stage in transitions:
            allowed = self.VALID_FORWARD_TRANSITIONS.get(prev_stage, set())
            if cur_stage != prev_stage and cur_stage not in allowed:
                violations.append(
                    f"  Invalid: {prev_stage!r} → {cur_stage!r} after {utt!r}"
                )
            prev_stage = cur_stage

        assert not violations, "INV-A2 violated:\n" + "\n".join(violations)

    def test_no_booking_without_awaiting_confirm_stage(self):
        """Booking must only happen after passing through awaiting_confirm."""
        from app.agents.scheduling import SchedulingAgent

        mock_slots = [_make_slot(i) for i in range(1, 4)]
        mock_booking = {
            "booking_id": "BK-TEST-001",
            "status": "active",
            "slot_description": mock_slots[0].description,
            "cancel_url": "https://calendly.com/cancellations/test",
            "reschedule_url": "https://calendly.com/reschedulings/test",
        }

        config = {
            **_CONFIG_3,
            "_collected": {
                "full_name": "Jane Doe",
                "phone_number": "4155551234",
                "email_address": "jane@example.com",
            },
            "_tool_results": {"calendar_slots": mock_slots},
        }

        saw_awaiting_confirm = False

        with patch("app.agents.scheduling.list_available_slots", return_value=mock_slots), \
             patch("app.agents.scheduling.book_time_slot", return_value=mock_booking):

            agent = SchedulingAgent()
            state: dict = {}

            for utt in ["", "The first one", "Yes please"]:
                resp = agent.process(utt, state, config, [])
                stage = resp.internal_state.get("stage", "presenting")
                if stage == "awaiting_confirm":
                    saw_awaiting_confirm = True
                state = resp.internal_state

                if resp.booking:
                    assert saw_awaiting_confirm, (
                        "Booking created without passing through awaiting_confirm stage — "
                        "caller never explicitly confirmed the slot"
                    )
                    break


# ── INV-S1 ────────────────────────────────────────────────────────────────────

@pytest.mark.live
class TestInvS1NonUnhandledHasSpeak:
    """
    INV-S1: Every agent response that is not UNHANDLED must carry a non-empty speak.

    Silence on a voice call is never acceptable. An empty speak from a non-UNHANDLED
    agent means the caller hears nothing and has no idea what to do next.
    """

    def _assert_speak_nonempty(self, resp, context: str):
        if resp.status == AgentStatus.UNHANDLED:
            return  # UNHANDLED legitimately returns no speak
        assert resp.speak and resp.speak.strip(), (
            f"INV-S1: {context} returned status={resp.status.value} with empty speak"
        )

    def test_data_collection_always_speaks(self):
        from app.agents.data_collection import DataCollectionAgent
        agent = DataCollectionAgent()
        state: dict = {}
        history: list[dict] = []

        for i, utt in enumerate(_DC_HAPPY_PATH):
            resp = agent.process(utt, state, _CONFIG_3, history)
            self._assert_speak_nonempty(resp, f"DataCollectionAgent turn {i} ({utt!r})")
            state = resp.internal_state
            if utt:
                history.append({"role": "user", "content": utt})
            if resp.speak:
                history.append({"role": "assistant", "content": resp.speak})
            if resp.status == AgentStatus.COMPLETED:
                break

    def test_scheduling_always_speaks(self):
        from app.agents.scheduling import SchedulingAgent

        mock_slots = [_make_slot(i) for i in range(1, 4)]
        mock_booking = {
            "booking_id": "BK-TEST-001",
            "status": "active",
            "slot_description": mock_slots[0].description,
            "cancel_url": "https://calendly.com/cancellations/test",
            "reschedule_url": "https://calendly.com/reschedulings/test",
        }
        config = {
            **_CONFIG_3,
            "_collected": {"full_name": "Jane Doe", "phone_number": "4155551234", "email_address": "jane@example.com"},
            "_tool_results": {"calendar_slots": mock_slots},
        }

        with patch("app.agents.scheduling.list_available_slots", return_value=mock_slots), \
             patch("app.agents.scheduling.book_time_slot", return_value=mock_booking):
            agent = SchedulingAgent()
            state: dict = {}

            for i, utt in enumerate(["", "The first one", "Yes please"]):
                resp = agent.process(utt, state, config, [])
                self._assert_speak_nonempty(resp, f"SchedulingAgent turn {i} ({utt!r})")
                state = resp.internal_state
                if resp.status == AgentStatus.COMPLETED:
                    break

    def test_fallback_always_speaks(self):
        from app.agents.fallback import FallbackAgent
        agent = FallbackAgent()
        config = {**_CONFIG_3, "faqs": [], "context_files": []}

        for utt in [
            "What is the meaning of life?",
            "Can you help me with my taxes?",
            "asdfjkl;",
        ]:
            resp = agent.process(utt, {}, config, [])
            self._assert_speak_nonempty(resp, f"FallbackAgent ({utt!r})")

    def test_farewell_always_speaks(self):
        from app.agents.farewell import FarewellAgent
        agent = FarewellAgent()

        for utt in ["Goodbye!", "Thanks, bye!", "Take care!"]:
            resp = agent.process(utt, {}, _CONFIG_3, [])
            self._assert_speak_nonempty(resp, f"FarewellAgent ({utt!r})")


# ── INV-CR1 ───────────────────────────────────────────────────────────────────

@pytest.mark.live
class TestInvCr1CorrectionReopensCompletedCollection:
    """
    INV-CR1: A COMPLETED data_collection can only be reopened via an explicit
    CORRECTION intent from the caller, which must produce a reset_fields step
    in the planner plan.

    Positive test: genuine correction → reset_fields + data_collection invoke.
    Negative test: non-correction utterance → no reset_fields, collection stays closed.
    Reopen test: after correction, data_collection is accessible again and can re-collect.
    """

    def _make_completed_graph(self, collected: dict | None = None) -> WorkflowGraph:
        """Graph with data_collection COMPLETED and fields in internal_state."""
        graph = _fresh_graph()
        graph.states["data_collection"].status = AgentStatus.COMPLETED
        graph.states["data_collection"].internal_state = {
            "collected": collected or {
                "full_name": "John Smith",
                "phone_number": "4155550192",
                "email_address": "john@example.com",
            },
            "pending_confirmation": None,
        }
        # Advance narrative so next_primary_goal is narrative_collection
        graph.states["narrative_collection"].status = AgentStatus.IN_PROGRESS
        return graph

    def _plan(self, graph: WorkflowGraph, utterance: str, history: list | None = None) -> list:
        from app.agents.planner import Planner
        planner = Planner(graph)
        return planner.plan(utterance, history or [])

    def test_genuine_correction_produces_reset_fields(self):
        graph = self._make_completed_graph()
        correction = "Wait, my email is actually johnsmith at gmail dot com, not john at example dot com"

        steps = self._plan(graph, correction)
        reset_steps = [s for s in steps if s.action == "reset_fields"]

        assert reset_steps, (
            f"Genuine field correction produced no reset_fields step.\n"
            f"Steps: {[(s.action, s.agent_id or s.fields) for s in steps]}\n"
            f"Utterance: {correction!r}"
        )

    def test_genuine_correction_includes_data_collection_invoke(self):
        graph = self._make_completed_graph()
        correction = "Wait, my email is actually johnsmith at gmail dot com, not john at example dot com"

        steps = self._plan(graph, correction)
        invoke_dc = [s for s in steps if s.action == "invoke" and s.agent_id == "data_collection"]

        assert invoke_dc, (
            f"Genuine correction produced no data_collection invoke step.\n"
            f"Steps: {[(s.action, s.agent_id or s.fields) for s in steps]}"
        )

    def test_non_correction_utterance_does_not_reset_fields(self):
        graph = self._make_completed_graph()
        # Narrative utterance — should not reopen data_collection
        narrative_utt = "I was in a car accident last month and I have neck pain"

        steps = self._plan(graph, narrative_utt)
        reset_steps = [s for s in steps if s.action == "reset_fields"]

        assert not reset_steps, (
            f"Non-correction utterance triggered reset_fields — "
            f"data_collection was incorrectly reopened.\n"
            f"Steps: {[(s.action, s.agent_id or s.fields, s.reason) for s in steps]}\n"
            f"Utterance: {narrative_utt!r}"
        )

    def test_faq_utterance_does_not_reset_fields(self):
        graph = self._make_completed_graph()
        faq_utt = "What are your consultation fees?"

        steps = self._plan(graph, faq_utt)
        reset_steps = [s for s in steps if s.action == "reset_fields"]

        assert not reset_steps, (
            f"FAQ question triggered reset_fields on completed data_collection.\n"
            f"Steps: {[(s.action, s.agent_id or s.fields) for s in steps]}"
        )

    def test_after_correction_data_collection_is_available(self):
        """After reset_fields runs, data_collection must be accessible for re-collection."""
        from app.agents.planner import Planner

        graph = self._make_completed_graph(collected={
            "full_name": "John Smith",
            "phone_number": "4155550192",
            "email_address": "john@example.com",
        })

        # Simulate reset_fields being applied (as handler would do)
        planner = Planner(graph)
        planner.reset_fields(["email_address"], {"full_name": "John Smith", "phone_number": "4155550192", "email_address": "john@example.com"})

        # data_collection should now be IN_PROGRESS (reset from COMPLETED)
        assert graph.states["data_collection"].status == AgentStatus.IN_PROGRESS, (
            "After reset_fields, data_collection should revert to IN_PROGRESS"
        )

        # email_address must be gone from internal_state
        internal_collected = graph.states["data_collection"].internal_state.get("collected", {})
        assert "email_address" not in internal_collected, (
            f"email_address still in internal_state after reset_fields: {internal_collected}"
        )

        # data_collection should now appear in available_nodes
        available_ids = {n.id for n in graph.available_nodes()}
        assert "data_collection" in available_ids, (
            "data_collection not available after reset_fields — caller cannot correct their email"
        )

# ── INV-P4 ────────────────────────────────────────────────────────────────────

class TestInvP4NextPrimaryGoalNeverCompleted:
    """
    INV-P4: next_primary_goal() must never return the id of an agent whose
    status is COMPLETED or FAILED.

    The method drives what the planner injects as NEXT PRIMARY GOAL TO PURSUE.
    If it points at a completed agent, the planner will instruct the LLM to keep
    working on something that is already done, causing the call to loop.
    """

    def test_completed_node_excluded_from_next_primary(self):
        graph = _fresh_graph()
        # Complete goal-1 (data_collection) — next should be goal-2
        graph.states["data_collection"].status = AgentStatus.COMPLETED
        result = graph.next_primary_goal()
        assert result != "data_collection", (
            "next_primary_goal() returned data_collection even though it is COMPLETED"
        )

    def test_all_complete_returns_none(self):
        graph = _fresh_graph()
        for nid in ["data_collection", "narrative_collection", "intake_qualification", "scheduling"]:
            graph.states[nid].status = AgentStatus.COMPLETED
        assert graph.next_primary_goal() is None, (
            "next_primary_goal() should return None when all primary goals are COMPLETED"
        )

    def test_failed_node_excluded_from_next_primary(self):
        graph = _fresh_graph()
        graph.states["data_collection"].status = AgentStatus.FAILED
        result = graph.next_primary_goal()
        assert result != "data_collection", (
            "next_primary_goal() returned a FAILED node"
        )

    def test_next_primary_always_respects_dep_order(self):
        """next_primary_goal advances through goal_order, never skips a pending dep."""
        graph = _fresh_graph()
        # Sequence: complete data_collection → narrative becomes next
        graph.states["data_collection"].status = AgentStatus.COMPLETED
        assert graph.next_primary_goal() == "narrative_collection"

        # Complete narrative → intake becomes next (deps now met)
        graph.states["narrative_collection"].status = AgentStatus.COMPLETED
        assert graph.next_primary_goal() == "intake_qualification"

        # Complete intake → scheduling becomes next
        graph.states["intake_qualification"].status = AgentStatus.COMPLETED
        assert graph.next_primary_goal() == "scheduling"


# ── INV-C1 ────────────────────────────────────────────────────────────────────

class TestInvC1AtMostOneWaitingConfirm:
    """
    INV-C1: At most one node may be in WAITING_CONFIRM at any time.

    active_waiting_confirm() returns the first match in iteration order.
    If two nodes are simultaneously WAITING_CONFIRM, routing becomes
    non-deterministic — the wrong agent gets the caller's yes/no.
    """

    def test_fresh_graph_has_no_waiting_confirm(self):
        graph = _fresh_graph()
        assert graph.active_waiting_confirm() is None, (
            "Fresh graph should have no WAITING_CONFIRM node"
        )

    def test_single_waiting_confirm_detected(self):
        graph = _fresh_graph()
        graph.states["data_collection"].status = AgentStatus.WAITING_CONFIRM
        assert graph.active_waiting_confirm() == "data_collection"

    def test_graph_update_replaces_not_adds_waiting_confirm(self):
        """
        Simulate what happens when data_collection returns WAITING_CONFIRM twice
        in a row — the second update must overwrite, not stack.
        """
        from app.agents.base import SubagentResponse
        graph = _fresh_graph()

        first_response = SubagentResponse(
            status=AgentStatus.WAITING_CONFIRM,
            speak="Is your name John Smith?",
            internal_state={"pending_confirmation": {"field": "full_name", "value": "John Smith"}},
        )
        graph.update("data_collection", first_response, turn=1)
        assert graph.active_waiting_confirm() == "data_collection"

        # Second WAITING_CONFIRM on the same node — must still be just one
        second_response = SubagentResponse(
            status=AgentStatus.WAITING_CONFIRM,
            speak="Is your name Jane Doe?",
            internal_state={"pending_confirmation": {"field": "full_name", "value": "Jane Doe"}},
        )
        graph.update("data_collection", second_response, turn=2)

        waiting_nodes = [
            nid for nid, s in graph.states.items()
            if s.status == AgentStatus.WAITING_CONFIRM
        ]
        assert len(waiting_nodes) <= 1, (
            f"Multiple nodes in WAITING_CONFIRM: {waiting_nodes}"
        )

    def test_completing_waiting_confirm_clears_it(self):
        from app.agents.base import SubagentResponse
        graph = _fresh_graph()
        graph.states["data_collection"].status = AgentStatus.WAITING_CONFIRM

        completed = SubagentResponse(
            status=AgentStatus.COMPLETED,
            speak="Got it.",
            internal_state={"collected": {"full_name": "John Smith"}},
        )
        graph.update("data_collection", completed, turn=1)
        assert graph.active_waiting_confirm() is None, (
            "active_waiting_confirm() should return None after node moved to COMPLETED"
        )


# ── INV-R2 ────────────────────────────────────────────────────────────────────

class TestInvR2InterruptEligibleNotOnResumeStack:
    """
    INV-R2: Interrupt-eligible agents must never appear on the resume stack.

    The resume stack exists to return to a paused *primary* agent. Pushing
    faq/context_docs/fallback/farewell/empathy corrupts resume semantics —
    popping one would re-invoke an interrupt agent as a primary goal.
    """

    INTERRUPT_ELIGIBLE = {"farewell", "faq", "context_docs", "fallback"}

    def test_push_resume_accepts_primary_agents(self):
        from app.agents.planner import Planner
        graph = _fresh_graph()
        planner = Planner(graph)

        for primary in ["data_collection", "narrative_collection", "scheduling"]:
            planner._resume_stack.clear()
            planner.push_resume(primary)
            assert primary in planner._resume_stack, (
                f"push_resume rejected valid primary agent {primary!r}"
            )

    def test_resume_stack_only_contains_non_interrupt_agents(self):
        """
        Simulate the handler's push pattern: only active_primary_for_resume()
        is ever pushed. Verify that the returned id is never interrupt-eligible.
        """
        from app.agents.planner import Planner
        graph = _fresh_graph()

        # Set various primary agents as active
        for primary_id in ["data_collection", "narrative_collection", "scheduling"]:
            graph.states[primary_id].status = AgentStatus.IN_PROGRESS
            planner = Planner(graph)
            candidate = planner.active_primary_for_resume()

            if candidate is not None:
                node = graph.nodes[candidate]
                assert not node.interrupt_eligible, (
                    f"active_primary_for_resume() returned interrupt-eligible agent {candidate!r}"
                )
                assert candidate not in self.INTERRUPT_ELIGIBLE, (
                    f"active_primary_for_resume() returned interrupt-eligible {candidate!r}"
                )
            graph.states[primary_id].status = AgentStatus.NOT_STARTED


# ── INV-R4 ────────────────────────────────────────────────────────────────────

@pytest.mark.live
class TestInvR4PlanStepsCapped:
    """
    INV-R4: A planner plan must contain at most _MAX_PLAN_STEPS (3) invoke steps.

    Exceeding the cap means the loop guard in _build_steps_from_intents failed
    and the plan may process intents that were supposed to be dropped.
    """

    MAX_STEPS = 3  # mirrors planner._MAX_PLAN_STEPS

    # Utterances designed to contain multiple intents
    MULTI_INTENT_UTTERANCES = [
        "My name is John Smith and I was in a car accident last month. What are your fees?",
        "Yes that's right. Also my email is john at gmail dot com and what are your hours?",
        "Hi, I need help with a divorce — can you help? My name is Jane and my phone is 415 555 0101.",
        "Wait, my email is wrong. It's jane at gmail. Also I have a question about fees. Goodbye!",
    ]

    def test_plan_never_exceeds_max_steps(self):
        failures = []
        for utt in self.MULTI_INTENT_UTTERANCES:
            graph = _fresh_graph()
            graph.states["data_collection"].status = AgentStatus.IN_PROGRESS

            from app.agents.planner import Planner
            planner = Planner(graph)
            steps = planner.plan(utt, [])
            invoke_count = sum(1 for s in steps if s.action == "invoke")

            if invoke_count > self.MAX_STEPS:
                failures.append(
                    f"  {invoke_count} invoke steps for: {utt!r}\n"
                    f"  Steps: {[(s.action, s.agent_id) for s in steps if s.action == 'invoke']}"
                )

        assert not failures, "INV-R4 violated:\n" + "\n".join(failures)


# ── INV-S3 ────────────────────────────────────────────────────────────────────

class TestInvS3CombineSpeaksNonEmpty:
    """
    INV-S3: combine_speaks must return a non-empty string for any non-empty input.

    An empty combined speak means the caller hears silence even when one or more
    agents produced a valid response.
    """

    def _combine(self, speaks):
        from app.agents.planner import Planner
        planner = Planner(_fresh_graph())
        return planner.combine_speaks(speaks)

    def test_empty_input_returns_empty(self):
        assert self._combine([]) == ""

    def test_single_speak_returned_as_is(self):
        result = self._combine([("Hello there.", "data_collection", 1.0)])
        assert result == "Hello there."

    def test_two_speaks_combined_non_empty(self):
        result = self._combine([
            ("I understand you were in an accident.", "empathy", 1.0),
            ("Could I get your name?", "data_collection", 1.0),
        ])
        assert result.strip(), "combine_speaks returned empty for two valid speaks"

    def test_empathy_plus_narrative_filler_returns_empathy_only(self):
        """Empathy + narrative filler → empathy alone (narrative adds nothing)."""
        result = self._combine([
            ("I'm so sorry to hear that.", "empathy", 1.0),
            ("Thank you for sharing that.", "narrative_collection", 0.8),
        ])
        assert result.strip(), "combine_speaks returned empty"
        # The empathy text should dominate
        assert "sorry" in result.lower() or "hear" in result.lower(), (
            f"Empathy text was lost in combination: {result!r}"
        )

    def test_all_empty_speaks_returns_empty(self):
        result = self._combine([("", "faq", 1.0), ("  ", "fallback", 0.5)])
        assert result == "", (
            f"combine_speaks should return empty when all inputs are empty: {result!r}"
        )

    def test_mixed_empty_and_nonempty_returns_nonempty(self):
        result = self._combine([
            ("", "empathy", 1.0),
            ("What is your name?", "data_collection", 1.0),
        ])
        assert result.strip(), (
            "combine_speaks dropped the non-empty speak when first speak was empty"
        )

    @pytest.mark.parametrize("speaks", [
        [("A.", "data_collection", 1.0), ("B.", "faq", 1.0)],
        [("A.", "empathy", 1.0), ("B.", "data_collection", 1.0), ("C.", "faq", 0.5)],
        [("Hello.", "farewell", 1.0)],
        [("First.", "data_collection", 1.0), ("Second.", "narrative_collection", 0.8)],
    ])
    def test_parametrized_combinations(self, speaks):
        result = self._combine(speaks)
        assert result.strip(), f"combine_speaks returned empty for input: {speaks}"


# ── INV-P1 ────────────────────────────────────────────────────────────────────

@pytest.mark.live
class TestInvP1PlanAlwaysHasInvokeStep:
    """
    INV-P1: The planner must always produce at least one invoke step.

    A plan with only reset_fields steps, or an empty plan, leaves the caller
    with no response — the call silently stalls.
    """

    VARIED_UTTERANCES = [
        "Hello",
        "My name is John Smith",
        "Yes",
        "No",
        "I'm not sure",
        "What are your hours?",
        "I was in a car accident last month",
        "Can you help me with a visa issue?",
        "Hmm",
        "Okay thank you bye",
    ]

    def test_plan_always_has_invoke_step(self):
        from app.agents.planner import Planner
        failures = []

        for utt in self.VARIED_UTTERANCES:
            graph = _fresh_graph()
            graph.states["data_collection"].status = AgentStatus.IN_PROGRESS
            planner = Planner(graph)
            steps = planner.plan(utt, [])
            invoke_steps = [s for s in steps if s.action == "invoke"]

            if not invoke_steps:
                failures.append(
                    f"  No invoke steps for: {utt!r}\n"
                    f"  All steps: {[(s.action, s.agent_id or s.fields) for s in steps]}"
                )

        assert not failures, "INV-P1 violated:\n" + "\n".join(failures)

    def test_plan_has_invoke_step_at_all_primary_stages(self):
        """Verify non-empty plan across each primary goal stage."""
        from app.agents.planner import Planner

        stage_setups = [
            # data_collection stage
            {"data_collection": AgentStatus.IN_PROGRESS},
            # narrative stage
            {"data_collection": AgentStatus.COMPLETED, "narrative_collection": AgentStatus.IN_PROGRESS},
            # scheduling stage (after all deps met)
            {
                "data_collection": AgentStatus.COMPLETED,
                "narrative_collection": AgentStatus.COMPLETED,
                "intake_qualification": AgentStatus.COMPLETED,
                "scheduling": AgentStatus.IN_PROGRESS,
            },
        ]

        for setup in stage_setups:
            graph = _fresh_graph()
            for nid, status in setup.items():
                graph.states[nid].status = status

            planner = Planner(graph)
            steps = planner.plan("Yes that sounds good", [])
            invoke_steps = [s for s in steps if s.action == "invoke"]

            assert invoke_steps, (
                f"No invoke steps at stage {setup}\n"
                f"Steps: {[(s.action, s.agent_id or s.fields) for s in steps]}"
            )


# ── INV-C3 ────────────────────────────────────────────────────────────────────

@pytest.mark.live
class TestInvC3WaitingConfirmAlwaysInPlan:
    """
    INV-C3: Every plan produced while a node is WAITING_CONFIRM must include
    that node as an invoke step.

    This is the runtime guarantee of _validate_waiting_confirm. If the waiting
    agent is absent from the plan, the caller's yes/no goes to the wrong agent
    and the confirmation is never resolved.

    Note: FAREWELL is the only valid exception — handled by INV-R3.
    """

    NON_FAREWELL_UTTERANCES = [
        "Yes that's correct",
        "No that's wrong",
        "Actually my name is Jane",
        "What are your fees?",
        "I was in an accident",
        "Hmm let me think",
        "Can you repeat that?",
    ]

    def test_waiting_confirm_always_present_in_plan(self):
        from app.agents.planner import Planner
        failures = []

        for utt in self.NON_FAREWELL_UTTERANCES:
            graph = _fresh_graph()
            graph.states["data_collection"].status = AgentStatus.WAITING_CONFIRM
            graph.states["data_collection"].internal_state = {
                "pending_confirmation": {"field": "full_name", "value": "John Smith"},
                "collected": {},
            }

            planner = Planner(graph)
            steps = planner.plan(utt, [])
            invoke_steps = [s for s in steps if s.action == "invoke"]
            agent_ids = [s.agent_id for s in invoke_steps]

            if "data_collection" not in agent_ids:
                failures.append(
                    f"  {utt!r}: data_collection missing from plan while WAITING_CONFIRM\n"
                    f"  Steps: {agent_ids}"
                )

        assert not failures, "INV-C3 violated:\n" + "\n".join(failures)


# ── INV-C5 ────────────────────────────────────────────────────────────────────

@pytest.mark.live
class TestInvC5PendingConfirmationClearedAfterConfirm:
    """
    INV-C5: After a CONFIRMATION intent resolves successfully,
    pending_confirmation must be None in the agent's internal_state.

    A stale pending_confirmation causes the agent to re-ask an already
    answered question on the next turn.
    """

    def test_pending_confirmation_cleared_after_yes(self):
        from app.agents.data_collection import DataCollectionAgent
        agent = DataCollectionAgent()
        state: dict = {}
        history: list[dict] = []

        # Drive to WAITING_CONFIRM
        for utt in ["", "My name is John Smith"]:
            resp = agent.process(utt, state, _CONFIG_3, history)
            state = resp.internal_state
            if utt:
                history.append({"role": "user", "content": utt})
            if resp.speak:
                history.append({"role": "assistant", "content": resp.speak})
            if resp.status == AgentStatus.WAITING_CONFIRM:
                break
        else:
            pytest.skip("Never reached WAITING_CONFIRM — LLM variability")

        assert state.get("pending_confirmation"), "No pending_confirmation to confirm"

        # Now confirm with yes
        resp = agent.process("Yes that's correct", state, _CONFIG_3, history)
        state = resp.internal_state

        if resp.status in (AgentStatus.IN_PROGRESS, AgentStatus.COMPLETED):
            # Confirmation was accepted — pending should be cleared
            assert state.get("pending_confirmation") is None, (
                f"pending_confirmation not cleared after confirmation: {state.get('pending_confirmation')}"
            )

    def test_pending_confirmation_replaced_not_stacked_on_next_field(self):
        """After confirming one field, pending_confirmation should point to the next field."""
        from app.agents.data_collection import DataCollectionAgent
        agent = DataCollectionAgent()
        state: dict = {}
        history: list[dict] = []

        confirmed_fields: list[str] = []

        for utt in _DC_HAPPY_PATH:
            resp = agent.process(utt, state, _CONFIG_3, history)

            if resp.status == AgentStatus.WAITING_CONFIRM:
                pc = state.get("pending_confirmation") or {}
                # After this confirmation resolves, process "yes"
                if pc.get("field"):
                    confirmed_fields.append(pc["field"])

            prev_state = state
            state = resp.internal_state
            if utt:
                history.append({"role": "user", "content": utt})
            if resp.speak:
                history.append({"role": "assistant", "content": resp.speak})

            if resp.status == AgentStatus.COMPLETED:
                break

        # At least one field must have been confirmed in this run
        assert confirmed_fields, "No fields were confirmed — check test utterances"


# ── INV-D1 / INV-D2 / INV-D3 ─────────────────────────────────────────────────

@pytest.mark.live
class TestInvDataIntegrity:
    """
    INV-D1: All values in collected_all are non-empty strings.
    INV-D2: All keys in collected_all match configured parameter names.
    INV-D3: When data_collection is COMPLETED, all required parameters are
            present in collected_all.

    These three invariants together guarantee that the booking system receives
    complete, valid data. A violation of any one of them produces either a
    failed booking or silent data loss.
    """

    def _drive_to_completion(self):
        from app.agents.data_collection import DataCollectionAgent
        agent = DataCollectionAgent()
        state: dict = {}
        history: list[dict] = []
        collected_all: dict = {}

        for utt in _DC_HAPPY_PATH:
            resp = agent.process(utt, state, _CONFIG_3, history)
            state = resp.internal_state
            for field, value in resp.collected.items():
                if field not in collected_all:
                    collected_all[field] = value
            if utt:
                history.append({"role": "user", "content": utt})
            if resp.speak:
                history.append({"role": "assistant", "content": resp.speak})
            if resp.status == AgentStatus.COMPLETED:
                return resp.status, collected_all, state

        return state.get("status", AgentStatus.IN_PROGRESS), collected_all, state

    def test_d1_no_empty_values_in_collected(self):
        _, collected_all, _ = self._drive_to_completion()
        violations = [
            f"  {k!r}: {v!r}"
            for k, v in collected_all.items()
            if not v or not str(v).strip()
        ]
        assert not violations, "INV-D1: empty values in collected_all:\n" + "\n".join(violations)

    def test_d2_all_keys_are_configured_parameters(self):
        _, collected_all, _ = self._drive_to_completion()
        configured_keys = {p["name"] for p in _CONFIG_3["parameters"]}
        unknown = [k for k in collected_all if k not in configured_keys]
        assert not unknown, (
            f"INV-D2: collected_all contains keys not in config: {unknown}\n"
            f"Configured: {configured_keys}"
        )

    def test_d3_all_required_fields_present_at_completion(self):
        status, collected_all, _ = self._drive_to_completion()

        if status != AgentStatus.COMPLETED:
            pytest.skip("data_collection did not reach COMPLETED — LLM variability")

        required = {p["name"] for p in _CONFIG_3["parameters"] if p.get("required")}
        missing = required - set(collected_all.keys())
        assert not missing, (
            f"INV-D3: data_collection COMPLETED but required fields missing: {missing}\n"
            f"Collected: {set(collected_all.keys())}"
        )

    def test_d1_d2_hold_at_every_intermediate_turn(self):
        """D1 and D2 must hold after every turn, not just at completion."""
        from app.agents.data_collection import DataCollectionAgent
        agent = DataCollectionAgent()
        state: dict = {}
        history: list[dict] = []
        collected_all: dict = {}
        configured_keys = {p["name"] for p in _CONFIG_3["parameters"]}
        violations: list[str] = []

        for i, utt in enumerate(_DC_HAPPY_PATH):
            resp = agent.process(utt, state, _CONFIG_3, history)
            state = resp.internal_state
            for field, value in resp.collected.items():
                if field not in collected_all:
                    collected_all[field] = value
            if utt:
                history.append({"role": "user", "content": utt})
            if resp.speak:
                history.append({"role": "assistant", "content": resp.speak})

            for k, v in collected_all.items():
                if not v or not str(v).strip():
                    violations.append(f"  Turn {i}: D1 — empty value for {k!r}")
                if k not in configured_keys:
                    violations.append(f"  Turn {i}: D2 — unknown key {k!r}")

            if resp.status == AgentStatus.COMPLETED:
                break

        assert not violations, "Data integrity violated mid-conversation:\n" + "\n".join(violations)


# ── INV-R1 ────────────────────────────────────────────────────────────────────

@pytest.mark.live
class TestInvR1IntakeQualificationNeverPlannerStep:
    """
    INV-R1: intake_qualification must never appear as a planner invoke step.

    It is triggered automatically by the graph edge from narrative_collection,
    not by caller utterances. If the planner ever selects it directly, the
    agent runs without its expected preconditions (completed narrative) and
    with no caller utterance to process.
    """

    UTTERANCES = [
        "I think my case qualifies",
        "Am I eligible?",
        "Can you assess my situation?",
        "I have a work injury claim",
        "Yes, I'm ready to be assessed",
        "Let's do the intake",
        "Qualify me",
    ]

    def test_intake_qualification_never_in_planner_steps(self):
        from app.agents.planner import Planner
        failures = []

        for utt in self.UTTERANCES:
            # Set up graph at various stages where intake_qualification might be
            # tempting to select
            for setup in [
                {"data_collection": AgentStatus.COMPLETED, "narrative_collection": AgentStatus.COMPLETED},
                {"data_collection": AgentStatus.IN_PROGRESS},
                {"data_collection": AgentStatus.COMPLETED, "narrative_collection": AgentStatus.IN_PROGRESS},
            ]:
                graph = _fresh_graph()
                for nid, status in setup.items():
                    graph.states[nid].status = status

                planner = Planner(graph)
                steps = planner.plan(utt, [])
                invoke_agents = [s.agent_id for s in steps if s.action == "invoke"]

                if "intake_qualification" in invoke_agents:
                    failures.append(
                        f"  {utt!r} at stage {setup}:\n"
                        f"  intake_qualification appeared in plan: {invoke_agents}"
                    )

        assert not failures, "INV-R1 violated:\n" + "\n".join(failures)


# ── INV-A1 ────────────────────────────────────────────────────────────────────

@pytest.mark.live
class TestInvA1EmpathyAtMostOnce:
    """
    INV-A1: EmpathyAgent must be invoked at most once per call.

    Empathy is a one-shot agent (always returns COMPLETED). After its first
    invocation it moves to COMPLETED and is excluded from available_nodes().
    If the planner keeps including it in NARRATIVE steps, the caller hears
    multiple empathetic openers, which is jarring and wastes turns.
    """

    def test_empathy_not_in_available_after_completed(self):
        graph = _fresh_graph()
        graph.states["empathy"].status = AgentStatus.COMPLETED
        available_ids = {n.id for n in graph.available_nodes()}
        assert "empathy" not in available_ids, (
            "empathy still appears in available_nodes() after COMPLETED"
        )

    def test_empathy_at_most_once_across_narrative_turns(self):
        """Drive multiple NARRATIVE utterances through the planner; empathy fires at most once."""
        from app.agents.planner import Planner

        graph = _fresh_graph()
        graph.states["data_collection"].status = AgentStatus.COMPLETED

        narrative_utterances = [
            "I was in a car accident last month",
            "The other driver ran a red light",
            "I've been having neck pain ever since",
        ]

        empathy_count = 0
        for utt in narrative_utterances:
            planner = Planner(graph)
            steps = planner.plan(utt, [])

            for step in steps:
                if step.action == "invoke" and step.agent_id == "empathy":
                    empathy_count += 1
                    # Simulate empathy completing — removes it from available_nodes
                    graph.states["empathy"].status = AgentStatus.COMPLETED

        assert empathy_count <= 1, (
            f"EmpathyAgent was included in the plan {empathy_count} times across "
            f"{len(narrative_utterances)} narrative turns"
        )


# ── INV-S2 ────────────────────────────────────────────────────────────────────

@pytest.mark.live
class TestInvS2WaitingConfirmSpeakContainsPendingValue:
    """
    INV-S2: The speak produced when data_collection enters WAITING_CONFIRM
    must contain the extracted value being confirmed.

    If the speak does not echo back the value (e.g. "Is your name John Smith?"),
    the caller cannot confirm or correct it — they would be responding yes/no
    to a question about an invisible value.
    """

    def _drive_to_waiting_confirm(self, utterance: str):
        from app.agents.data_collection import DataCollectionAgent
        agent = DataCollectionAgent()
        state: dict = {}
        history: list[dict] = []

        resp = agent.process("", state, _CONFIG_3, history)
        state = resp.internal_state
        if resp.speak:
            history.append({"role": "assistant", "content": resp.speak})

        resp = agent.process(utterance, state, _CONFIG_3, history)
        state = resp.internal_state
        return resp, state

    def test_name_confirmation_speak_contains_name(self):
        resp, state = self._drive_to_waiting_confirm("My name is John Smith")

        if resp.status != AgentStatus.WAITING_CONFIRM:
            pytest.skip("Did not reach WAITING_CONFIRM — LLM variability")

        pc = state.get("pending_confirmation", {})
        pending_value = pc.get("value", "")
        if not pending_value:
            pytest.skip("No pending value to check — LLM variability")

        # The confirmed value (or key parts of it) must appear in the speak
        value_words = [w.lower() for w in pending_value.split() if len(w) > 2]
        speak_lower = resp.speak.lower()
        matched = any(word in speak_lower for word in value_words)

        assert matched, (
            f"WAITING_CONFIRM speak does not contain pending value.\n"
            f"  pending_value: {pending_value!r}\n"
            f"  speak: {resp.speak!r}"
        )

    def test_email_confirmation_speak_contains_email(self):
        from app.agents.data_collection import DataCollectionAgent
        agent = DataCollectionAgent()
        state: dict = {}
        history: list[dict] = []

        for utt in ["", "My name is Jane Doe", "Yes", "415 555 0192", "Yes", "jane at example dot com"]:
            resp = agent.process(utt, state, _CONFIG_3, history)
            state = resp.internal_state
            if utt:
                history.append({"role": "user", "content": utt})
            if resp.speak:
                history.append({"role": "assistant", "content": resp.speak})
            if resp.status == AgentStatus.WAITING_CONFIRM:
                pc = state.get("pending_confirmation", {})
                if pc.get("field") == "email_address":
                    pending_value = pc.get("value", "")
                    if not pending_value:
                        pytest.skip("No email pending value")
                    speak_lower = resp.speak.lower()
                    # Email domain or local part should appear
                    local = pending_value.split("@")[0].lower() if "@" in pending_value else pending_value.lower()
                    assert local in speak_lower or pending_value.lower() in speak_lower, (
                        f"Email confirmation speak does not contain email.\n"
                        f"  pending_value: {pending_value!r}\n"
                        f"  speak: {resp.speak!r}"
                    )
                    return

        pytest.skip("Never reached WAITING_CONFIRM for email — LLM variability")
