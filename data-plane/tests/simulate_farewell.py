"""
Simulation tests for FarewellAgent and end-to-end farewell routing.

Tests:
1. FarewellAgent returns COMPLETED with a spoken farewell for any goodbye utterance.
2. Router correctly routes farewell utterances to the `farewell` agent (not data_collection,
   fallback, or any other node).
3. End-to-end: farewell mid-intake terminates the call cleanly via the graph edge.
4. End-to-end: farewell after scheduling terminates cleanly.
5. Edge cases: "Thank you" alone, multi-sentence with farewell at the end.

Run:
    python3 -m pytest tests/simulate_farewell.py -v -s
"""
from __future__ import annotations

import pytest
from app.agents.farewell import FarewellAgent
from app.agents.base import AgentStatus
from app.agents.graph_config import APPOINTMENT_BOOKING
from app.agents.workflow import WorkflowGraph
from app.agents.router import Router


# ── Shared config ─────────────────────────────────────────────────────────────

_CONFIG = {
    "assistant": {"persona_name": "Aria at Nexus Law"},
    "practice_areas": [{"name": "Immigration Law", "description": "H1B, visas, green cards."}],
    "parameters": [
        {"name": "full_name",    "display_label": "Full Name",    "data_type": "name",  "required": True,  "collection_order": 0},
        {"name": "phone_number", "display_label": "Phone Number", "data_type": "phone", "required": True,  "collection_order": 1},
        {"name": "email_address","display_label": "Email Address","data_type": "email", "required": False, "collection_order": 2},
    ],
    "faqs": [],
    "context_files": [],
    "global_policy_documents": [],
    "calendly_api_token": None,
    "calendly_event_types": [],
    "webhook_endpoints": [],
}


# ── Unit tests: FarewellAgent ─────────────────────────────────────────────────

class TestFarewellAgent:

    def _agent(self):
        return FarewellAgent()

    def test_returns_completed(self):
        agent = self._agent()
        resp = agent.process("Thank you, bye!", {}, _CONFIG, [])
        assert resp.status == AgentStatus.COMPLETED

    def test_speak_is_non_empty(self):
        agent = self._agent()
        resp = agent.process("Goodbye", {}, _CONFIG, [])
        assert resp.speak and len(resp.speak) > 5

    def test_various_farewells_all_complete(self):
        agent = self._agent()
        farewells = [
            "Thanks!",
            "Thank you so much.",
            "Bye.",
            "Okay, bye bye.",
            "Have a good day.",
            "That's all, thanks.",
            "Thank you. Bye.",
            "Okay. Thank you.",
            "I'm done, goodbye.",
        ]
        for utterance in farewells:
            resp = agent.process(utterance, {}, _CONFIG, [])
            assert resp.status == AgentStatus.COMPLETED, f"Expected COMPLETED for: {utterance!r}"

    def test_speak_does_not_ask_question(self):
        """Farewell response should close the call, not ask another question."""
        agent = self._agent()
        resp = agent.process("Thanks, goodbye!", {}, _CONFIG, [])
        # A farewell response should not end with a question mark
        assert not resp.speak.strip().endswith("?"), f"Unexpected question in farewell: {resp.speak!r}"

    def test_internal_state_preserved(self):
        """Agent must pass internal_state through unchanged."""
        agent = self._agent()
        state = {"full_name": "John", "stage": "collecting"}
        resp = agent.process("Bye!", state, _CONFIG, [])
        assert resp.internal_state == state


# ── Integration tests: graph node wiring ─────────────────────────────────────

class TestFarewellGraphNode:

    def test_farewell_node_exists(self):
        assert "farewell" in {n.id for n in APPOINTMENT_BOOKING.nodes}

    def test_farewell_is_interrupt_eligible(self):
        node = APPOINTMENT_BOOKING.node("farewell")
        assert node.interrupt_eligible is True

    def test_farewell_on_complete_goes_to_end(self):
        node = APPOINTMENT_BOOKING.node("farewell")
        assert node.on_complete.target == "end"

    def test_farewell_on_failed_goes_to_end(self):
        node = APPOINTMENT_BOOKING.node("farewell")
        assert node.on_failed.target == "end"

    def test_farewell_in_interrupt_policy(self):
        assert "farewell" in APPOINTMENT_BOOKING.interrupt_policy.eligible_agents

    def test_graph_validates(self):
        """WorkflowDefinition.__post_init__ raises on invalid edges — passing means graph is valid."""
        graph = WorkflowGraph(APPOINTMENT_BOOKING)
        assert "farewell" in graph.nodes


# ── Simulation tests: router routing ─────────────────────────────────────────

class TestFarewellRouting:
    """
    These tests exercise the real LLM-based router to confirm it routes
    farewell utterances to the `farewell` agent, not to data_collection or fallback.
    """

    def _router(self):
        graph = WorkflowGraph(APPOINTMENT_BOOKING)
        from app.agents.base import AgentStatus
        graph.states["data_collection"].status = AgentStatus.IN_PROGRESS
        return Router(graph), graph

    def test_sim_bye_routes_to_farewell(self):
        router, _ = self._router()
        agent_id, _ = router.select("Bye!", [], )
        assert agent_id == "farewell", f"Expected farewell, got {agent_id!r}"

    def test_sim_thank_you_bye_routes_to_farewell(self):
        router, _ = self._router()
        agent_id, _ = router.select("Thank you. Bye.", [])
        assert agent_id == "farewell", f"Expected farewell, got {agent_id!r}"

    def test_sim_thanks_so_much_routes_to_farewell(self):
        router, _ = self._router()
        agent_id, _ = router.select("Thanks so much, have a good day!", [])
        assert agent_id == "farewell", f"Expected farewell, got {agent_id!r}"

    def test_sim_okay_thank_you_routes_to_farewell(self):
        router, _ = self._router()
        agent_id, _ = router.select("Okay. Thank you.", [])
        assert agent_id == "farewell", f"Expected farewell, got {agent_id!r}"

    def test_sim_goodbye_routes_to_farewell(self):
        router, _ = self._router()
        agent_id, _ = router.select("Goodbye!", [])
        assert agent_id == "farewell", f"Expected farewell, got {agent_id!r}"

    def test_sim_providing_name_does_not_route_to_farewell(self):
        """'My name is John' should go to data_collection, not farewell."""
        router, _ = self._router()
        agent_id, _ = router.select("My name is John Smith.", [])
        assert agent_id != "farewell", f"Name utterance incorrectly routed to farewell"

    def test_sim_question_does_not_route_to_farewell(self):
        """'What are your fees?' should go to faq/fallback, not farewell."""
        router, _ = self._router()
        agent_id, _ = router.select("What are your fees?", [])
        assert agent_id != "farewell", f"FAQ utterance incorrectly routed to farewell"

    def test_sim_farewell_not_interrupt(self):
        """Farewell should NOT be an interrupt — it terminates, not suspends."""
        router, _ = self._router()
        _, interrupt = router.select("Thank you, bye!", [])
        assert interrupt is False, "Farewell should not set interrupt=True"


# ── End-to-end simulation: farewell mid-intake ────────────────────────────────

class TestFarewellEndToEnd:

    def _setup(self):
        from app.agents.registry import build_registry
        graph = WorkflowGraph(APPOINTMENT_BOOKING)
        router = Router(graph)
        registry = build_registry("sim-farewell", [])
        graph.states["data_collection"].status = AgentStatus.IN_PROGRESS
        return graph, router, registry

    def _invoke(self, agent_id, utterance, graph, registry, config, history):
        """Single synchronous agent invocation."""
        agent = registry[agent_id]
        state = graph.states[agent_id]
        response = agent.process(utterance, dict(state.internal_state), config, history)
        graph.update(agent_id, response, turn=1)
        return response

    def test_sim_farewell_mid_intake_completes(self):
        """Caller says bye while data_collection is still in progress."""
        graph, router, registry = self._setup()

        # Step 1: open with data_collection
        open_resp = self._invoke("data_collection", "", graph, registry, _CONFIG, [])
        assert open_resp.speak  # opening question asked

        # Step 2: caller says goodbye
        agent_id, _ = router.select("Okay, thank you. Bye!", [
            {"role": "assistant", "content": open_resp.speak},
        ])
        assert agent_id == "farewell"

        farewell_resp = self._invoke("farewell", "Okay, thank you. Bye!", graph, registry, _CONFIG, [])
        assert farewell_resp.status == AgentStatus.COMPLETED
        assert farewell_resp.speak

        # Edge should resolve to "end"
        edge = graph.get_edge("farewell", farewell_resp.status)
        assert edge.target == "end"
        assert edge.reason == "caller_farewell"

    def test_sim_farewell_after_booking_completes(self):
        """Caller says thank you after scheduling — call should end cleanly."""
        graph, router, registry = self._setup()

        # Mark scheduling as completed (simulates post-booking state)
        from app.agents.base import AgentStatus as AS
        for nid in ["data_collection", "narrative_collection", "intake_qualification", "scheduling"]:
            graph.states[nid].status = AS.COMPLETED

        agent_id, _ = router.select("Thank you so much, goodbye!", [
            {"role": "assistant", "content": "Your meeting has been scheduled. We look forward to speaking with you!"},
            {"role": "user", "content": "Thank you so much, goodbye!"},
        ])
        assert agent_id == "farewell"

        farewell_resp = self._invoke("farewell", "Thank you so much, goodbye!", graph, registry, _CONFIG, [])
        assert farewell_resp.status == AgentStatus.COMPLETED
        edge = graph.get_edge("farewell", farewell_resp.status)
        assert edge.target == "end"
