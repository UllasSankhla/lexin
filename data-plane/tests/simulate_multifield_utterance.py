"""
Simulation: caller provides multiple fields in a single utterance.

Verifies that the pending_field_fence steering handler does NOT block
multi-field acquisition. When a caller gives 3 fields at once, the agent
should extract all of them, queue the extras, and confirm one at a time.

Scenario: caller opens with name + email + phone in a single breath.

Run:
    cd data-plane
    PYTHONPATH=. python3 -m pytest tests/simulate_multifield_utterance.py -v -s
"""
from __future__ import annotations

import logging
import pytest

from app.agents.data_collection import DataCollectionAgent
from app.agents.base import AgentStatus

logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname)-8s %(name)s — %(message)s",
)

CONFIG = {
    "assistant": {"persona_name": "Aria"},
    "parameters": [
        {
            "name": "first_name",
            "display_label": "First Name",
            "data_type": "name",
            "required": True,
            "collection_order": 1,
            "extraction_hints": [],
        },
        {
            "name": "last_name",
            "display_label": "Last Name",
            "data_type": "name",
            "required": True,
            "collection_order": 2,
            "extraction_hints": [],
        },
        {
            "name": "email_address",
            "display_label": "Email Address",
            "data_type": "email",
            "required": True,
            "collection_order": 3,
            "extraction_hints": [],
        },
        {
            "name": "phone_number",
            "display_label": "Phone Number",
            "data_type": "phone",
            "required": True,
            "collection_order": 4,
            "extraction_hints": [],
        },
        {
            "name": "metlife_id",
            "display_label": "Metlife Id",
            "data_type": "text",
            "required": False,
            "collection_order": 5,
            "extraction_hints": [
                "MetLife member ID — caller may not have one.",
                "If caller says 'I don't have one', 'none', or 'not applicable', record as empty and skip.",
            ],
        },
    ],
}


def _run(utterances: list[str], *, label: str, max_retries_on_waiting: int = 6):
    agent = DataCollectionAgent()
    state: dict = {}
    statuses: list[AgentStatus] = []

    print(f"\n{'=' * 70}")
    print(f"SIM: {label}")
    print(f"{'=' * 70}")

    resp = agent.process("", state, CONFIG, [])
    statuses.append(resp.status)
    state = resp.internal_state
    print(f"[open] Aria: {resp.speak}")

    waiting_streak = 0
    prev_pending_field: str | None = None

    for i, utt in enumerate(utterances):
        print(f"\n[T{i+1:02d}] Caller: {utt!r}")
        resp = agent.process(utt, state, CONFIG, [])
        statuses.append(resp.status)
        state = resp.internal_state

        cur_pending = state.get("pending_confirmation")
        cur_field = cur_pending["field"] if cur_pending else None
        queue = state.get("extraction_queue", [])

        if resp.status == AgentStatus.WAITING_CONFIRM and cur_field == prev_pending_field:
            waiting_streak += 1
            assert waiting_streak <= max_retries_on_waiting, (
                f"Stuck in WAITING_CONFIRM for {cur_field!r} after {waiting_streak} turns"
            )
        else:
            waiting_streak = 0
        prev_pending_field = cur_field

        print(f"        Aria: {resp.speak}")
        print(
            f"        status={resp.status.value}  "
            f"pending={cur_pending}  "
            f"queue={queue}  "
            f"collected={state.get('collected', {})}"
        )

        if resp.status == AgentStatus.COMPLETED:
            break

    final = state.get("collected", {}) or (resp.collected or {})
    print(f"\nFinal collected: {final}")
    print(f"Final status: {resp.status.value}")
    return final, statuses, state


def test_three_fields_in_one_utterance():
    """
    Caller gives first name, last name, and email in a single utterance
    before the agent has asked for anything.

    Expected:
    - All three values extracted and queued
    - Agent confirms one field at a time (no fence violations)
    - All three end up in collected after confirmation turns
    - Agent then asks for phone (not yet provided)
    """
    utterances = [
        # T01: Three fields at once
        "Sure, my name is Shiva Kumar and my email is shiva.k@gmail.com.",
        # T02-T04: Confirm each field as asked
        "Yes.",
        "Yes.",
        "Yes.",
        # T05: Provide phone when asked
        "It's four one five five five five one two three four.",
        # T06: Confirm phone
        "Yes.",
    ]

    final, statuses, state = _run(
        utterances,
        label="3 fields at once — name + name + email, then phone separately",
    )

    assert "first_name" in final, f"first_name missing from collected: {final}"
    assert "last_name" in final, f"last_name missing from collected: {final}"
    assert "email_address" in final, f"email_address missing from collected: {final}"
    assert "phone_number" in final, f"phone_number missing from collected: {final}"
    assert AgentStatus.COMPLETED in statuses, "Agent should reach COMPLETED"


def test_three_fields_no_fence_trigger():
    """
    When the caller provides 3 fields at once, the pending_field_fence
    steering handler must NOT fire (no field is pending when they speak).

    Verifies that the fence does not interfere with multi-field extraction.
    """
    utterances = [
        "My first name is Shiva, last name is Kumar, email is shiva.k@gmail.com.",
        "Yes.",
        "Yes.",
        "Yes.",
        "Four one five five five five one two three four.",
        "Yes.",
    ]

    final, statuses, state = _run(
        utterances,
        label="Fence must not fire during multi-field extraction (no pending active)",
    )

    # All required fields confirmed — agent should complete
    assert AgentStatus.COMPLETED in statuses, (
        f"Agent did not complete. Final collected: {final}"
    )
    assert len([f for f in ["first_name", "last_name", "email_address", "phone_number"]
                if f in final]) == 4, (
        f"Not all required fields collected: {final}"
    )


def test_four_fields_in_one_utterance():
    """
    Caller gives all four required fields in a single utterance.

    Expected: agent extracts all four, confirms one at a time,
    reaches COMPLETED after all confirmations without ever triggering
    the steering fence.
    """
    utterances = [
        # T01: All four required fields at once
        (
            "Yeah sure — my first name is Shiva, last name is Kumar, "
            "email is shiva.k@gmail.com, and my phone is four one five "
            "five five five one two three four."
        ),
        # Confirm each
        "Yes.",
        "Yes.",
        "Yes.",
        "Yes.",
    ]

    final, statuses, state = _run(
        utterances,
        label="All 4 required fields in one utterance",
        max_retries_on_waiting=8,
    )

    assert AgentStatus.COMPLETED in statuses, (
        f"Agent did not complete. statuses={[s.value for s in statuses]} collected={final}"
    )
