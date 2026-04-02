"""
Simulation: "new" STT artifact triggering false MetLife ID extraction.

Reproduces the real call:
  1. Caller gives First Name "Olas" — confirmed
  2. Caller gives Last Name "Ravi" — AI reads back, pending confirmation active
  3. Caller says "new." (STT mishearing of "no")

Root cause found in logs:
  - dc_confirm_classify LLM call returns 256 output tokens but empty content
    (Cerebras API glitch on single-word inputs in JSON schema mode)
  - Exception handler fallback: ConfirmationSignal(signal="correct_or_add", is_affirmative=True)
  - Mega-prompt hint becomes "caller CONFIRMED the pending value and also provided
    new information. Handle both."
  - Mega-prompt locks in last_name="Ravi" (CONFIRMED) and extracts "new" into
    the next text field — MetLife ID

State is seeded directly to bypass the unrelated _template_question bug
that causes "Could I start with your full name" instead of "Last Name" —
we want to test exactly the moment the bug fires.

Run:
    cd data-plane
    PYTHONPATH=. python3 -m pytest tests/simulate_olas_new_as_no.py -v -s
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
logger = logging.getLogger(__name__)


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

# State at the exact moment the bug fires:
# first_name confirmed, last_name pending confirmation for "Ravi"
SEEDED_STATE = {
    "collected": {"first_name": "Olas"},
    "pending_confirmation": {"field": "last_name", "value": "Ravi"},
    "extraction_queue": [],
    "retry_count": 0,
    "history": [
        {"role": "assistant", "content": "Could I get your First Name, please?"},
        {"role": "user",      "content": "Yeah. My first name is Olas."},
        {"role": "assistant", "content": "I have Olas — is that correct?"},
        {"role": "user",      "content": "Yeah."},
        {"role": "assistant", "content": "Great, and could I get your Last Name, please?"},
        {"role": "user",      "content": "Uh, it's, uh, Ravi."},
        {"role": "assistant", "content": "I have Ravi — is that correct?"},
    ],
}


def test_new_is_not_metlife_id():
    """
    BUG REPRODUCTION: caller says 'new.' while last_name confirmation pending.

    The classifier fails on single-word input → fallback fires as
    correct_or_add/is_affirmative=True → mega-prompt CONFIRMS last_name
    AND extracts 'new' into metlife_id.

    This test should FAIL before the fix and PASS after.
    """
    agent = DataCollectionAgent()

    print("\n" + "=" * 70)
    print("SIM: 'new' as STT artifact for 'no' — must not populate MetLife ID")
    print("=" * 70)
    print(f"[seeded] pending = {SEEDED_STATE['pending_confirmation']}")
    print(f"[seeded] collected = {SEEDED_STATE['collected']}")

    resp = agent.process("new.", SEEDED_STATE, CONFIG, [])
    state = resp.internal_state or {}
    final = state.get("collected", {})
    pending = state.get("pending_confirmation")

    print(f"\n[T01] Caller: 'new.'")
    print(f"      Aria: {resp.speak}")
    print(f"      status={resp.status.value}  pending={pending}  collected={final}")

    assert final.get("metlife_id") != "new", (
        f"BUG REPRODUCED: 'new' (STT artifact for 'no') was incorrectly extracted "
        f"as metlife_id='{final.get('metlife_id')}' while last_name confirmation was pending."
    )


def test_new_does_not_confirm_last_name():
    """
    'new.' is a rejection of the last_name readback — last_name must NOT
    be recorded as confirmed after this utterance.

    This test should FAIL before the fix and PASS after.
    """
    agent = DataCollectionAgent()

    print("\n" + "=" * 70)
    print("SIM: 'new' must not confirm last_name='Ravi'")
    print("=" * 70)

    resp = agent.process("new.", SEEDED_STATE, CONFIG, [])
    state = resp.internal_state or {}
    final = state.get("collected", {})

    print(f"\n[T01] Caller: 'new.'")
    print(f"      Aria: {resp.speak}")
    print(f"      status={resp.status.value}  collected={final}")

    assert final.get("last_name") != "Ravi", (
        f"BUG: last_name was confirmed as 'Ravi' even though caller said 'new' (no)."
    )

    assert resp.status != AgentStatus.COMPLETED, (
        "Agent must not reach COMPLETED — last_name is not confirmed yet."
    )


def test_new_agent_re_asks_last_name():
    """
    After 'new.', the agent must re-ask for last_name (as a rejection),
    not advance to any other field.

    This test should FAIL before the fix and PASS after.
    """
    agent = DataCollectionAgent()

    print("\n" + "=" * 70)
    print("SIM: After 'new', agent must re-ask for last_name")
    print("=" * 70)

    resp = agent.process("new.", SEEDED_STATE, CONFIG, [])
    state = resp.internal_state or {}
    pending = state.get("pending_confirmation")

    print(f"\n[T01] Caller: 'new.'")
    print(f"      Aria: {resp.speak}")
    print(f"      status={resp.status.value}  pending={pending}")

    # After rejection the agent should be asking for last_name again
    # (either pending is None and it re-asks, or pending is still last_name)
    collected = state.get("collected", {})
    assert "last_name" not in collected, (
        f"last_name should not be in collected after rejection, got: {collected}"
    )
    assert "metlife_id" not in collected, (
        f"metlife_id should not be collected at all yet, got: {collected}"
    )
    # The speak should reference "last name" in some form
    speak_lower = resp.speak.lower()
    assert "last name" in speak_lower or "last" in speak_lower, (
        f"Expected agent to re-ask for last name, but got: {resp.speak!r}"
    )
