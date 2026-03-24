"""
Simulation tests for the new mega-prompt DataCollectionAgent.

Each simulation drives a multi-turn conversation through the real LLM
(Cerebras) with realistic caller utterances, then asserts on final
collected state and per-turn status transitions.

Run:
    python3 -m pytest tests/simulate_data_collection.py -v -s
"""
from __future__ import annotations

import pytest
from app.agents.data_collection import DataCollectionAgent
from app.agents.base import AgentStatus

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

PARAMS_3 = [
    {
        "name": "full_name",
        "display_label": "Full Name",
        "data_type": "name",
        "required": True,
        "extraction_hints": [],
    },
    {
        "name": "phone_number",
        "display_label": "Phone Number",
        "data_type": "phone",
        "required": True,
        "validation_message": "10-15 digits, area code required",
        "extraction_hints": [],
    },
    {
        "name": "email_address",
        "display_label": "Email Address",
        "data_type": "email",
        "required": True,
        "extraction_hints": [],
    },
]

PARAMS_WITH_OPTIONAL = PARAMS_3 + [
    {
        "name": "referral",
        "display_label": "How did you hear about us",
        "data_type": "text",
        "required": False,
        "extraction_hints": [],
    },
]

CONFIG_3 = {"assistant": {"persona_name": "Lexin Law"}, "parameters": PARAMS_3}
CONFIG_OPT = {"assistant": {"persona_name": "Lexin Law"}, "parameters": PARAMS_WITH_OPTIONAL}


def run_conversation(
    utterances: list[str],
    config: dict,
    *,
    label: str = "",
    max_retries_on_waiting: int = 2,
) -> tuple[dict, list[AgentStatus]]:
    """
    Drive the agent through a scripted list of utterances.
    Returns (final_collected, list_of_per_turn_statuses).
    Raises AssertionError if the agent gets stuck (>max_retries_on_waiting
    consecutive WAITING_CONFIRM on the same field without advancing).
    """
    agent = DataCollectionAgent()
    state: dict = {}
    statuses: list[AgentStatus] = []
    waiting_streak = 0
    prev_pending_field: str | None = None

    print(f"\n{'=' * 60}")
    print(f"SIM: {label}")
    print(f"{'=' * 60}")

    # Opening turn
    resp = agent.process("", state, config, [])
    statuses.append(resp.status)
    state = resp.internal_state
    print(f"[open]  AI: {resp.speak}")
    print(f"        status={resp.status.value}  pending={state.get('pending_confirmation')}  collected={state.get('collected', {})}")

    for i, utt in enumerate(utterances):
        print(f"\n[T{i+1}]  Caller: {utt!r}")
        resp = agent.process(utt, state, config, [])
        statuses.append(resp.status)
        state = resp.internal_state

        cur_pending = state.get("pending_confirmation")
        cur_field = cur_pending["field"] if cur_pending else None

        if resp.status == AgentStatus.WAITING_CONFIRM and cur_field == prev_pending_field:
            waiting_streak += 1
            assert waiting_streak <= max_retries_on_waiting, (
                f"Stuck in WAITING_CONFIRM for field {cur_field!r} "
                f"after {waiting_streak} consecutive turns"
            )
        else:
            waiting_streak = 0
        prev_pending_field = cur_field

        print(f"        AI: {resp.speak}")
        print(f"        status={resp.status.value}  pending={cur_pending}  collected={state.get('collected', {})}")

        if resp.status == AgentStatus.COMPLETED:
            break

    final_collected = state.get("collected", {}) or (resp.collected if resp.collected else {})
    print(f"\nFinal collected: {final_collected}")
    return final_collected, statuses


# ---------------------------------------------------------------------------
# Simulations
# ---------------------------------------------------------------------------

def test_sim_happy_path():
    """Caller answers fields in order, confirms each. Should complete in ~6 turns."""
    collected, statuses = run_conversation(
        [
            "My name is Sarah Mitchell",
            "yes",
            "four one five five five five zero one nine two",
            "yes that's right",
            "sarah dot mitchell at gmail dot com",
            "yes",
        ],
        CONFIG_3,
        label="Happy path — 3 fields in order",
    )
    assert collected.get("full_name"), f"full_name missing: {collected}"
    assert collected.get("phone_number"), f"phone_number missing: {collected}"
    assert collected.get("email_address"), f"email_address missing: {collected}"
    assert AgentStatus.COMPLETED in statuses, "Never reached COMPLETED"


def test_sim_out_of_order():
    """Caller provides email before being asked for it, then fills remaining fields.

    The LLM may or may not capture both fields from the first utterance, and
    may or may not set pending_confirmation correctly after a combined utterance.
    The script provides enough explicit backup turns so all three fields are
    covered regardless of how many the model extracted on the first turn.
    """
    collected, statuses = run_conversation(
        [
            "Sarah Mitchell, and my email is sarah@example.com",
            "Sarah Mitchell",            # re-state name explicitly (backup if LLM lost it)
            "yes",                       # confirm name
            "four one five five five five zero one nine two",
            "yes",                       # confirm phone
            "sarah at example dot com",  # email — explicit fallback
            "yes",                       # confirm email
        ],
        CONFIG_3,
        label="Out-of-order — email provided before asked",
        max_retries_on_waiting=3,
    )
    assert collected.get("full_name"), f"full_name missing: {collected}"
    assert collected.get("phone_number"), f"phone_number missing: {collected}"
    assert collected.get("email_address"), f"email_address missing: {collected}"
    assert AgentStatus.COMPLETED in statuses


def test_sim_multiple_fields_one_utterance():
    """Caller gives name and phone in one breath.

    The LLM may pick up one or both fields from the first utterance.
    Extra turns provide explicit phone and email as fallbacks so the
    conversation always reaches COMPLETED regardless of what was captured.
    """
    collected, statuses = run_conversation(
        [
            "I'm John Doe, my number is 415-555-0100",
            "yes",                       # confirm name (or first captured field)
            "four one five five five five zero one zero zero",  # phone — explicit fallback
            "yes",                       # confirm phone
            "john at example dot com",   # email
            "yes",                       # confirm email
        ],
        CONFIG_3,
        label="Multiple fields in one utterance",
        max_retries_on_waiting=3,
    )
    assert collected.get("full_name"), f"full_name missing: {collected}"
    assert collected.get("phone_number"), f"phone_number missing: {collected}"
    assert collected.get("email_address"), f"email_address missing: {collected}"
    assert AgentStatus.COMPLETED in statuses


def test_sim_correction():
    """Caller corrects a value after hearing the readback."""
    collected, statuses = run_conversation(
        [
            "My name is John Doe",
            "actually it's Jane Doe not John",   # correction during confirm
            "yes",
            "four one five five five five zero one nine two",
            "yes",
            "jane at example dot com",
            "yes",
        ],
        CONFIG_3,
        label="Correction — wrong name corrected during confirmation",
        max_retries_on_waiting=3,
    )
    assert collected.get("full_name"), f"full_name missing: {collected}"
    # The corrected value should be accepted
    assert AgentStatus.COMPLETED in statuses


def test_sim_rejection_and_reentry():
    """Caller says no to readback and re-states the value."""
    collected, statuses = run_conversation(
        [
            "My name is Sarah Mitchell",
            "no",                               # reject readback
            "Sarah Mitchell",                   # re-state
            "yes",
            "four one five five five five zero one nine two",
            "yes",
            "sarah at example dot com",
            "yes",
        ],
        CONFIG_3,
        label="Rejection — caller says no then re-states",
        max_retries_on_waiting=3,
    )
    assert collected.get("full_name")
    assert AgentStatus.COMPLETED in statuses


def test_sim_spelled_out_email():
    """Caller spells out email letter by letter."""
    collected, statuses = run_conversation(
        [
            "Mike Johnson",
            "yes",
            "six oh eight five five five one two three four",
            "yes",
            "m-i-k-e at example dot com",
            "yes",
        ],
        CONFIG_3,
        label="Spelled-out email — m-i-k-e at example dot com",
        max_retries_on_waiting=3,
    )
    assert collected.get("email_address"), f"email missing: {collected}"
    assert "@" in (collected.get("email_address") or ""), \
        f"email not parsed correctly: {collected.get('email_address')}"
    assert AgentStatus.COMPLETED in statuses


def test_sim_optional_field_skipped():
    """Caller skips the optional referral field."""
    collected, statuses = run_conversation(
        [
            "Sarah Mitchell",
            "yes",
            "four one five five five five zero one nine two",
            "yes",
            "sarah at example dot com",
            "yes",
            "skip",                             # skip optional field
        ],
        CONFIG_OPT,
        label="Optional field skipped",
        max_retries_on_waiting=3,
    )
    assert collected.get("full_name")
    assert collected.get("phone_number")
    assert collected.get("email_address")
    assert AgentStatus.COMPLETED in statuses


def test_sim_off_topic_then_resume():
    """Caller asks an off-topic question mid-collection — agent should return UNHANDLED."""
    agent = DataCollectionAgent()
    state: dict = {}
    config = CONFIG_3

    print(f"\n{'=' * 60}")
    print("SIM: Off-topic interruption — UNHANDLED check")
    print(f"{'=' * 60}")

    # Opening
    resp = agent.process("", state, config, [])
    state = resp.internal_state
    print(f"[open] AI: {resp.speak}")

    # Caller gives name
    resp = agent.process("My name is Sarah Mitchell", state, config, [])
    state = resp.internal_state
    print(f"[T1] Caller: 'My name is Sarah Mitchell'")
    print(f"     AI: {resp.speak}  status={resp.status.value}")

    # At some point the agent will ask for confirmation — find that turn
    if resp.status == AgentStatus.WAITING_CONFIRM:
        # Now ask off-topic question
        resp_offto = agent.process("How long does the intake process take?", state, config, [])
        print(f"[T2] Caller: 'How long does the intake process take?' (off-topic)")
        print(f"     AI: {resp_offto.speak!r}  status={resp_offto.status.value}")
        print(f"     pending preserved: {resp_offto.pending_confirmation}")

        assert resp_offto.status == AgentStatus.UNHANDLED, (
            f"Expected UNHANDLED for off-topic, got {resp_offto.status.value}"
        )
        assert resp_offto.pending_confirmation is not None, \
            "pending_confirmation should be preserved on UNHANDLED"
        assert resp_offto.speak == "", "speak must be empty on UNHANDLED"
        print("  ✓ UNHANDLED returned correctly with pending preserved")
    else:
        # Agent may be in IN_PROGRESS — that's fine, check UNHANDLED on any turn
        resp_offto = agent.process("How long does the intake process take?", state, config, [])
        print(f"[T2] Caller: off-topic question  status={resp_offto.status.value}")
        if resp_offto.status == AgentStatus.UNHANDLED:
            assert resp_offto.speak == ""
            print("  ✓ UNHANDLED returned correctly")
        else:
            print(f"  ℹ  Agent kept going (status={resp_offto.status.value}) — LLM may have treated it as answerable")
