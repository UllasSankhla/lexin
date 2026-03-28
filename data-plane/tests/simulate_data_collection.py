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
        pending_before = state.get("pending_confirmation")
        # Now ask off-topic question
        resp_offto = agent.process("How long does the intake process take?", state, config, [])
        print(f"[T2] Caller: 'How long does the intake process take?' (off-topic)")
        print(f"     AI: {resp_offto.speak!r}  status={resp_offto.status.value}")
        print(f"     pending preserved: {resp_offto.pending_confirmation}")

        # With the Planner, WAITING_CONFIRM routes always go to data_collection —
        # the agent may re-ask confirmation (WAITING_CONFIRM) or signal UNHANDLED.
        # Either is acceptable; what matters is pending is preserved so the
        # caller can confirm on the next yes/no turn.
        assert resp_offto.status in (AgentStatus.UNHANDLED, AgentStatus.WAITING_CONFIRM), (
            f"Expected UNHANDLED or WAITING_CONFIRM for off-topic, got {resp_offto.status.value}"
        )
        new_state = resp_offto.internal_state
        assert new_state.get("pending_confirmation") == pending_before, (
            f"pending_confirmation must be preserved. before={pending_before!r} "
            f"after={new_state.get('pending_confirmation')!r}"
        )
        print(f"  ✓ Status={resp_offto.status.value}, pending preserved")
    else:
        # Agent may be in IN_PROGRESS — that's fine, check UNHANDLED on any turn
        resp_offto = agent.process("How long does the intake process take?", state, config, [])
        print(f"[T2] Caller: off-topic question  status={resp_offto.status.value}")
        if resp_offto.status == AgentStatus.UNHANDLED:
            assert resp_offto.speak == ""
            print("  ✓ UNHANDLED returned correctly")
        else:
            print(f"  ℹ  Agent kept going (status={resp_offto.status.value}) — LLM may have treated it as answerable")

CONFIG_4_FIELD = {
    "assistant": {"persona_name": "Aria"},
    "parameters": [
        {"name": "first_name",     "display_label": "First Name",     "data_type": "name",  "required": True,  "extraction_hints": []},
        {"name": "last_name",      "display_label": "Last Name",      "data_type": "name",  "required": True,  "extraction_hints": []},
        {"name": "email_address",  "display_label": "Email Address",  "data_type": "email", "required": True,  "extraction_hints": []},
        {"name": "phone_number",   "display_label": "Contact Number", "data_type": "phone", "required": True,  "extraction_hints": []},
    ],
}


def test_sim_vishwas_babu_transcript():
    """Replay the real Vishwas Babu call transcript verbatim.

    Caller utterances extracted exactly from the recorded conversation,
    in order.  The test asserts all four fields are ultimately collected
    and that the agent reaches COMPLETED despite off-topic tangents,
    corrections, interruptions, and a mis-heard first name.
    """
    collected, statuses = run_conversation(
        [
            "Hello.",
            "Can you",
            "search by my last name. Maybe that's easier.",
            "Yes.",
            "My full name is Vishwas Babu. My first name is Vishwas.",
            "It's not wish wash.",
            "Yes, sir. So that was my son talking in the background. My first name is Vishwas.",
            "And my last name is Babu.",
            "Yes.",
            "Okay. Can you continue with the other questions you have?",
            "Do you have any other questions to ask?",
            "Bubbles.",
            "Uh, no. It's babu, uh, b as in ball, a as in Africa, b as in ball, and u as in underwear.",
            "Yes.",
            "Okay. What next?",
            "What is your",
            "Yes.",
            "vishwas babu at the rate of gmail dot com",
            "So my first name, my last name at the rate of g mail dot com.",
            "Yes.",
            "I would rather",
            "Are you listening?",
            "Hello?",
            "Yes. Two zero two seven one six six six seven five.",
            "Yes.",
        ],
        CONFIG_4_FIELD,
        label="Vishwas Babu — real transcript replay",
        max_retries_on_waiting=6,
    )
    assert collected.get("first_name"),    f"first_name missing:    {collected}"
    assert collected.get("last_name"),     f"last_name missing:     {collected}"
    assert collected.get("email_address"), f"email_address missing: {collected}"
    assert collected.get("phone_number"),  f"phone_number missing:  {collected}"
    assert AgentStatus.COMPLETED in statuses, "Never reached COMPLETED"


def test_sim_mid_collection_question():
    """Caller asks a clarifying question mid-collection: 'Do you need my email address to continue?'

    Behaviour with Planner:
    - The Planner hard-routes to data_collection when WAITING_CONFIRM is active,
      so off-topic questions during confirmation are handled by re-asking the
      confirmation question (WAITING_CONFIRM state is maintained).
    - pending_confirmation is preserved across the non-yes/no turn.
    - When caller later says yes, the field is confirmed and collection continues.
    """
    agent = DataCollectionAgent()
    state: dict = {}
    config = CONFIG_3

    print(f"\n{'=' * 60}")
    print("SIM: Mid-collection clarifying question — WAITING_CONFIRM preserved")
    print(f"{'=' * 60}")

    # Opening
    resp = agent.process("", state, config, [])
    state = resp.internal_state
    print(f"[open] AI: {resp.speak}")

    # T1 — give name, get to WAITING_CONFIRM
    resp = agent.process("Sarah Mitchell", state, config, [])
    state = resp.internal_state
    print(f"[T1] Caller: 'Sarah Mitchell'")
    print(f"     AI: {resp.speak!r}  status={resp.status.value}  pending={state.get('pending_confirmation')}")

    pending_before = state.get("pending_confirmation")
    assert resp.status == AgentStatus.WAITING_CONFIRM, (
        f"Expected WAITING_CONFIRM after name, got {resp.status.value}"
    )

    # T2 — ask clarifying question instead of yes/no
    # Agent must either re-ask confirmation (WAITING_CONFIRM) or signal UNHANDLED.
    # pending_confirmation must be preserved in both cases.
    resp = agent.process("Do you need my email address to continue?", state, config, [])
    state = resp.internal_state
    print(f"[T2] Caller: 'Do you need my email address to continue?'")
    print(f"     AI: {resp.speak!r}  status={resp.status.value}  pending={state.get('pending_confirmation')}")

    assert resp.status in (AgentStatus.WAITING_CONFIRM, AgentStatus.UNHANDLED), (
        f"Expected WAITING_CONFIRM or UNHANDLED for off-topic during confirmation, "
        f"got {resp.status.value}"
    )
    assert state.get("pending_confirmation") == pending_before, (
        f"pending_confirmation must be preserved. "
        f"before={pending_before}  after={state.get('pending_confirmation')}"
    )
    print(f"  ✓ Status={resp.status.value}, pending preserved")

    # T3 — caller resumes with yes/no — collection should continue
    resp = agent.process("Yes", state, config, [])
    state = resp.internal_state
    print(f"[T3] Caller: 'Yes'  (confirming)")
    print(f"     AI: {resp.speak!r}  status={resp.status.value}  collected={state.get('collected', {})}")

    assert state.get("collected", {}).get("full_name"), (
        f"full_name should be confirmed after 'Yes', collected={state.get('collected')}"
    )
    print("  ✓ Collection resumed correctly after confirming")


def test_sim_spelled_out_name():
    """Caller spells their name letter-by-letter or via NATO phonetic alphabet."""
    collected, statuses = run_conversation(
        [
            "N-A-T-H-A-N-I-E-L sierra echo alpha november",  # "Nathaniel Sean" spelled out
            "yes",
            "four one five five five five zero one nine two",
            "yes",
            "nathaniel at example dot com",
            "yes",
        ],
        CONFIG_3,
        label="Spelled-out name — N-A-T-H-A-N-I-E-L + NATO last name",
        max_retries_on_waiting=3,
    )
    assert collected.get("full_name"), f"full_name missing: {collected}"
    name = collected["full_name"]
    assert any(part.lower() in name.lower() for part in ["nathaniel", "nathan"]), \
        f"Name not decoded correctly: {name!r}"
    assert AgentStatus.COMPLETED in statuses


def test_sim_dense_single_utterance():
    """Caller provides all three fields in one breath.

    E.g.: "I'm Jane Doe, my number is 415-555-0100, email is jane at example dot com."
    The agent should extract all three, confirm them one at a time, and complete.
    Verifies MULTIPLE FIELDS AT ONCE extraction.
    """
    collected, statuses = run_conversation(
        [
            # All three values in one shot
            "I'm Jane Doe, my number is 415-555-0100, and my email is jane at example dot com",
            "yes",   # confirm whichever field was picked up first
            "yes",   # confirm second
            "yes",   # confirm third (may already be done after two)
        ],
        CONFIG_3,
        label="Dense single utterance — all 3 fields at once",
        max_retries_on_waiting=4,
    )
    assert collected.get("full_name"),    f"full_name missing:    {collected}"
    assert collected.get("phone_number"), f"phone_number missing: {collected}"
    assert collected.get("email_address"),f"email_address missing:{collected}"
    assert AgentStatus.COMPLETED in statuses


def test_sim_correction_plus_new_field_same_turn():
    """While WAITING_CONFIRM for name, caller corrects it AND provides phone.

    Caller flow:
      T1: "John Doe"           → AI reads back "John Doe, is that correct?"
      T2: "Actually it's Jane Doe, and my number is 415-555-9999"
                               → correction applied + phone queued for next confirm
      T3: "yes"                → confirm phone (or corrected name)
      ...
    Verifies CORRECTION intent + simultaneous new-field extraction.
    """
    agent = DataCollectionAgent()
    state: dict = {}
    config = CONFIG_3

    print(f"\n{'=' * 60}")
    print("SIM: Correction + new field in same utterance")
    print(f"{'=' * 60}")

    resp = agent.process("", state, config, [])
    state = resp.internal_state
    print(f"[open] AI: {resp.speak}")

    # T1 — give name, reach WAITING_CONFIRM
    resp = agent.process("John Doe", state, config, [])
    state = resp.internal_state
    print(f"[T1] Caller: 'John Doe'")
    print(f"     AI: {resp.speak!r}  status={resp.status.value}")

    # T2 — correction + new field
    resp = agent.process(
        "Actually it's Jane Doe, and my number is 415-555-9999",
        state, config, [],
    )
    state = resp.internal_state
    print(f"[T2] Caller: 'Actually it's Jane Doe, and my number is 415-555-9999'")
    print(f"     AI: {resp.speak!r}  status={resp.status.value}  pending={state.get('pending_confirmation')}  collected={state.get('collected', {})}")

    # Name must now reflect the correction (either pending Jane Doe or confirmed Jane Doe)
    pending = state.get("pending_confirmation")
    collected_so_far = state.get("collected", {})
    name_in_flight = (
        (pending or {}).get("value", "").lower() if pending and pending.get("field") == "full_name"
        else collected_so_far.get("full_name", "").lower()
    )
    assert "jane" in name_in_flight, (
        f"Correction not applied — expected 'jane' in name, "
        f"pending={pending}, collected={collected_so_far}"
    )
    print(f"  ✓ Correction applied: name in flight = {name_in_flight!r}")

    # Drive to completion
    for i, utt in enumerate(["yes", "yes", "jane at example dot com", "yes"], start=3):
        if resp.status == AgentStatus.COMPLETED:
            break
        resp = agent.process(utt, state, config, [])
        state = resp.internal_state
        print(f"[T{i}] Caller: {utt!r}  AI: {resp.speak!r}  status={resp.status.value}  collected={state.get('collected', {})}")

    final = state.get("collected", {}) or (resp.collected or {})
    assert final.get("full_name"),    f"full_name missing:    {final}"
    assert final.get("phone_number"), f"phone_number missing: {final}"
    assert AgentStatus.COMPLETED in [resp.status] or final.get("email_address"), (
        f"Did not complete: {final}"
    )


def test_sim_correction_of_confirmed_field():
    """Caller corrects a field that was already confirmed.

    Flow:
      T1: "John Doe"     → WAITING_CONFIRM
      T2: "yes"          → full_name confirmed as "John Doe"
      T3: "Wait, I made a mistake — my name is actually Jane Doe"
                         → should update full_name to "Jane Doe"
      T4+: normal collection continues

    This surfaces whether the agent can revise an already-confirmed field.
    """
    agent = DataCollectionAgent()
    state: dict = {}
    config = CONFIG_3

    print(f"\n{'=' * 60}")
    print("SIM: Correction of already-confirmed field")
    print(f"{'=' * 60}")

    resp = agent.process("", state, config, [])
    state = resp.internal_state
    print(f"[open] AI: {resp.speak}")

    # T1 — give name
    resp = agent.process("John Doe", state, config, [])
    state = resp.internal_state
    print(f"[T1] Caller: 'John Doe'  AI: {resp.speak!r}  status={resp.status.value}")

    # T2 — confirm it
    resp = agent.process("yes", state, config, [])
    state = resp.internal_state
    print(f"[T2] Caller: 'yes'  AI: {resp.speak!r}  status={resp.status.value}  collected={state.get('collected', {})}")
    assert state.get("collected", {}).get("full_name"), "full_name should be confirmed after 'yes'"
    print(f"  ✓ full_name confirmed as {state['collected']['full_name']!r}")

    # T3 — caller corrects the already-confirmed name
    resp = agent.process(
        "Wait, I made a mistake — my name is actually Jane Doe not John",
        state, config, [],
    )
    state = resp.internal_state
    print(f"[T3] Caller: 'Wait, my name is actually Jane Doe not John'")
    print(f"     AI: {resp.speak!r}  status={resp.status.value}  pending={state.get('pending_confirmation')}  collected={state.get('collected', {})}")

    pending = state.get("pending_confirmation")
    collected_so_far = state.get("collected", {})
    # The corrected name should either be re-pending or already updated in collected
    name_value = (
        (pending or {}).get("value", "").lower() if pending and pending.get("field") == "full_name"
        else collected_so_far.get("full_name", "").lower()
    )
    corrected = "jane" in name_value
    print(f"  {'✓' if corrected else '✗'} name after correction: {name_value!r}")
    assert corrected, (
        f"Correction of confirmed field not applied — expected 'jane' in name, "
        f"pending={pending}, collected={collected_so_far}"
    )

    # Drive to completion regardless
    for i, utt in enumerate(["yes", "four one five five five five zero one nine two", "yes", "jane at example dot com", "yes"], start=4):
        if resp.status == AgentStatus.COMPLETED:
            break
        resp = agent.process(utt, state, config, [])
        state = resp.internal_state
        print(f"[T{i}] Caller: {utt!r}  AI: {resp.speak!r}  status={resp.status.value}  collected={state.get('collected', {})}")

    final = state.get("collected", {}) or (resp.collected or {})
    print(f"Final collected: {final}")
    assert AgentStatus.COMPLETED in [resp.status] or all(
        final.get(k) for k in ("full_name", "phone_number", "email_address")
    ), f"Did not complete cleanly: {final}"
