"""
Simulation: Kavita Sharma transcript — NATO phonetic name spelling + referral field.

Reproduces a real call where:
  1. Caller opens with their name AND the referral source in one breath.
  2. AI reads back the name and enters WAITING_CONFIRM.
  3. Caller spells out their full name via NATO phonetic alphabet
     ("K as in kilo, A as in alpha, V as in victor, ...").
  4. AI must decode the phonetic spelling to "Kavita Sharma" and confirm it.
  5. AI then moves to the referral field ("How Did You Hear About Us") and
     correctly extracts "MetLife Legal Plans website" from the opening utterance.
  6. The name must still be present in collected at the end — not dropped.

Known failure mode observed in production:
  - After the NATO spelling turn, the AI skipped confirming the decoded name
    and jumped straight to confirming the referral source.
  - Later in the same call, the AI forgot the caller's name entirely.

Run:
    cd data-plane
    PYTHONPATH=. python3 -m pytest tests/simulate_kavita_sharma.py -v -s
"""
from __future__ import annotations

import pytest
from app.agents.data_collection import DataCollectionAgent
from app.agents.base import AgentStatus


# ── Config ────────────────────────────────────────────────────────────────────

CONFIG = {
    "assistant": {"persona_name": "Aria at Sterling Immigration Law"},
    "parameters": [
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
            "extraction_hints": [],
        },
        {
            "name": "email_address",
            "display_label": "Email Address",
            "data_type": "email",
            "required": True,
            "extraction_hints": [],
        },
        {
            "name": "referral_source",
            "display_label": "How Did You Hear About Us",
            "data_type": "text",
            "required": False,
            "extraction_hints": ["MetLife Legal Plans", "referral source"],
        },
    ],
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _run(utterances: list[str], *, label: str, max_retries_on_waiting: int = 4):
    """Drive the DataCollectionAgent through scripted utterances and return state."""
    agent = DataCollectionAgent()
    state: dict = {}
    statuses: list[AgentStatus] = []

    print(f"\n{'=' * 70}")
    print(f"SIM: {label}")
    print(f"{'=' * 70}")

    resp = agent.process("", state, CONFIG, [])
    statuses.append(resp.status)
    state = resp.internal_state
    print(f"[open] AI: {resp.speak}")

    waiting_streak = 0
    prev_pending_field: str | None = None

    for i, utt in enumerate(utterances):
        print(f"\n[T{i+1}] Caller: {utt!r}")
        resp = agent.process(utt, state, CONFIG, [])
        statuses.append(resp.status)
        state = resp.internal_state

        cur_pending = state.get("pending_confirmation")
        cur_field = cur_pending["field"] if cur_pending else None

        if resp.status == AgentStatus.WAITING_CONFIRM and cur_field == prev_pending_field:
            waiting_streak += 1
            assert waiting_streak <= max_retries_on_waiting, (
                f"Stuck in WAITING_CONFIRM for {cur_field!r} after {waiting_streak} turns"
            )
        else:
            waiting_streak = 0
        prev_pending_field = cur_field

        print(f"       AI: {resp.speak}")
        print(f"       status={resp.status.value}  pending={cur_pending}  collected={state.get('collected', {})}")

        if resp.status == AgentStatus.COMPLETED:
            break

    final = state.get("collected", {}) or (resp.collected or {})
    print(f"\nFinal collected: {final}")
    return final, statuses, state


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_nato_phonetic_name_decoded():
    """
    Caller opens with name + referral, then spells the name via NATO phonetic.

    Core assertions:
      1. The name is decoded as "Kavita Sharma" (not garbled).
      2. The referral source "MetLife Legal Plans website" is captured.
      3. full_name is still present at the end — not forgotten.
    """
    final, statuses, state = _run(
        [
            # T1: Opening — name + referral source + reason for call
            (
                "Hi yes, good afternoon. My name is Kavita Sharma. "
                "I found your number through the MetLife Legal Plans website "
                "and I'm looking for help with some H1B related questions."
            ),
            # T2: Caller spells out name via NATO phonetic alphabet
            (
                "Sure. First name is Kavita. K as in kilo, A as in alpha, "
                "V as in victor, I as in india, T as in tango, A as in alpha. "
                "Last name Sharma. S as in sierra, H as in hotel, A as in alpha, "
                "R as in romeo, M as in mike, A as in alpha."
            ),
            # T3: Confirm name (yes/no readback)
            "Yes, that's correct.",
            # T4: Confirm referral source if asked
            "Yes, MetLife Legal Plans website.",
            # T5-T8: remaining required fields
            "six five zero five five five one two three four",
            "yes",
            "kavita dot sharma at gmail dot com",
            "yes",
        ],
        label="Kavita Sharma — NATO phonetic name + MetLife referral",
        max_retries_on_waiting=4,
    )

    # --- Name assertions ---
    name = final.get("full_name", "")
    assert name, f"full_name was not collected at all: {final}"
    assert "kavita" in name.lower(), (
        f"[BUG] First name 'Kavita' not in collected name: {name!r}"
    )
    assert "sharma" in name.lower(), (
        f"[BUG] Last name 'Sharma' not in collected name: {name!r}"
    )

    # --- Referral assertions ---
    referral = final.get("referral_source", "")
    assert referral, f"referral_source was not collected: {final}"
    assert "metlife" in referral.lower(), (
        f"[BUG] 'MetLife' not in referral_source: {referral!r}"
    )

    assert AgentStatus.COMPLETED in statuses, "Never reached COMPLETED"
    print(f"\n  ✓ full_name={name!r}")
    print(f"  ✓ referral_source={referral!r}")


def test_name_not_forgotten_after_referral_field():
    """
    After the referral field is confirmed, full_name must still be in collected.

    Regression guard: the production call showed the AI 'forgetting' the caller's
    name after moving past the referral confirmation step.
    """
    agent = DataCollectionAgent()
    state: dict = {}

    print(f"\n{'=' * 70}")
    print("SIM: Name retention — full_name must survive all subsequent turns")
    print(f"{'=' * 70}")

    # Opening
    resp = agent.process("", state, CONFIG, [])
    state = resp.internal_state
    print(f"[open] AI: {resp.speak}")

    # T1: Provide name + referral in one utterance
    resp = agent.process(
        "My name is Kavita Sharma. I found your number through MetLife Legal Plans.",
        state, CONFIG, [],
    )
    state = resp.internal_state
    print(f"[T1] AI: {resp.speak!r}  status={resp.status.value}")

    # Confirm name
    resp = agent.process("Yes", state, CONFIG, [])
    state = resp.internal_state
    print(f"[T2] AI: {resp.speak!r}  status={resp.status.value}  collected={state.get('collected', {})}")

    name_after_confirm = state.get("collected", {}).get("full_name", "")
    assert name_after_confirm, (
        f"[BUG] full_name not in collected after confirmation: {state.get('collected')}"
    )
    print(f"  ✓ full_name confirmed: {name_after_confirm!r}")

    # Drive through referral confirmation and at least one more field
    for utt in [
        "Yes, MetLife Legal Plans website.",  # confirm referral
        "six five zero five five five one two three four",   # phone
        "yes",
        "kavita at example dot com",          # email
        "yes",
    ]:
        if resp.status == AgentStatus.COMPLETED:
            break
        resp = agent.process(utt, state, CONFIG, [])
        state = resp.internal_state
        print(f"  Caller: {utt!r}  AI: {resp.speak!r}  status={resp.status.value}")

        # Assert name has not been dropped from collected at any turn
        cur_collected = state.get("collected", {})
        if cur_collected:
            assert cur_collected.get("full_name"), (
                f"[BUG] full_name was dropped from collected mid-conversation: {cur_collected}"
            )

    final = state.get("collected", {}) or (resp.collected or {})
    assert final.get("full_name"), f"[BUG] full_name missing from final collected: {final}"
    assert "kavita" in final["full_name"].lower(), (
        f"[BUG] final full_name does not contain 'Kavita': {final['full_name']!r}"
    )
    print(f"\n  ✓ full_name preserved throughout: {final['full_name']!r}")


def test_nato_spelling_readback_is_correct_name():
    """
    After the caller spells out 'Kavita Sharma' via NATO phonetic, the AI's
    readback must contain the decoded name, not the raw phonetic letters.

    Catches the specific production failure: AI read back "Just to confirm —
    your How Did You Hear About Us is MetLife Legal Plans website. Is that
    correct?" — skipping the name readback entirely after the phonetic spelling.
    """
    agent = DataCollectionAgent()
    state: dict = {}

    print(f"\n{'=' * 70}")
    print("SIM: Readback after NATO spelling must decode to 'Kavita Sharma'")
    print(f"{'=' * 70}")

    resp = agent.process("", state, CONFIG, [])
    state = resp.internal_state
    print(f"[open] AI: {resp.speak}")

    # T1: Opening with name
    resp = agent.process(
        "Hi, my name is Kavita Sharma. I found your number through MetLife Legal Plans.",
        state, CONFIG, [],
    )
    state = resp.internal_state
    print(f"[T1] AI: {resp.speak!r}  status={resp.status.value}")

    # T2: Caller spells out the name (as if the AI may have mis-heard)
    resp = agent.process(
        "K as in kilo, A as in alpha, V as in victor, I as in india, T as in tango, "
        "A as in alpha. Last name: S as in sierra, H as in hotel, A as in alpha, "
        "R as in romeo, M as in mike, A as in alpha.",
        state, CONFIG, [],
    )
    state = resp.internal_state
    print(f"[T2] AI: {resp.speak!r}  status={resp.status.value}  pending={state.get('pending_confirmation')}")

    # After this turn the agent must be in WAITING_CONFIRM for full_name
    # (not for referral_source, which would indicate it skipped the name)
    pending = state.get("pending_confirmation")
    if pending:
        assert pending.get("field") == "full_name", (
            f"[BUG] After NATO spelling the agent is confirming {pending.get('field')!r} "
            f"instead of full_name. This matches the production bug where the agent "
            f"jumped to confirming the referral source."
        )
        decoded_name = pending.get("value", "")
        assert "kavita" in decoded_name.lower() and "sharma" in decoded_name.lower(), (
            f"[BUG] Pending value after NATO spelling not decoded correctly: {decoded_name!r}"
        )
        print(f"  ✓ pending full_name={decoded_name!r}")
    else:
        # May have already confirmed — check collected
        collected_name = state.get("collected", {}).get("full_name", "")
        assert "kavita" in collected_name.lower() and "sharma" in collected_name.lower(), (
            f"[BUG] Name after NATO spelling not in collected: {collected_name!r}, "
            f"collected={state.get('collected')}"
        )
        print(f"  ✓ collected full_name={collected_name!r}")

    # The AI's spoken text must NOT be jumping to the referral field
    assert "how did you hear" not in resp.speak.lower(), (
        f"[BUG] AI skipped name confirmation and jumped to referral: {resp.speak!r}"
    )
    assert "metlife" not in resp.speak.lower() or "kavita" in resp.speak.lower(), (
        f"[BUG] AI is confirming referral source before confirming the name: {resp.speak!r}"
    )
