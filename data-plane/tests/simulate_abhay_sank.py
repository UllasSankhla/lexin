"""
Simulation: Abhay Sank transcript — split first/last name fields, off-topic questions,
partial spelling, and optional MetLife ID decline.

Reproduces the exact call sequence:
  Aria: Could I get your First Name, please?
  You:  Yeah. My first name is, uh, Abhay, a b h o a y.
  Aria: I have your first name as Abhay — is that correct?
  You:  Yeah.
  Aria: Great, and could I get your Last Name, please?
  You:  What information do you need to schedule, uh, my call?
  Aria: <lists remaining fields, re-asks last name>
  You:  So, yeah, I can tell you my last name, but what what... can you help me schedule a call
  You:  with the with the lawyer.
  Aria: <lists remaining fields, re-asks last name again>
  You:  My last name is s a n k.
  Aria: I have your last name as Sank — is that correct?
  You:  Yeah.
  Aria: Great, and could I get your Email Address, please?
  You:  It's a b dot s a n k at c dot com.
  Aria: I have your email as ab dot sank at c dot com — is that correct?
  You:  Yeah.
  Aria: <acknowledges + asks for phone number>
  You:  It's two zero six seven seven nine four one four six.
  Aria: I have your phone number as two zero six, seven seven nine, four one four six — is that correct?
  You:  That's correct.
  Aria: Great, and could I get your Metlife Id, please?
  You:  one right now.          ← ambiguous / doesn't have one
  Aria: <asks again>
  You:  I don't have a MetLife ID.
  Aria: Got it, thanks for that — Perfect, I have what I need for now.
                                ← COMPLETED — then AI "fails to proceed further"

Known failure mode to investigate:
  - After the MetLife ID decline the agent says "Perfect, I have what I need for now"
    (COMPLETED) but the call does not proceed to the next stage (scheduling).
  - Also investigate: does the agent correctly skip the optional MetLife ID and
    set status=completed, or does it loop / get stuck?

Run:
    cd data-plane
    PYTHONPATH=. python3 -m pytest tests/simulate_abhay_sank.py -v -s 2>&1 | tee /tmp/abhay_sank_run.log
"""
from __future__ import annotations

import logging
import pytest

from app.agents.data_collection import DataCollectionAgent
from app.agents.base import AgentStatus

# ── Enable DEBUG-level logs so we can inspect internal agent decisions ─────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ── Config matching the Aria call (split first/last name, not full_name) ──────

# ── Minimal config (4 req + 1 opt) — does NOT reproduce the Step D bug ──────
CONFIG_MINIMAL = {
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

# ── Production-like config (4 req + 2 opt) — reproduces the Step D bug ──────
# The second optional field (referral_source) is never addressed because the
# LLM correctly completes after required fields are done — but the Python
# Step D check (not remaining_all) would wrongly prevent COMPLETED.
CONFIG = {
    **CONFIG_MINIMAL,
    "parameters": CONFIG_MINIMAL["parameters"] + [
        {
            "name": "referral_source",
            "display_label": "How Did You Hear About Us",
            "data_type": "text",
            "required": False,
            "collection_order": 6,
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
            f"collected={state.get('collected', {})}"
        )

        if resp.status == AgentStatus.COMPLETED:
            logger.info(
                "Agent reached COMPLETED at turn T%02d. "
                "Final collected: %s",
                i + 1, state.get("collected"),
            )
            break

    final = state.get("collected", {}) or (resp.collected or {})
    print(f"\nFinal collected: {final}")
    print(f"Final status: {resp.status.value}")
    return final, statuses, state


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_abhay_sank_full_transcript():
    """
    Reproduce the exact Abhay Sank call sequence.

    Assertions:
      1. first_name = "Abhay"
      2. last_name  = "Sank"
      3. email_address confirmed (ab.sank@c.com or equivalent)
      4. phone_number confirmed (2067794146)
      5. metlife_id skipped (optional, caller declined)
      6. Agent reaches COMPLETED status
      7. Agent does NOT loop indefinitely on metlife_id after caller declines
    """
    utterances = [
        # T01: Provides first name with partial wrong spelling "a b h o a y"
        "Yeah. My first name is, uh, Abhay, a b h o a y.",
        # T02: Confirms first name
        "Yeah.",
        # T03: Off-topic — asks what info is needed
        "What information do you need to schedule, uh, my call?",
        # T04: Mixes willingness to give last name with off-topic scheduling question
        "So, yeah, I can tell you my last name, but what what... can you help me schedule a call",
        # T05: Continues previous sentence (scheduling question)
        "with the with the lawyer.",
        # T06: Provides last name via letter spelling "s a n k"
        "My last name is s a n k.",
        # T07: Confirms last name
        "Yeah.",
        # T08: Provides email via spoken form
        "It's a b dot s a n k at c dot com.",
        # T09: Confirms email
        "Yeah.",
        # T10: Provides phone number
        "It's two zero six seven seven nine four one four six.",
        # T11: Confirms phone number
        "That's correct.",
        # T12: Ambiguous / doesn't actually give a MetLife ID
        "one right now.",
        # T13: Explicitly declines MetLife ID
        "I don't have a MetLife ID.",
    ]

    final, statuses, state = _run(
        utterances,
        label="Abhay Sank — split names, off-topic questions, optional MetLife ID decline",
        max_retries_on_waiting=4,
    )

    # ── Field assertions ──────────────────────────────────────────────────────

    first = final.get("first_name", "")
    assert first, f"first_name was not collected: {final}"
    assert "abhay" in first.lower(), f"[BUG] Expected 'Abhay' in first_name, got: {first!r}"

    last = final.get("last_name", "")
    assert last, f"last_name was not collected: {final}"
    assert "sank" in last.lower(), f"[BUG] Expected 'Sank' in last_name, got: {last!r}"

    email = final.get("email_address", "")
    assert email, f"email_address was not collected: {final}"
    assert "@" in email or "at" in email.lower(), (
        f"[BUG] email_address doesn't look like an email: {email!r}"
    )

    phone = final.get("phone_number", "")
    assert phone, f"phone_number was not collected: {final}"
    digits = "".join(c for c in phone if c.isdigit())
    assert len(digits) >= 10, (
        f"[BUG] phone_number has fewer than 10 digits: {phone!r} (digits={digits!r})"
    )
    assert "2067794146" in digits or digits.endswith("7794146"), (
        f"[BUG] phone_number digits don't match expected 2067794146: {digits!r}"
    )

    # MetLife ID is optional — either absent or empty string are both acceptable
    metlife = final.get("metlife_id", None)
    assert metlife is None or metlife == "", (
        f"[BUG] metlife_id should be skipped (None or ''), but got: {metlife!r}"
    )

    assert AgentStatus.COMPLETED in statuses, (
        f"[BUG] Agent never reached COMPLETED. Final status sequence: "
        + str([s.value for s in statuses])
    )

    print(f"\n  ✓ first_name={first!r}")
    print(f"  ✓ last_name={last!r}")
    print(f"  ✓ email_address={email!r}")
    print(f"  ✓ phone_number={phone!r}")
    print(f"  ✓ metlife_id skipped (value={metlife!r})")


def test_step_d_completes_when_required_done_optional_remain():
    """
    Regression: Step D completion check must trigger on required_remaining=[]
    even when uncollected optional fields are still present in the config.

    Root cause (observed in production log):
      DataCollection: optional field 'metlife_id' skipped by caller
      Agent response | status=in_progress | edge=in_progress→decider

    The LLM correctly speaks 'Perfect, I have what I need for now.' but the Python
    Step D check used `not remaining_all` (ALL params) instead of `not required_remaining`
    (REQUIRED params only).  With a second optional field (referral_source) still in
    remaining_all, the condition was False → status=in_progress → edge→decider →
    narrative_collection never starts.

    This test uses CONFIG (4 required + 2 optional) to reproduce the condition and
    verify the fix (Step D now uses required_remaining).
    """
    agent = DataCollectionAgent()
    state: dict = {}

    print(f"\n{'=' * 70}")
    print("SIM: Step D — required-done + 2 optional remain → must be COMPLETED")
    print(f"{'=' * 70}")

    resp = agent.process("", state, CONFIG, [])
    state = resp.internal_state
    print(f"[open] Aria: {resp.speak}")

    required_turns = [
        ("My first name is Abhay.", "first_name pending"),
        ("Yeah.", "first_name confirmed"),
        ("My last name is Sank.", "last_name pending"),
        ("Yeah.", "last_name confirmed"),
        ("It's ab dot sank at c dot com.", "email pending"),
        ("Yeah.", "email confirmed"),
        ("Two zero six seven seven nine four one four six.", "phone pending"),
        ("That's correct.", "phone confirmed → asks MetLife ID"),
    ]

    for utt, note in required_turns:
        resp = agent.process(utt, state, CONFIG, [])
        state = resp.internal_state
        print(
            f"  [{note}] Caller: {utt!r}\n"
            f"    Aria: {resp.speak!r}\n"
            f"    status={resp.status.value}  collected={state.get('collected', {})}\n"
        )
        if resp.status == AgentStatus.COMPLETED:
            break

    if resp.status == AgentStatus.COMPLETED:
        # With the fix (required_remaining check), the agent correctly completes here
        # without ever asking for metlife_id or referral_source.  This is the fix working.
        print("  (Completed after required fields done — optional fields correctly skipped)")
        return

    # Agent should now be asking for MetLife ID (or referral_source)
    # Caller declines MetLife ID.  referral_source remains in remaining_all.
    resp = agent.process("I don't have a MetLife ID.", state, CONFIG, [])
    state = resp.internal_state
    print(
        f"  [decline metlife] Caller: 'I don't have a MetLife ID.'\n"
        f"    Aria: {resp.speak!r}\n"
        f"    status={resp.status.value}\n"
        f"    collected={state.get('collected', {})}\n"
        f"    pending={state.get('pending_confirmation')}\n"
    )

    assert resp.status == AgentStatus.COMPLETED, (
        f"[BUG — Step D] After declining MetLife ID (optional), with referral_source "
        f"still uncollected, the agent returned status={resp.status.value!r} instead "
        f"of COMPLETED.\n"
        f"Root cause: Step D used 'not remaining_all' (includes optional fields) "
        f"instead of 'not required_remaining'.\n"
        f"Fix: change Step D condition to 'if not required_remaining and not new_pending'.\n"
        f"speak={resp.speak!r}"
    )
    print("  ✓ COMPLETED even though referral_source was never collected")


def test_metlife_id_skipped_triggers_completed():
    """
    Isolated regression: when the caller explicitly declines the optional MetLife ID
    after all required fields are confirmed, the agent must immediately set
    status=completed — not loop or ask again.

    This directly tests the 'fails to proceed further' failure: if the agent stays
    in_progress or waiting_confirm after the decline, downstream call routing never
    triggers the scheduling stage.
    """
    agent = DataCollectionAgent()
    state: dict = {}

    print(f"\n{'=' * 70}")
    print("SIM: MetLife ID decline → COMPLETED (isolated)")
    print(f"{'=' * 70}")

    # Opening
    resp = agent.process("", state, CONFIG_MINIMAL, [])
    state = resp.internal_state
    print(f"[open] Aria: {resp.speak}")

    # Drive through all required fields quickly
    required_turns = [
        # first_name
        ("My first name is Abhay.", "Abhay confirmed"),
        ("Yeah.", "first_name confirmed"),
        # last_name
        ("My last name is Sank.", "Sank pending"),
        ("Yeah.", "last_name confirmed"),
        # email
        ("It's ab dot sank at c dot com.", "email pending"),
        ("Yeah.", "email confirmed"),
        # phone
        ("Two zero six seven seven nine four one four six.", "phone pending"),
        ("That's correct.", "phone confirmed"),
    ]

    for utt, note in required_turns:
        resp = agent.process(utt, state, CONFIG_MINIMAL, [])
        state = resp.internal_state
        print(
            f"  [{note}] Caller: {utt!r}\n"
            f"    Aria: {resp.speak!r}\n"
            f"    status={resp.status.value}  collected={state.get('collected', {})}\n"
        )
        if resp.status == AgentStatus.COMPLETED:
            break

    if resp.status == AgentStatus.COMPLETED:
        # Completed before even reaching MetLife ID — that's fine if all required
        # fields were already gathered
        print("  (Completed before MetLife ID question — all required fields done)")
        return

    # Now the agent should be asking for MetLife ID (optional)
    assert resp.status in (AgentStatus.IN_PROGRESS, AgentStatus.WAITING_CONFIRM), (
        f"Unexpected status before MetLife step: {resp.status.value}"
    )

    # T-decline: caller explicitly says they don't have one
    resp = agent.process("I don't have a MetLife ID.", state, CONFIG_MINIMAL, [])
    state = resp.internal_state
    print(
        f"  [decline metlife] Caller: 'I don't have a MetLife ID.'\n"
        f"    Aria: {resp.speak!r}\n"
        f"    status={resp.status.value}  collected={state.get('collected', {})}\n"
    )

    assert resp.status == AgentStatus.COMPLETED, (
        f"[BUG] After declining optional MetLife ID the agent is stuck at "
        f"status={resp.status.value!r} instead of COMPLETED. "
        f"This is why the AI 'fails to proceed further' — COMPLETED is never signalled "
        f"to the call router so the scheduling stage never starts.\n"
        f"speak={resp.speak!r}\n"
        f"pending={state.get('pending_confirmation')}\n"
        f"collected={state.get('collected')}"
    )

    print("  ✓ COMPLETED reached after MetLife ID decline")
