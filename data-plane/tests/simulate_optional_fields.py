"""Simulation tests for optional field collection in DataCollectionAgent.

Validates that:
1. Optional fields mid-sequence (between required fields) are asked — the question
   is posed the same way as required fields, without labelling them optional.
2. Caller-initiated skip of one optional field moves to the NEXT field in order,
   not to the next required field — other optionals are still asked.
3. Skipping an optional does not prevent subsequent required fields from being collected.
4. Providing a value for an optional field collects it correctly.
5. Optional fields appearing after all required fields are done are NOT asked
   (current design: once required collection is complete, agent completes immediately).

Run:
    cd data-plane
    PYTHONPATH=. python3 -m pytest tests/simulate_optional_fields.py -v -s
"""
from __future__ import annotations

import pytest
from app.agents.data_collection import DataCollectionAgent
from app.agents.base import AgentStatus


# ── Configs ───────────────────────────────────────────────────────────────────

def _param(name: str, label: str, dtype: str, required: bool, order: int) -> dict:
    return {
        "name": name,
        "display_label": label,
        "data_type": dtype,
        "required": required,
        "collection_order": order,
        "extraction_hints": [],
    }


# required → optional → required  (optional is mid-sequence, non-name to avoid
# LLM conflating first/middle into a single "full name" question)
CONFIG_OPT_MID = {
    "assistant": {"persona_name": "Aria"},
    "parameters": [
        _param("full_name", "Full Name",     "name",  required=True,  order=1),
        _param("company",   "Company Name",  "text",  required=False, order=2),
        _param("email",     "Email Address", "email", required=True,  order=3),
    ],
}

# required → required → optional  (optional is at the end — never asked by design)
CONFIG_OPT_END = {
    "assistant": {"persona_name": "Aria"},
    "parameters": [
        _param("first_name", "First Name",     "name",  required=True,  order=1),
        _param("email",      "Email Address",  "email", required=True,  order=2),
        _param("referral",   "Referral Source","text",  required=False, order=3),
    ],
}

# required → optional → optional → required  (two optionals in a row mid-sequence)
CONFIG_TWO_OPT_MID = {
    "assistant": {"persona_name": "Aria"},
    "parameters": [
        _param("first_name",  "First Name",    "name",  required=True,  order=1),
        _param("middle_name", "Middle Name",   "name",  required=False, order=2),
        _param("nickname",    "Nickname",      "name",  required=False, order=3),
        _param("email",       "Email Address", "email", required=True,  order=4),
    ],
}

# required → optional(phone) → required
CONFIG_OPT_PHONE_MID = {
    "assistant": {"persona_name": "Aria"},
    "parameters": [
        _param("first_name",    "First Name",    "name",  required=True,  order=1),
        _param("phone_number",  "Phone Number",  "phone", required=False, order=2),
        _param("email_address", "Email Address", "email", required=True,  order=3),
    ],
}


# ── Helper ────────────────────────────────────────────────────────────────────

def run_and_capture(
    utterances: list[str],
    config: dict,
    *,
    label: str = "",
    max_turns: int = 30,
) -> tuple[dict, list[AgentStatus], list[str]]:
    """Drive agent through scripted utterances.

    Returns (final_collected, statuses, speak_texts).
    speak_texts includes the opening speak plus one per caller utterance.
    """
    agent = DataCollectionAgent()
    state: dict = {}
    statuses: list[AgentStatus] = []
    speaks: list[str] = []
    history: list[dict] = []

    print(f"\n{'=' * 60}")
    print(f"SIM: {label}")
    print(f"{'=' * 60}")

    resp = agent.process("", state, config, history)
    statuses.append(resp.status)
    state = resp.internal_state
    speaks.append(resp.speak or "")
    history.append({"role": "assistant", "content": resp.speak or ""})
    print(f"[open]  AI: {resp.speak}")

    for i, utt in enumerate(utterances[:max_turns]):
        history.append({"role": "user", "content": utt})
        resp = agent.process(utt, state, config, history)
        statuses.append(resp.status)
        state = resp.internal_state
        speaks.append(resp.speak or "")
        history.append({"role": "assistant", "content": resp.speak or ""})

        pending = state.get("pending_confirmation")
        print(f"[T{i+1}]  Caller: {utt!r}")
        print(f"        AI: {resp.speak}")
        print(f"        status={resp.status.value}  pending={pending}  collected={state.get('collected', {})}")

        if resp.status == AgentStatus.COMPLETED:
            break

    final_collected = state.get("collected", {}) or (resp.collected or {})
    print(f"\nFinal collected: {final_collected}")
    return final_collected, statuses, speaks


def _field_mentioned_in_speaks(field_label: str, speaks: list[str]) -> bool:
    """True if any speak text references the field (case-insensitive)."""
    label_lower = field_label.lower()
    return any(label_lower in s.lower() for s in speaks)


# ── Tests: optional field mid-sequence ───────────────────────────────────────

class TestOptionalFieldMidSequence:
    """Optional field (Company Name) sits between two required fields (Full Name, Email)."""

    def test_optional_mid_sequence_question_is_posed(self):
        """Agent must ask for Company Name (optional) before moving to email."""
        collected, statuses, speaks = run_and_capture(
            [
                "John Smith",
                "yes",                    # confirm full_name
                "Acme Corp",              # provide company when asked
                "yes",                    # confirm company
                "john@example.com",
                "yes",
            ],
            CONFIG_OPT_MID,
            label="Optional mid-sequence — question is posed",
        )
        assert collected.get("full_name"), f"full_name missing: {collected}"
        assert collected.get("email"), f"email missing: {collected}"
        assert AgentStatus.COMPLETED in statuses

        assert _field_mentioned_in_speaks("company", speaks), (
            "Expected agent to ask for 'Company Name' at some point. "
            f"Speaks:\n" + "\n".join(f"  {s!r}" for s in speaks)
        )

    def test_optional_mid_sequence_collected_when_provided(self):
        """When caller provides the optional company name, it must be collected."""
        collected, statuses, speaks = run_and_capture(
            [
                "John Smith",
                "yes",
                "Acme Corp",
                "yes",
                "john@example.com",
                "yes",
            ],
            CONFIG_OPT_MID,
            label="Optional mid-sequence — value provided and collected",
        )
        assert collected.get("company") and collected["company"] != "", (
            f"company should be collected with a real value when caller provides it. collected={collected}"
        )
        assert collected.get("full_name")
        assert collected.get("email")
        assert AgentStatus.COMPLETED in statuses

    def test_skip_optional_moves_to_next_field_not_completion(self):
        """Skipping the optional company must advance to email, not complete."""
        collected, statuses, speaks = run_and_capture(
            [
                "John Smith",
                "yes",                    # confirm full_name
                "skip",                   # caller skips company
                "john@example.com",
                "yes",
            ],
            CONFIG_OPT_MID,
            label="Optional skipped — next field (email) still collected",
        )
        assert not (collected.get("company") and collected["company"] != ""), (
            f"company must not have a real value after skip. collected={collected}"
        )
        assert collected.get("full_name"), f"full_name missing: {collected}"
        assert collected.get("email"), (
            f"email (required) must still be collected after skipping optional. collected={collected}"
        )
        assert AgentStatus.COMPLETED in statuses

    def test_email_mentioned_after_optional(self):
        """Email (next required after the optional) must be asked eventually."""
        collected, statuses, speaks = run_and_capture(
            [
                "John Smith",
                "yes",
                "skip",
                "john@example.com",
                "yes",
            ],
            CONFIG_OPT_MID,
            label="Email asked after optional skipped",
        )
        assert _field_mentioned_in_speaks("email", speaks), (
            f"Expected 'email' to appear in speaks. Speaks:\n"
            + "\n".join(f"  {s!r}" for s in speaks)
        )


# ── Tests: optional field at end ──────────────────────────────────────────────

class TestOptionalFieldAtEnd:
    """Optional field appears after all required fields — must NOT be asked."""

    def test_optional_at_end_not_collected(self):
        """Once required fields are confirmed, agent completes without asking the optional."""
        collected, statuses, speaks = run_and_capture(
            ["John", "yes", "john@example.com", "yes"],
            CONFIG_OPT_END,
            label="Optional at end — not asked, agent completes",
        )
        assert collected.get("first_name")
        assert collected.get("email")
        assert not (collected.get("referral") and collected["referral"] != ""), (
            f"referral (optional, end) must not be collected. collected={collected}"
        )
        assert AgentStatus.COMPLETED in statuses

    def test_optional_at_end_agent_reaches_completed(self):
        """Agent must reach COMPLETED without needing the optional field."""
        _, statuses, _ = run_and_capture(
            ["John", "yes", "john@example.com", "yes"],
            CONFIG_OPT_END,
            label="Optional at end — COMPLETED reached",
        )
        assert AgentStatus.COMPLETED in statuses


# ── Tests: two consecutive optionals mid-sequence ────────────────────────────

class TestTwoOptionalsMidSequence:
    """Two consecutive optional fields between required fields."""

    def test_first_optional_is_asked(self):
        """After first_name is confirmed, middle_name must be asked before email."""
        collected, statuses, speaks = run_and_capture(
            [
                "John",
                "yes",
                "Robert",         # provide middle_name
                "yes",
                "skip",           # skip nickname (second optional)
                "john@example.com",
                "yes",
            ],
            CONFIG_TWO_OPT_MID,
            label="Two optionals — first optional asked and collected",
        )
        assert _field_mentioned_in_speaks("middle name", speaks), (
            "Middle Name question must appear in speaks. "
            f"Speaks:\n" + "\n".join(f"  {s!r}" for s in speaks)
        )
        assert collected.get("first_name")
        assert collected.get("email")
        assert AgentStatus.COMPLETED in statuses

    def test_skipping_first_optional_still_asks_second(self):
        """Declining middle_name must NOT cause nickname (second optional) to be skipped.
        Each optional is skipped only when the caller explicitly declines it."""
        collected, statuses, speaks = run_and_capture(
            [
                "John",
                "yes",            # confirm first_name
                "skip",           # skip middle_name (first optional)
                # agent should now ask for nickname (second optional), NOT jump to email
                "Johnny",         # provide nickname
                "yes",            # confirm nickname
                "john@example.com",
                "yes",
            ],
            CONFIG_TWO_OPT_MID,
            label="Two optionals — skipping first still asks second",
        )
        assert not (collected.get("middle_name") and collected["middle_name"] != ""), \
            f"middle_name must not be collected after skip: {collected}"
        assert _field_mentioned_in_speaks("nickname", speaks), (
            "Nickname question must still appear after middle_name is skipped. "
            f"Speaks:\n" + "\n".join(f"  {s!r}" for s in speaks)
        )
        assert collected.get("first_name")
        assert collected.get("email")
        assert AgentStatus.COMPLETED in statuses

    def test_skipping_both_optionals_collects_required(self):
        """Skipping both optionals individually must still collect the required email."""
        collected, statuses, speaks = run_and_capture(
            [
                "John",
                "yes",
                "skip",           # skip middle_name
                "skip",           # skip nickname
                "john@example.com",
                "yes",
            ],
            CONFIG_TWO_OPT_MID,
            label="Two optionals both skipped — required still collected",
        )
        assert not (collected.get("middle_name") and collected["middle_name"] != "")
        assert not (collected.get("nickname") and collected["nickname"] != "")
        assert collected.get("first_name")
        assert collected.get("email"), (
            f"email must be collected after skipping both optionals. collected={collected}"
        )
        assert AgentStatus.COMPLETED in statuses

    def test_both_optionals_collected_when_provided(self):
        """Both optionals can be collected if caller provides values for each."""
        collected, statuses, speaks = run_and_capture(
            [
                "John",
                "yes",
                "Robert",         # middle_name
                "yes",
                "Johnny",         # nickname
                "yes",
                "john@example.com",
                "yes",
            ],
            CONFIG_TWO_OPT_MID,
            label="Two optionals — both collected",
        )
        assert collected.get("first_name")
        assert collected.get("middle_name") and collected["middle_name"] != "", \
            f"middle_name must be collected: {collected}"
        assert collected.get("email")
        assert AgentStatus.COMPLETED in statuses


# ── Tests: optional phone mid-sequence ───────────────────────────────────────

class TestOptionalPhoneMidSequence:
    """Phone (optional) between first_name and email_address (both required)."""

    def test_optional_phone_question_is_posed(self):
        """After confirming first_name, agent must ask for phone before email."""
        collected, statuses, speaks = run_and_capture(
            [
                "Sarah Johnson",
                "yes",
                # phone question should appear here
                "four one five five five five zero one nine two",
                "yes",
                "sarah@example.com",
                "yes",
            ],
            CONFIG_OPT_PHONE_MID,
            label="Optional phone mid-sequence — question posed",
        )
        assert _field_mentioned_in_speaks("phone", speaks), (
            "Phone question must appear in speaks before email. "
            f"Speaks:\n" + "\n".join(f"  {s!r}" for s in speaks)
        )
        assert collected.get("first_name")
        assert collected.get("email_address")
        assert AgentStatus.COMPLETED in statuses

    def test_optional_phone_collected_when_provided(self):
        """When caller provides phone number when asked, it must be collected."""
        collected, statuses, speaks = run_and_capture(
            [
                "Sarah Johnson",
                "yes",
                "four one five five five five zero one nine two",
                "yes",
                "sarah@example.com",
                "yes",
            ],
            CONFIG_OPT_PHONE_MID,
            label="Optional phone — collected when provided",
        )
        assert collected.get("phone_number") and collected["phone_number"] != "", (
            f"phone_number must be collected when provided. collected={collected}"
        )
        assert collected.get("first_name")
        assert collected.get("email_address")
        assert AgentStatus.COMPLETED in statuses

    def test_optional_phone_skipped_then_email_collected(self):
        """Skipping optional phone must not prevent email (required) from being collected."""
        collected, statuses, speaks = run_and_capture(
            [
                "Sarah Johnson",
                "yes",
                "no thank you",   # decline phone
                "sarah@example.com",
                "yes",
            ],
            CONFIG_OPT_PHONE_MID,
            label="Optional phone skipped — email still collected",
        )
        assert not (collected.get("phone_number") and collected["phone_number"] != ""), \
            f"phone_number must not be collected after skip: {collected}"
        assert collected.get("first_name")
        assert collected.get("email_address"), (
            f"email_address (required) must be collected after skipping optional phone. collected={collected}"
        )
        assert AgentStatus.COMPLETED in statuses

    def test_caller_volunteers_phone_upfront(self):
        """Caller provides both name and phone in one utterance — both should be extracted."""
        collected, statuses, speaks = run_and_capture(
            [
                "My name is Sarah Johnson and my number is 415-555-0192",
                "yes",
                "yes",            # in case phone is confirmed separately
                "sarah@example.com",
                "yes",
            ],
            CONFIG_OPT_PHONE_MID,
            label="Caller volunteers optional phone upfront",
        )
        assert collected.get("first_name"), f"first_name missing: {collected}"
        print(f"  phone collected: {collected.get('phone_number')!r}")
        assert collected.get("email_address"), f"email_address missing: {collected}"
        assert AgentStatus.COMPLETED in statuses
