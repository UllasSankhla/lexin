"""
Unit tests for NarrativeCollectionAgent.

Tests are designed to catch real bugs, not just happy-path behaviour with
ideal mocked LLM output.  Each test group documents the bug it guards against.

Covers:
  - Opening turn (empty utterance) produces a prompt
  - Segment accumulation and filler responses
  - Completion check gate (MIN_SEGMENTS, MIN_WORDS)
  - Transition to asking_done stage

  BUG GUARDS:
  - [BUG-1] Prompt framing: _DONE_INTENT_SYSTEM must not cause "No" → done=false
  - [BUG-2] Loop prevention: completion check must NOT re-fire immediately after
            returning from asking_done to collecting (no new content added yet)
  - [BUG-3] Gate response must NOT be appended to segments as narrative content
  - [BUG-4] Safety cap: after _MAX_DONE_ASKS attempts, complete regardless of
            what the LLM returns for done_intent

  - COMPLETED path: summary + case_type, collected dict, full_narrative join
  - Resume after interrupt (empty utterance) in both stages
  - Segments preserved across resume invocations
  - LLM failure resilience (defaults to safe conservative values)

All LLM calls are patched — no network access or API keys required.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from app.agents.narrative_collection import (
    NarrativeCollectionAgent,
    _MIN_SEGMENTS,
    _MIN_WORDS_IN_SEGMENT,
    _MAX_DONE_ASKS,
    _DONE_INTENT_SYSTEM,
)
from app.agents.base import AgentStatus


# ── Helpers ───────────────────────────────────────────────────────────────────

AGENT = NarrativeCollectionAgent()
CONFIG = {"practice_areas": ["personal injury", "family law", "employment"]}
HISTORY: list[dict] = []


def _call(utterance: str, state: dict) -> tuple:
    resp = AGENT.process(utterance, state, CONFIG, HISTORY)
    return resp, resp.internal_state


def _enough_segments() -> list[str]:
    return [
        "I was in a car accident last month on the highway",
        "The other driver ran a red light and hit my passenger door",
    ]


def _asking_done_state() -> dict:
    """State already in asking_done with enough segments and first ask recorded."""
    return {
        "stage": "asking_done",
        "segments": list(_enough_segments()),
        "summary": None,
        "case_type": None,
        "done_ask_count": 1,
        "segments_at_done_check": 2,
    }


# ── Opening turn ──────────────────────────────────────────────────────────────

def test_opening_turn_returns_in_progress_with_prompt():
    resp, state = _call("", {})
    assert resp.status == AgentStatus.IN_PROGRESS
    assert resp.speak
    assert state["stage"] == "collecting"
    assert state["segments"] == []


def test_opening_turn_initialises_all_state_keys():
    _, state = _call("", {})
    for key in ("stage", "segments", "summary", "case_type", "done_ask_count", "segments_at_done_check"):
        assert key in state, f"Missing state key: {key}"


# ── Segment accumulation ──────────────────────────────────────────────────────

@patch("app.agents.narrative_collection.llm_json_call", return_value={"complete": False})
def test_utterance_appended_to_segments(mock_llm):
    _, state = _call("", {})
    _, state = _call("I was in a car accident last month.", state)
    assert "I was in a car accident last month." in state["segments"]


@patch("app.agents.narrative_collection.llm_json_call", return_value={"complete": False})
def test_multiple_utterances_all_appended(mock_llm):
    _, state = _call("", {})
    _, state = _call("I was in a car accident.", state)
    _, state = _call("The other driver ran a red light.", state)
    assert len(state["segments"]) == 2


@patch("app.agents.narrative_collection.llm_json_call", return_value={"complete": False})
def test_filler_response_while_collecting(mock_llm):
    _, state = _call("", {})
    resp, _ = _call("I was in a car accident.", state)
    assert resp.status == AgentStatus.IN_PROGRESS
    assert resp.speak


# ── Completion check gate ─────────────────────────────────────────────────────

@patch("app.agents.narrative_collection.llm_json_call", return_value={"complete": True})
def test_completion_check_not_called_before_min_segments(mock_llm):
    _, state = _call("", {})
    _, state = _call("Just one short sentence.", state)
    mock_llm.assert_not_called()
    assert state["stage"] == "collecting"


@patch("app.agents.narrative_collection.llm_json_call", return_value={"complete": True})
def test_completion_check_not_called_for_short_segments(mock_llm):
    _, state = _call("", {})
    _, state = _call("Short.", state)
    _, state = _call("Also short.", state)
    mock_llm.assert_not_called()
    assert state["stage"] == "collecting"


@patch("app.agents.narrative_collection.llm_json_call", return_value={"complete": True})
def test_completion_check_fires_after_enough_content(mock_llm):
    _, state = _call("", {})
    _, state = _call("I was in a car accident last month on the highway.", state)
    _, state = _call("The other driver ran a red light and hit my car door.", state)
    mock_llm.assert_called_once()
    assert state["stage"] == "asking_done"


@patch("app.agents.narrative_collection.llm_json_call", return_value={"complete": True})
def test_done_ask_count_incremented_on_first_ask(mock_llm):
    _, state = _call("", {})
    _, state = _call("I was in a car accident last month on the highway.", state)
    _, state = _call("The other driver ran a red light and hit my car door.", state)
    assert state["done_ask_count"] == 1


@patch("app.agents.narrative_collection.llm_json_call", return_value={"complete": True})
def test_segments_at_done_check_recorded(mock_llm):
    _, state = _call("", {})
    _, state = _call("I was in a car accident last month on the highway.", state)
    _, state = _call("The other driver ran a red light and hit my car door.", state)
    # 2 meaningful segments at the point of asking
    assert state["segments_at_done_check"] == 2


# ── [BUG-1] Prompt framing ────────────────────────────────────────────────────

def test_done_intent_prompt_has_explicit_no_example():
    """
    BUG-1: The original _DONE_INTENT_SYSTEM asked 'Does their response indicate
    they are done?' — the LLM could answer done=false for 'No' because 'No' is
    a negative word, even though 'No' to 'anything else?' means done=true.

    Guard: prompt must contain an explicit example mapping 'No' → done=true.
    """
    prompt_lower = _DONE_INTENT_SYSTEM.lower()
    # Must frame as stop/continue, not as a yes/no question about done-ness
    assert "stop" in prompt_lower or "nothing more" in prompt_lower
    # Must have an example showing 'No' maps to done=true
    assert "'no'" in prompt_lower or "\"no\"" in prompt_lower


def test_done_intent_prompt_has_examples_for_both_directions():
    """Prompt must give examples for STOP and for CONTINUE to anchor the LLM."""
    prompt_lower = _DONE_INTENT_SYSTEM.lower()
    assert "continue" in prompt_lower or "more to say" in prompt_lower
    assert "stop" in prompt_lower or "nothing more" in prompt_lower


def test_done_intent_prompt_not_a_double_negative():
    """
    The old framing 'Does their response indicate they are done?' with the
    caller saying 'No' creates a double negative. The new prompt should frame
    the question as stop vs continue, not done vs not-done.
    """
    # Should NOT use the old ambiguous phrasing
    assert "does their response indicate they are done" not in _DONE_INTENT_SYSTEM.lower()


# ── [BUG-2] No immediate re-ask after returning to collecting ─────────────────

@patch("app.agents.narrative_collection.llm_json_call", return_value={"complete": True})
def test_completion_check_does_not_refire_immediately_after_returning_to_collecting(mock_llm):
    """
    BUG-2: After done_intent=False, stage goes back to collecting. The narrative
    was already 'complete' before. Without a guard, the very next utterance would
    immediately trigger the completion check again and re-ask 'Is there anything
    else?' — creating an infinite loop.

    Guard: completion check must not fire unless at least one new meaningful
    segment has been added beyond segments_at_done_check.
    """
    # Start in collecting, build up enough content to reach asking_done
    _, state = _call("", {})
    _, state = _call("I was in a car accident last month on the highway.", state)
    mock_llm.return_value = {"complete": True}
    _, state = _call("The other driver ran a red light and hit my car door.", state)
    assert state["stage"] == "asking_done"

    # Simulate done_intent=False — caller goes back to collecting
    with patch("app.agents.narrative_collection.llm_json_call", return_value={"done": False}):
        _, state = _call("Actually wait there is more", state)
    assert state["stage"] == "collecting"

    # Reset mock to return complete=True again
    mock_llm.reset_mock()
    mock_llm.return_value = {"complete": True}

    # The VERY NEXT utterance should NOT immediately ask "Is there anything else?"
    # because segments_at_done_check equals the current meaningful segment count —
    # no new meaningful content has been added yet.
    resp, state = _call("short", state)  # short utterance — not meaningful
    mock_llm.assert_not_called()
    assert state["stage"] == "collecting", (
        "Agent asked 'Is there anything else?' without any new meaningful content — loop bug"
    )


@patch("app.agents.narrative_collection.llm_json_call", return_value={"complete": True})
def test_completion_check_refires_after_new_meaningful_segment(mock_llm):
    """
    After returning to collecting and then adding a real new segment (enough
    words), the completion check SHOULD fire again.
    """
    state = {
        "stage": "collecting",
        "segments": list(_enough_segments()),
        "summary": None,
        "case_type": None,
        "done_ask_count": 1,
        "segments_at_done_check": 2,  # already asked once with 2 meaningful segments
    }

    # Add a new meaningful segment (> 4 words) — now meaningful_count=3 > 2
    resp, state = _call("Also I sustained significant injuries to my back and neck.", state)
    mock_llm.assert_called_once()
    assert state["stage"] == "asking_done"
    assert state["done_ask_count"] == 2


# ── [BUG-3] Gate response not appended to segments ───────────────────────────

@patch("app.agents.narrative_collection.llm_json_call", return_value={"done": False})
def test_gate_response_not_appended_to_segments(mock_llm):
    """
    BUG-3: When asking_done and done_intent=False, the original code appended
    the utterance ('No', 'Actually...') to segments. This meant:
    1. 'No' ended up in full_narrative ("...accident. No.")
    2. The completion check saw 'No' as a new segment and immediately re-asked.

    Guard: segments must be unchanged after a done_intent=False response.
    """
    state = _asking_done_state()
    original_segments = list(state["segments"])

    _, new_state = _call("No actually wait", state)

    assert new_state["segments"] == original_segments, (
        "Gate response was appended to segments — will corrupt full_narrative"
    )


@patch("app.agents.narrative_collection.llm_json_call", return_value={"done": False})
def test_stage_returns_to_collecting_after_not_done(mock_llm):
    state = _asking_done_state()
    _, new_state = _call("Actually I have more to say", state)
    assert new_state["stage"] == "collecting"


# ── [BUG-4] Safety cap ───────────────────────────────────────────────────────

def test_max_done_asks_constant_is_defined_and_positive():
    """_MAX_DONE_ASKS must be defined and ≥ 1."""
    assert _MAX_DONE_ASKS >= 1


@patch("app.agents.narrative_collection.llm_text_call", return_value="Thank you.")
@patch("app.agents.narrative_collection.llm_json_call")
def test_safety_cap_completes_after_max_done_asks(mock_json, mock_text):
    """
    BUG-4: If the LLM repeatedly returns done=False even when the caller says
    'No', the agent would loop forever. The safety cap must complete the agent
    after _MAX_DONE_ASKS regardless of the LLM's answer.

    This test simulates the exact failure mode: done_intent always returns False,
    but the agent must still complete once the cap is reached.
    """
    mock_json.side_effect = [
        {"done": False},  # first ask: LLM wrong
        {"summary": "Summary.", "case_type": "personal injury"},
    ]

    # State already at the cap
    state = {
        "stage": "asking_done",
        "segments": list(_enough_segments()),
        "summary": None,
        "case_type": None,
        "done_ask_count": _MAX_DONE_ASKS,  # already at the limit
        "segments_at_done_check": 2,
    }

    resp, _ = _call("No", state)
    assert resp.status == AgentStatus.COMPLETED, (
        f"Agent should have completed after {_MAX_DONE_ASKS} asks but got {resp.status}"
    )
    # done_intent LLM should NOT have been called — cap fires before it
    assert mock_json.call_count == 1  # only the summary call


@patch("app.agents.narrative_collection.llm_text_call", return_value="Thank you.")
@patch("app.agents.narrative_collection.llm_json_call")
def test_safety_cap_not_triggered_before_max(mock_json, mock_text):
    """Safety cap must not fire before the limit is reached."""
    mock_json.return_value = {"done": False}

    state = {
        "stage": "asking_done",
        "segments": list(_enough_segments()),
        "summary": None,
        "case_type": None,
        "done_ask_count": _MAX_DONE_ASKS - 1,  # one below the cap
        "segments_at_done_check": 2,
    }

    resp, _ = _call("No", state)
    # Should still be in_progress — cap not yet reached, LLM said not done
    assert resp.status == AgentStatus.IN_PROGRESS


# ── COMPLETED path ────────────────────────────────────────────────────────────

@patch("app.agents.narrative_collection.llm_text_call", return_value="Thank you, I've noted your matter.")
@patch("app.agents.narrative_collection.llm_json_call")
def test_completed_on_done_intent_true(mock_json, mock_text):
    mock_json.side_effect = [
        {"done": True},
        {"summary": "Caller was in a car accident.", "case_type": "personal injury"},
    ]
    resp, _ = _call("No that's all", _asking_done_state())
    assert resp.status == AgentStatus.COMPLETED


@patch("app.agents.narrative_collection.llm_text_call", return_value="Noted.")
@patch("app.agents.narrative_collection.llm_json_call")
def test_completed_populates_collected_dict(mock_json, mock_text):
    mock_json.side_effect = [
        {"done": True},
        {"summary": "Caller was in a car accident.", "case_type": "personal injury"},
    ]
    resp, _ = _call("No", _asking_done_state())
    assert resp.collected["narrative_summary"] == "Caller was in a car accident."
    assert resp.collected["case_type"] == "personal injury"
    assert resp.collected["full_narrative"]


@patch("app.agents.narrative_collection.llm_text_call", return_value="Noted.")
@patch("app.agents.narrative_collection.llm_json_call")
def test_full_narrative_does_not_include_gate_responses(mock_json, mock_text):
    """
    Gate responses ('No', 'That\\'s all') must not appear in full_narrative.
    Since we no longer append gate responses to segments, they won't be in the join.
    """
    mock_json.side_effect = [
        {"done": True},
        {"summary": "Summary.", "case_type": "unknown"},
    ]
    state = _asking_done_state()
    resp, _ = _call("No that's all", state)
    assert "No" not in resp.collected["full_narrative"]
    assert "That's all" not in resp.collected["full_narrative"]


@patch("app.agents.narrative_collection.llm_text_call", return_value="Noted.")
@patch("app.agents.narrative_collection.llm_json_call")
def test_full_narrative_is_join_of_segments(mock_json, mock_text):
    mock_json.side_effect = [
        {"done": True},
        {"summary": "Summary.", "case_type": "unknown"},
    ]
    segs = ["First segment about the accident.", "Second segment about the injuries."]
    state = {**_asking_done_state(), "segments": segs}
    resp, _ = _call("No", state)
    assert resp.collected["full_narrative"] == " ".join(segs)


# ── Resume handling ───────────────────────────────────────────────────────────

def test_resume_while_collecting_re_prompts():
    state = {
        "stage": "collecting",
        "segments": ["I was in an accident."],
        "summary": None,
        "case_type": None,
        "done_ask_count": 0,
        "segments_at_done_check": 0,
    }
    resp, new_state = _call("", state)
    assert resp.status == AgentStatus.IN_PROGRESS
    assert resp.speak
    # Empty utterance must not be appended as a segment
    assert new_state["segments"] == ["I was in an accident."]


def test_resume_while_asking_done_re_asks_gate():
    state = _asking_done_state()
    resp, new_state = _call("", state)
    assert resp.status == AgentStatus.IN_PROGRESS
    assert "anything else" in resp.speak.lower()
    assert new_state["stage"] == "asking_done"


def test_segments_preserved_across_resume():
    original_segs = list(_enough_segments())
    state = {
        "stage": "collecting",
        "segments": original_segs,
        "summary": None,
        "case_type": None,
        "done_ask_count": 0,
        "segments_at_done_check": 0,
    }
    _, new_state = _call("", state)
    assert new_state["segments"] == original_segs


# ── LLM failure resilience ────────────────────────────────────────────────────

@patch("app.agents.narrative_collection.llm_json_call", side_effect=Exception("timeout"))
def test_done_intent_failure_defaults_to_not_done(mock_json):
    """On LLM failure, default to not-done — never prematurely complete."""
    resp, new_state = _call("No that's all", _asking_done_state())
    assert resp.status == AgentStatus.IN_PROGRESS
    assert new_state["stage"] == "collecting"


@patch("app.agents.narrative_collection.llm_json_call", side_effect=Exception("timeout"))
def test_completion_check_failure_keeps_collecting(mock_json):
    _, state = _call("", {})
    _, state = _call("I was in a car accident last month on the highway.", state)
    resp, state = _call("The other driver ran a red light and hit my car door.", state)
    assert state["stage"] == "collecting"


@patch("app.agents.narrative_collection.llm_text_call", return_value="")
@patch("app.agents.narrative_collection.llm_json_call")
def test_summarise_failure_uses_fallback_values(mock_json, mock_text):
    mock_json.side_effect = [{"done": True}, Exception("summarise failed")]
    resp, _ = _call("No", _asking_done_state())
    assert resp.status == AgentStatus.COMPLETED
    assert resp.collected["narrative_summary"]
    assert resp.collected["case_type"]
    assert resp.speak  # hardcoded fallback
