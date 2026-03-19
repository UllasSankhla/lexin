"""
Unit tests for NarrativeCollectionAgent (simplified one-ask flow).

Flow under test:
  1. Collect caller utterances, respond with neutral fillers.
  2. Once _has_enough_content (MIN_SEGMENTS meaningful segments), ask
     "Is there anything else?" — exactly once.
  3. Caller responds:
       done    → complete with existing segments
       not done → collect that utterance as a final segment, then complete
  There is no loop, no LLM completion check, no "anything else?" in fillers.

Bug guards:
  [BUG-A] _pick_filler must never contain "Is there anything else?"
          (old bug: filler bypassed stage control, caller's "No" became a segment)
  [BUG-B] asking_done must always complete — never loop back to collecting
  [BUG-C] _DONE_INTENT_SYSTEM must not use double-negative framing that
          causes "No" → done=false
  [BUG-D] Gate-only responses ("No", "That's all") must not contaminate
          segments when the caller is truly done

All LLM calls are patched — no network access or API keys required.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from app.agents.narrative_collection import (
    NarrativeCollectionAgent,
    _MIN_SEGMENTS,
    _MIN_WORDS_IN_SEGMENT,
    _FILLERS,
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
    """Two meaningful segments — satisfies _has_enough_content."""
    return [
        "I was in a car accident last month on the highway",
        "The other driver ran a red light and hit my passenger door",
    ]


def _asking_done_state() -> dict:
    return {
        "stage": "asking_done",
        "segments": list(_enough_segments()),
        "summary": None,
        "case_type": None,
    }


# ── Opening turn ──────────────────────────────────────────────────────────────

def test_opening_turn_returns_prompt():
    resp, state = _call("", {})
    assert resp.status == AgentStatus.IN_PROGRESS
    assert resp.speak
    assert state["stage"] == "collecting"
    assert state["segments"] == []


# ── Collecting: fillers ───────────────────────────────────────────────────────

def test_short_utterance_gets_filler_not_done_question():
    """A segment below MIN_WORDS should get a plain filler, not 'Is there anything else?'"""
    _, state = _call("", {})
    resp, state = _call("Short.", state)
    assert resp.status == AgentStatus.IN_PROGRESS
    assert "anything else" not in resp.speak.lower()
    assert state["stage"] == "collecting"


def test_first_meaningful_segment_gets_filler():
    """After first meaningful segment (only 1, below MIN_SEGMENTS), expect filler."""
    _, state = _call("", {})
    resp, state = _call("I was in a car accident last month on the highway.", state)
    assert resp.status == AgentStatus.IN_PROGRESS
    assert "anything else" not in resp.speak.lower()
    assert state["stage"] == "collecting"


def test_utterance_appended_during_collecting():
    _, state = _call("", {})
    _, state = _call("I was in a car accident last month.", state)
    assert "I was in a car accident last month." in state["segments"]


# ── [BUG-A] Fillers must not contain "Is there anything else?" ───────────────

def test_no_filler_contains_anything_else():
    """
    BUG-A: The old _pick_filler returned 'Is there anything else?' for
    segment_count >= MIN_SEGMENTS+2. The caller's 'No' then hit the collecting
    branch, was appended as a segment, and the logic broke.

    Guard: no filler phrase may ask a question that the agent cannot handle.
    """
    for filler in _FILLERS:
        assert "anything else" not in filler.lower(), (
            f"Filler contains 'anything else': {filler!r}"
        )


# ── Transition to asking_done ─────────────────────────────────────────────────

def test_transitions_to_asking_done_after_min_segments():
    """After MIN_SEGMENTS meaningful segments, stage must become asking_done."""
    _, state = _call("", {})
    _, state = _call("I was in a car accident last month on the highway.", state)
    resp, state = _call("The other driver ran a red light and hit my car door.", state)
    assert state["stage"] == "asking_done"
    assert "anything else" in resp.speak.lower()


def test_does_not_ask_done_before_min_segments():
    _, state = _call("", {})
    resp, state = _call("I was in a car accident last month on the highway.", state)
    assert state["stage"] == "collecting"
    assert "anything else" not in resp.speak.lower()


def test_does_not_ask_done_for_short_segments():
    """Two segments but each under MIN_WORDS — should stay collecting."""
    _, state = _call("", {})
    _, state = _call("Short.", state)
    resp, state = _call("Also short.", state)
    assert state["stage"] == "collecting"


# ── [BUG-B] asking_done always completes — never loops ───────────────────────

@patch("app.agents.narrative_collection.llm_text_call", return_value="Thank you.")
@patch("app.agents.narrative_collection.llm_json_call")
def test_asking_done_completes_when_done(mock_json, mock_text):
    """done_intent=True → COMPLETED."""
    mock_json.side_effect = [
        {"done": True},
        {"summary": "Summary.", "case_type": "personal injury"},
    ]
    resp, _ = _call("No that's all", _asking_done_state())
    assert resp.status == AgentStatus.COMPLETED


@patch("app.agents.narrative_collection.llm_text_call", return_value="Thank you.")
@patch("app.agents.narrative_collection.llm_json_call")
def test_asking_done_completes_even_when_not_done(mock_json, mock_text):
    """
    BUG-B: The old code looped back to collecting when done_intent=False.
    Now: if the caller has more to add, collect it as a final segment, then
    complete immediately. There is no path back to collecting.
    """
    mock_json.side_effect = [
        {"done": False},
        {"summary": "Summary.", "case_type": "personal injury"},
    ]
    resp, _ = _call("Actually I should also mention the injuries.", _asking_done_state())
    assert resp.status == AgentStatus.COMPLETED, (
        "Agent must complete after asking_done regardless of done_intent — "
        "the old code looped back to collecting here"
    )


@patch("app.agents.narrative_collection.llm_text_call", return_value="Thank you.")
@patch("app.agents.narrative_collection.llm_json_call")
def test_asking_done_never_returns_in_progress_with_stage_collecting(mock_json, mock_text):
    """
    Stage must never go back to 'collecting' from 'asking_done'.
    """
    mock_json.side_effect = [
        {"done": False},
        {"summary": "Summary.", "case_type": "personal injury"},
    ]
    _, new_state = _call("Wait I want to add more", _asking_done_state())
    assert new_state.get("stage") != "collecting", (
        "Stage returned to collecting from asking_done — loop bug"
    )


# ── [BUG-D] Segments content when done ───────────────────────────────────────

@patch("app.agents.narrative_collection.llm_text_call", return_value="Noted.")
@patch("app.agents.narrative_collection.llm_json_call")
def test_done_response_not_added_to_segments(mock_json, mock_text):
    """
    BUG-D: When done_intent=True, the caller's 'No' should NOT be appended
    to segments — it's a conversational reply, not narrative content.
    """
    mock_json.side_effect = [
        {"done": True},
        {"summary": "Summary.", "case_type": "unknown"},
    ]
    state = _asking_done_state()
    original_segment_count = len(state["segments"])

    resp, _ = _call("No that's all", state)
    assert "No that's all" not in resp.hidden_collected["full_narrative"]
    # Segment count should not have grown
    assert len(resp.hidden_collected["full_narrative"].split(". ")) <= original_segment_count + 1


@patch("app.agents.narrative_collection.llm_text_call", return_value="Noted.")
@patch("app.agents.narrative_collection.llm_json_call")
def test_additional_content_added_to_segments_when_not_done(mock_json, mock_text):
    """When done_intent=False, the utterance IS appended (it's new narrative content)."""
    mock_json.side_effect = [
        {"done": False},
        {"summary": "Summary.", "case_type": "personal injury"},
    ]
    extra = "I also sustained a back injury and missed three weeks of work."
    resp, _ = _call(extra, _asking_done_state())
    assert extra in resp.hidden_collected["full_narrative"]


# ── [BUG-C] Prompt framing ────────────────────────────────────────────────────

def test_done_intent_prompt_not_double_negative():
    """
    BUG-C: Old prompt asked 'Does their response indicate they are done?'
    LLM saw 'No' and answered done=false (double negative). New prompt must
    use stop/continue framing with explicit 'No' example.
    """
    assert "does their response indicate they are done" not in _DONE_INTENT_SYSTEM.lower()


def test_done_intent_prompt_has_no_example_for_stop():
    prompt_lower = _DONE_INTENT_SYSTEM.lower()
    assert "'no'" in prompt_lower or '"no"' in prompt_lower


def test_done_intent_prompt_has_stop_and_continue_framing():
    prompt_lower = _DONE_INTENT_SYSTEM.lower()
    assert "stop" in prompt_lower or "nothing more" in prompt_lower
    assert "continue" in prompt_lower or "more to say" in prompt_lower


# ── COMPLETED output ──────────────────────────────────────────────────────────

@patch("app.agents.narrative_collection.llm_text_call", return_value="Noted.")
@patch("app.agents.narrative_collection.llm_json_call")
def test_completed_collected_dict(mock_json, mock_text):
    mock_json.side_effect = [
        {"done": True},
        {"summary": "Car accident on highway.", "case_type": "personal injury"},
    ]
    resp, _ = _call("No", _asking_done_state())
    assert resp.collected["narrative_summary"] == "Car accident on highway."
    assert resp.hidden_collected["case_type"] == "personal injury"
    assert resp.hidden_collected["full_narrative"]


@patch("app.agents.narrative_collection.llm_text_call", return_value="Noted.")
@patch("app.agents.narrative_collection.llm_json_call")
def test_full_narrative_is_segment_join(mock_json, mock_text):
    mock_json.side_effect = [
        {"done": True},
        {"summary": "Summary.", "case_type": "unknown"},
    ]
    segs = ["First part of narrative.", "Second part of narrative."]
    state = {**_asking_done_state(), "segments": segs}
    resp, _ = _call("No", state)
    assert resp.hidden_collected["full_narrative"] == " ".join(segs)


# ── Resume handling ───────────────────────────────────────────────────────────

def test_resume_collecting_re_prompts_without_appending():
    state = {
        "stage": "collecting",
        "segments": ["I was in an accident."],
        "summary": None,
        "case_type": None,
    }
    resp, new_state = _call("", state)
    assert resp.status == AgentStatus.IN_PROGRESS
    assert new_state["segments"] == ["I was in an accident."]


def test_resume_asking_done_re_asks_gate():
    resp, new_state = _call("", _asking_done_state())
    assert resp.status == AgentStatus.IN_PROGRESS
    assert "anything else" in resp.speak.lower()
    assert new_state["stage"] == "asking_done"


def test_segments_preserved_across_resume():
    original = ["I was in a car accident last month on the highway.",
                "The other driver ran a red light."]
    state = {"stage": "collecting", "segments": original, "summary": None, "case_type": None}
    _, new_state = _call("", state)
    assert new_state["segments"] == original


# ── LLM failure resilience ────────────────────────────────────────────────────

@patch("app.agents.narrative_collection.llm_json_call", side_effect=Exception("timeout"))
def test_done_intent_failure_still_completes(mock_json):
    """
    On LLM failure in done_intent, default is not-done (collect utterance).
    Either way, the agent must complete — no loop.
    """
    with patch("app.agents.narrative_collection.llm_text_call", return_value=""):
        resp, _ = _call("No", _asking_done_state())
    assert resp.status == AgentStatus.COMPLETED


@patch("app.agents.narrative_collection.llm_text_call", return_value="")
@patch("app.agents.narrative_collection.llm_json_call")
def test_summarise_failure_uses_fallback(mock_json, mock_text):
    mock_json.side_effect = [{"done": True}, Exception("summarise failed")]
    resp, _ = _call("No", _asking_done_state())
    assert resp.status == AgentStatus.COMPLETED
    assert resp.collected["narrative_summary"]
    assert resp.hidden_collected["case_type"]
