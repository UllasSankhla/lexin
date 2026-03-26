"""Turn-by-turn LLM evaluator — compares AI responses to human agent responses.

For each (caller_utterance, human_response, ai_response) triple the evaluator
asks an LLM to score the AI response and identify the root cause of any gap.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Ensure data-plane is on the path so we can reuse the Cerebras LLM client.
_DATA_PLANE = Path(__file__).parents[2] / "data-plane"
if str(_DATA_PLANE) not in sys.path:
    sys.path.insert(0, str(_DATA_PLANE))

os.environ.setdefault("DEEPGRAM_API_KEY",      "eval-not-used")
os.environ.setdefault("CONTROL_PLANE_API_KEY", "eval-not-used")

from app.agents.llm_utils import llm_json_call  # noqa: E402

from evaluator.models import (
    ReplayResult, EvaluationFinding, FindingCategory, ConversationReport,
)
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# ── Evaluation prompt ─────────────────────────────────────────────────────────

_EVAL_SYSTEM = """\
You are an expert evaluator assessing an AI receptionist's performance in a law firm \
intake call. You will be given a caller utterance, the human agent's response, and the \
AI agent's response to the same utterance.

Score the AI response on a 1–5 scale:
  5 — Equally good or better than human (correct info, appropriate tone, nothing missing)
  4 — Good with only minor stylistic differences
  3 — Acceptable but noticeably missing something the human provided
  2 — Below par; significant information or tone gap compared to human
  1 — Poor; wrong, unhelpful, or confusing

Identify the ROOT CAUSE of any gap (score < 5):
  "faq"     — the AI lacked a specific factual answer the human had (needs a FAQ entry)
  "prompt"  — the AI's tone, phrasing, or behaviour needs a prompt instruction
  "context" — the AI lacked business/procedural context the human referenced
  "none"    — no meaningful gap; AI matched or exceeded human quality

Severity of gap:
  "critical" — caller would have a very poor experience or be misinformed
  "moderate" — noticeable but not harmful
  "minor"    — trivial stylistic difference
  "none"     — no gap

Reply ONLY with valid JSON (no markdown, no extra text):
{
  "quality_score": <1–5 integer>,
  "gap_description": "<one sentence; use 'No gap' when score >= 4 and category is none>",
  "category": "faq" | "prompt" | "context" | "none",
  "severity": "critical" | "moderate" | "minor" | "none"
}\
"""


def _evaluate_turn(
    caller_utterance: str,
    human_response: str,
    ai_response: str,
    conversation_context: str = "",
) -> dict:
    """Call the LLM evaluator for one turn. Returns parsed JSON dict."""
    user_msg = (
        f"CONVERSATION CONTEXT (preceding exchanges):\n{conversation_context}\n\n"
        if conversation_context else ""
    )
    user_msg += (
        f"CALLER SAID: \"{caller_utterance}\"\n\n"
        f"HUMAN AGENT RESPONSE:\n{human_response}\n\n"
        f"AI AGENT RESPONSE:\n{ai_response}"
    )
    return llm_json_call(_EVAL_SYSTEM, user_msg, max_tokens=512)


def evaluate_replay(
    replay_result: ReplayResult,
    config: dict,
) -> ConversationReport:
    """
    Evaluate each replayed turn against the human agent response from the TestCase.

    Pairs are matched by turn_index.  Turns with no paired human response (e.g.
    the AI continued past the human transcript) are skipped.
    """
    test_case = replay_result.test_case

    # Build lookup: caller turn_index → human agent response text
    human_by_turn: dict[int, str] = {
        caller_t.turn_index: agent_t.text
        for caller_t, agent_t in test_case.paired_exchanges
        if agent_t.text  # skip empty trailing agent turns
    }

    findings: list[EvaluationFinding] = []

    for replay_turn in replay_result.replay_turns:
        human_response = human_by_turn.get(replay_turn.turn_index)
        if not human_response:
            # No human baseline for this turn — skip evaluation
            continue

        if replay_turn.ai_response == "[ERROR]":
            findings.append(EvaluationFinding(
                turn_index=replay_turn.turn_index,
                caller_utterance=replay_turn.caller_utterance,
                human_agent_response=human_response,
                ai_response="[ERROR]",
                agent_id=replay_turn.agent_id,
                quality_score=1,
                gap_description="Agent threw an exception during replay.",
                category=FindingCategory.PROMPT,
                severity="critical",
            ))
            continue

        try:
            result = _evaluate_turn(
                caller_utterance=replay_turn.caller_utterance,
                human_response=human_response,
                ai_response=replay_turn.ai_response,
            )

            score    = int(result.get("quality_score", 3))
            score    = max(1, min(5, score))
            gap_desc = result.get("gap_description", "")
            raw_cat  = result.get("category", "none")
            severity = result.get("severity", "none")

            try:
                category = FindingCategory(raw_cat)
            except ValueError:
                category = FindingCategory.NONE

            findings.append(EvaluationFinding(
                turn_index=replay_turn.turn_index,
                caller_utterance=replay_turn.caller_utterance,
                human_agent_response=human_response,
                ai_response=replay_turn.ai_response,
                agent_id=replay_turn.agent_id,
                quality_score=score,
                gap_description=gap_desc,
                category=category,
                severity=severity,
            ))
            logger.info(
                "Evaluated turn %d | score=%d | cat=%s | sev=%s",
                replay_turn.turn_index, score, raw_cat, severity,
            )

        except Exception as exc:
            logger.warning(
                "Evaluation failed for turn %d: %s", replay_turn.turn_index, exc
            )

    overall = (
        sum(f.quality_score for f in findings) / len(findings)
        if findings else 0.0
    )

    return ConversationReport(
        conversation_id=replay_result.conversation_id,
        source_file=test_case.source_file,
        evaluated_at=datetime.now(timezone.utc).isoformat(),
        overall_quality_score=round(overall, 2),
        findings=findings,
        suggestions=[],  # populated by synthesis.py
        replay_turns=replay_result.replay_turns,
    )
