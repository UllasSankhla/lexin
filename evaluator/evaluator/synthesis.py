"""Suggestion synthesis — converts evaluation findings into actionable improvements.

Groups findings by category, asks an LLM to produce a concrete suggestion for
each cluster, and deduplicates across multiple conversation runs using a
persistent processed.json store keyed by suggestion_id.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from pathlib import Path

_DATA_PLANE = Path(__file__).parents[2] / "data-plane"
if str(_DATA_PLANE) not in sys.path:
    sys.path.insert(0, str(_DATA_PLANE))

os.environ.setdefault("DEEPGRAM_API_KEY",      "eval-not-used")
os.environ.setdefault("CONTROL_PLANE_API_KEY", "eval-not-used")

from app.agents.llm_utils import llm_json_call  # noqa: E402

from evaluator.models import (
    ConversationReport, EvaluationFinding, FindingCategory, ImprovementSuggestion,
    BatchReport,
)
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

_PROCESSED_PATH = Path(__file__).parents[1] / "data" / "processed.json"

# ── Suggestion generation prompts ─────────────────────────────────────────────

_FAQ_SUGGEST_SYSTEM = """\
You are improving an AI law firm receptionist's knowledge base.

Given one or more evaluation findings where the AI lacked factual knowledge, \
generate a single FAQ entry that would fix the gap.

Reply ONLY with valid JSON:
{
  "title": "<short descriptive title>",
  "faq_question": "<the question a caller would ask>",
  "faq_answer": "<the answer the receptionist should give>"
}\
"""

_PROMPT_SUGGEST_SYSTEM = """\
You are improving an AI law firm receptionist's system prompt.

Given one or more evaluation findings where the AI's tone, phrasing, or \
behaviour was subpar, write a single concise instruction rule to add to the \
agent's system prompt.

Reply ONLY with valid JSON:
{
  "title": "<short descriptive title>",
  "rule_to_add": "<the instruction to add to the prompt — one to three sentences>"
}\
"""

_CONTEXT_SUGGEST_SYSTEM = """\
You are improving an AI law firm receptionist's context documents.

Given one or more evaluation findings where the AI lacked business or procedural \
context, suggest a short context document that would address the gap.

Reply ONLY with valid JSON:
{
  "title": "<short descriptive title>",
  "document_name": "<descriptive filename, e.g. intake_policy.txt>",
  "document_content": "<the document text — one to five sentences>"
}\
"""


def _suggestion_id(category: str, title: str) -> str:
    """Stable short ID derived from category + normalised title."""
    key = f"{category}::{title.lower().strip()}"
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def _findings_block(findings: list[EvaluationFinding]) -> str:
    parts = []
    for f in findings:
        parts.append(
            f"Turn {f.turn_index} (score={f.quality_score}, severity={f.severity}):\n"
            f"  Caller: {f.caller_utterance}\n"
            f"  Human:  {f.human_agent_response}\n"
            f"  AI:     {f.ai_response}\n"
            f"  Gap:    {f.gap_description}"
        )
    return "\n\n".join(parts)


def _generate_suggestion(
    category: FindingCategory,
    findings: list[EvaluationFinding],
    conversation_ids: list[str],
) -> ImprovementSuggestion | None:
    if not findings:
        return None

    block = _findings_block(findings[:5])  # cap at 5 examples per LLM call

    system_map = {
        FindingCategory.FAQ:     _FAQ_SUGGEST_SYSTEM,
        FindingCategory.PROMPT:  _PROMPT_SUGGEST_SYSTEM,
        FindingCategory.CONTEXT: _CONTEXT_SUGGEST_SYSTEM,
    }
    system = system_map.get(category)
    if system is None:
        return None

    # Collect agent_ids (for PROMPT suggestions — agent_id is the target)
    agent_ids = list({f.agent_id for f in findings if f.agent_id and f.agent_id != "unknown"})
    agent_id = agent_ids[0] if len(agent_ids) == 1 else None

    avg_score = sum(f.quality_score for f in findings) / len(findings)
    composite = len(findings) * (5 - avg_score)  # more findings + lower scores = higher priority

    try:
        result = llm_json_call(system, f"FINDINGS:\n{block}", max_tokens=512)
    except Exception as exc:
        logger.warning("Suggestion LLM call failed for %s: %s", category.value, exc)
        return None

    title = result.get("title", f"Improvement for {category.value}")
    sid = _suggestion_id(category.value, title)

    return ImprovementSuggestion(
        suggestion_id=sid,
        category=category,
        agent_id=agent_id,
        title=title,
        faq_question=result.get("faq_question"),
        faq_answer=result.get("faq_answer"),
        rule_to_add=result.get("rule_to_add"),
        document_name=result.get("document_name"),
        document_content=result.get("document_content"),
        supporting_conversations=conversation_ids,
        frequency=len(findings),
        composite_score=round(composite, 2),
    )


# ── Deduplication store ───────────────────────────────────────────────────────

def _load_processed() -> dict:
    if _PROCESSED_PATH.exists():
        try:
            return json.loads(_PROCESSED_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_processed(data: dict) -> None:
    _PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    _PROCESSED_PATH.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ── Public API ────────────────────────────────────────────────────────────────

def synthesise_suggestions(
    reports: list[ConversationReport],
) -> list[ImprovementSuggestion]:
    """
    Aggregate findings across all conversation reports, cluster by category,
    and generate one improvement suggestion per cluster.

    Suggestions with duplicate IDs (same category + title seen in a prior run)
    have their frequency and supporting_conversations updated rather than being
    re-generated.

    Returns the full de-duplicated list of suggestions sorted by composite_score.
    """
    processed = _load_processed()

    # Collect all actionable findings (exclude NONE category)
    by_category: dict[FindingCategory, list[EvaluationFinding]] = {
        FindingCategory.FAQ:     [],
        FindingCategory.PROMPT:  [],
        FindingCategory.CONTEXT: [],
    }
    conv_ids_by_category: dict[FindingCategory, list[str]] = {k: [] for k in by_category}

    for report in reports:
        for finding in report.findings:
            if finding.category in by_category:
                by_category[finding.category].append(finding)
                if report.conversation_id not in conv_ids_by_category[finding.category]:
                    conv_ids_by_category[finding.category].append(report.conversation_id)

    suggestions: list[ImprovementSuggestion] = []

    for category, findings in by_category.items():
        if not findings:
            continue

        suggestion = _generate_suggestion(
            category, findings, conv_ids_by_category[category]
        )
        if suggestion is None:
            continue

        sid = suggestion.suggestion_id

        if sid in processed:
            # Merge: update frequency and supporting conversations
            existing = ImprovementSuggestion(**processed[sid])
            merged_convs = list(set(
                existing.supporting_conversations + suggestion.supporting_conversations
            ))
            suggestion.frequency            = existing.frequency + suggestion.frequency
            suggestion.composite_score      = existing.composite_score + suggestion.composite_score
            suggestion.supporting_conversations = merged_convs

        processed[sid] = {
            "suggestion_id":            suggestion.suggestion_id,
            "category":                 suggestion.category.value,
            "agent_id":                 suggestion.agent_id,
            "title":                    suggestion.title,
            "faq_question":             suggestion.faq_question,
            "faq_answer":               suggestion.faq_answer,
            "rule_to_add":              suggestion.rule_to_add,
            "document_name":            suggestion.document_name,
            "document_content":         suggestion.document_content,
            "supporting_conversations": suggestion.supporting_conversations,
            "frequency":                suggestion.frequency,
            "composite_score":          suggestion.composite_score,
        }
        suggestions.append(suggestion)

    _save_processed(processed)

    suggestions.sort(key=lambda s: s.composite_score, reverse=True)
    logger.info("Synthesis complete | %d suggestions generated", len(suggestions))
    return suggestions


def build_batch_report(
    reports: list[ConversationReport],
    suggestions: list[ImprovementSuggestion],
    skipped: int = 0,
) -> BatchReport:
    return BatchReport(
        run_timestamp=datetime.now(timezone.utc).isoformat(),
        conversations_processed=len(reports),
        conversations_skipped=skipped,
        all_suggestions=suggestions,
    )
