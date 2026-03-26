"""Data models for the evaluator pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from pydantic import BaseModel


# ── Raw ingestion ─────────────────────────────────────────────────────────────

@dataclass
class Turn:
    speaker: str        # "caller" | "agent"
    text: str
    turn_index: int


@dataclass
class RawConversation:
    source_file: str
    file_hash: str
    raw_text: str
    turns: list[Turn]


@dataclass
class CallerTurn:
    text: str
    turn_index: int


@dataclass
class AgentTurn:
    text: str
    turn_index: int


@dataclass
class TestCase:
    conversation_id: str
    source_file: str
    file_hash: str
    caller_turns: list[CallerTurn]
    human_agent_turns: list[AgentTurn]
    # Paired (caller utterance, human agent response that followed)
    paired_exchanges: list[tuple[CallerTurn, AgentTurn]]


# ── Replay ────────────────────────────────────────────────────────────────────

@dataclass
class ReplayTurn:
    caller_utterance: str
    ai_response: str
    agent_id: str
    agent_status: str
    turn_index: int


@dataclass
class ReplayResult:
    conversation_id: str
    test_case: TestCase
    replay_turns: list[ReplayTurn]
    error: Optional[str] = None


# ── Evaluation ────────────────────────────────────────────────────────────────

class FindingCategory(str, Enum):
    FAQ     = "faq"
    PROMPT  = "prompt"
    CONTEXT = "context"
    NONE    = "none"


class EvaluationFinding(BaseModel):
    turn_index: int
    caller_utterance: str
    human_agent_response: str
    ai_response: str
    agent_id: str
    quality_score: int          # 1–5 (5 = equally good or better than human)
    gap_description: str
    category: FindingCategory
    severity: str               # "critical" | "moderate" | "minor"


# ── Suggestions ───────────────────────────────────────────────────────────────

class ImprovementSuggestion(BaseModel):
    suggestion_id: str
    category: FindingCategory
    agent_id: Optional[str] = None          # PROMPT only
    title: str
    faq_question: Optional[str] = None
    faq_answer: Optional[str] = None
    rule_to_add: Optional[str] = None
    document_name: Optional[str] = None
    document_content: Optional[str] = None
    supporting_conversations: list[str] = field(default_factory=list)
    frequency: int = 1
    composite_score: float = 0.0


# ── Reports ───────────────────────────────────────────────────────────────────

@dataclass
class ConversationReport:
    conversation_id: str
    source_file: str
    evaluated_at: str
    overall_quality_score: float
    findings: list[EvaluationFinding]
    suggestions: list[ImprovementSuggestion]
    # Full replay transcript — every caller turn and AI response, evaluated or not
    replay_turns: list[ReplayTurn] = field(default_factory=list)


@dataclass
class BatchReport:
    run_timestamp: str
    conversations_processed: int
    conversations_skipped: int
    all_suggestions: list[ImprovementSuggestion]
