"""Pydantic schemas for all agent LLM responses (except data_collection).

All schemas are used with llm_structured_call() which passes them to Cerebras
via response_format json_schema strict mode — guaranteeing valid structured JSON.
"""
from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel


# ── FAQ ───────────────────────────────────────────────────────────────────────

class LegalDeflectSignal(BaseModel):
    is_legal_question: bool


class FAQMatchResult(BaseModel):
    matched: bool
    index: Optional[int] = None


class FAQQuestionMatch(BaseModel):
    """Single question extracted from a multi-question utterance."""
    is_legal: bool
    faq_index: Optional[int] = None  # None if no FAQ match or is_legal=True


class FAQMultiMatchResult(BaseModel):
    """All questions detected in a caller utterance with per-question match results."""
    questions: list[FAQQuestionMatch]


# ── Scheduling ────────────────────────────────────────────────────────────────

class SlotConfirmSignal(BaseModel):
    intent: Literal["confirm", "reject", "new_slot"]


class EventTypeMatch(BaseModel):
    index: Optional[int] = None


class SlotChoice(BaseModel):
    slot: Optional[int] = None


class DateRangePreference(BaseModel):
    found: bool = True
    start_time: Optional[str] = None
    end_time: Optional[str] = None


# ── Context docs ──────────────────────────────────────────────────────────────

class ContextDocsResult(BaseModel):
    found: bool
    answer: Optional[str] = None


# ── Narrative collection ──────────────────────────────────────────────────────

class DoneIntentSignal(BaseModel):
    done: bool


# ── Intake qualification ──────────────────────────────────────────────────────

class QualificationResult(BaseModel):
    decision: Literal["qualified", "ambiguous", "not_qualified"]
    matched_area: Optional[str] = None
    reason: str = ""


# ── Router ────────────────────────────────────────────────────────────────────

class RouterDecision(BaseModel):
    agent_id: str
    interrupt: bool = False
    reasoning: str = ""


# ── Planner ───────────────────────────────────────────────────────────────────

class IntentItem(BaseModel):
    """A single detected intent within the caller's utterance."""
    type: Literal[
        "FIELD_DATA",
        "CONFIRMATION",
        "CORRECTION",
        "NARRATIVE",
        "FAQ_QUESTION",
        "DATA_STATUS",
        "CONTINUATION",
        "FAREWELL",
    ]
    field: Optional[str] = None  # CORRECTION only: exact key from COLLECTED STATE
    reason: str = ""             # brief LLM reasoning for this intent


class MultiIntentLLMResponse(BaseModel):
    thinking: str
    intents: list[IntentItem]    # ordered by speech position, max 3
