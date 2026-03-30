"""Pydantic schema for the mega-prompt data collection LLM response."""
from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field


class PendingConfirmation(BaseModel):
    field: str
    value: str


class ConfirmationSignal(BaseModel):
    """Result of the lightweight confirmation classifier (used before mega-prompt)."""
    signal: Literal["confirm", "reject", "correct_or_add", "unrelated"]
    # confirm        — pure yes/agreement, no new field data
    # reject         — pure no/disagreement, no new field data
    # correct_or_add — contains new field data (digits, email, name, etc.)
    #                  caller may also be agreeing or correcting
    # unrelated      — question, off-topic remark, unclear — not a yes/no/correction
    is_affirmative: bool = True  # for correct_or_add: was the caller also confirming?


class ExtractedField(BaseModel):
    """A single extracted field name + value pair (used instead of dict[str, str]
    because Cerebras strict mode rejects open-ended additionalProperties objects)."""
    key: str
    value: str


class DataCollectionLLMResponse(BaseModel):
    intent: Literal[
        "answer",
        "confirm_yes",
        "confirm_no",
        "correction",
        "skip",
        "off_topic",
        "incomplete_utterance",
    ]
    extracted: list[ExtractedField] = Field(default_factory=list)
    correction_value: Optional[str] = None
    speak: str = ""
    status: Literal["in_progress", "waiting_confirm", "completed", "unhandled"]
    pending_confirmation: Optional[PendingConfirmation] = None
    incomplete_utterance: bool = False
    cannot_process: bool = False
    cannot_process_reason: Optional[str] = None
