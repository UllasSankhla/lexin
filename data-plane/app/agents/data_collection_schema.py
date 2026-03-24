"""Pydantic schema for the mega-prompt data collection LLM response."""
from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field


class PendingConfirmation(BaseModel):
    field: str
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
    extracted: dict[str, str] = Field(default_factory=dict)
    correction_value: Optional[str] = None
    speak: str = ""
    status: Literal["in_progress", "waiting_confirm", "completed", "unhandled"]
    pending_confirmation: Optional[PendingConfirmation] = None
    incomplete_utterance: bool = False
    cannot_process: bool = False
    cannot_process_reason: Optional[str] = None
