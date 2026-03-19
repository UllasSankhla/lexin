from datetime import datetime
from pydantic import BaseModel, Field


class PracticeAreaBase(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str | None = None
    qualification_criteria: str | None = None
    disqualification_signals: str | None = None
    ambiguous_signals: str | None = None
    referral_suggestion: str | None = None
    enabled: bool = True
    display_order: int = Field(default=0, ge=0)


class PracticeAreaCreate(PracticeAreaBase):
    pass


class PracticeAreaUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    qualification_criteria: str | None = None
    disqualification_signals: str | None = None
    ambiguous_signals: str | None = None
    referral_suggestion: str | None = None
    enabled: bool | None = None
    display_order: int | None = None


class PracticeAreaResponse(PracticeAreaBase):
    id: int
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
