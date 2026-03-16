from datetime import datetime
from pydantic import BaseModel, Field


class FAQBase(BaseModel):
    question: str = Field(min_length=5)
    answer: str = Field(min_length=5)
    category: str | None = None
    enabled: bool = True


class FAQCreate(FAQBase):
    pass


class FAQUpdate(BaseModel):
    question: str | None = None
    answer: str | None = None
    category: str | None = None
    enabled: bool | None = None


class FAQResponse(FAQBase):
    id: int
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
