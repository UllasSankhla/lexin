import re
import json
from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field, field_validator


DATA_TYPES = Literal["text", "email", "phone", "date", "number", "address", "insurance"]


class CollectionParameterBase(BaseModel):
    name: str = Field(min_length=1, max_length=100, pattern=r"^[a-z][a-z0-9_]*$")
    display_label: str = Field(min_length=1, max_length=200)
    data_type: DATA_TYPES = "text"
    required: bool = True
    collection_order: int = Field(default=0, ge=0)
    validation_regex: str | None = None
    validation_message: str | None = None
    extraction_hints: list[str] | None = None

    @field_validator("validation_regex")
    @classmethod
    def validate_regex(cls, v):
        if v is not None:
            try:
                re.compile(v)
            except re.error as e:
                raise ValueError(f"Invalid regex pattern: {e}") from e
        return v


class CollectionParameterCreate(CollectionParameterBase):
    pass


class CollectionParameterUpdate(BaseModel):
    display_label: str | None = None
    data_type: DATA_TYPES | None = None
    required: bool | None = None
    collection_order: int | None = None
    validation_regex: str | None = None
    validation_message: str | None = None
    extraction_hints: list[str] | None = None


class CollectionParameterReorder(BaseModel):
    items: list[dict]  # [{"id": 1, "collection_order": 0}, ...]


class CollectionParameterResponse(CollectionParameterBase):
    id: int
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}

    @field_validator("extraction_hints", mode="before")
    @classmethod
    def parse_hints(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except Exception:
                return None
        return v
