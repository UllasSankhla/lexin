from datetime import datetime
from pydantic import BaseModel


class ContextFileResponse(BaseModel):
    id: int
    filename: str
    original_name: str
    mime_type: str
    size_bytes: int
    description: str | None
    enabled: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class ContextFileUpdate(BaseModel):
    description: str | None = None
    enabled: bool | None = None
