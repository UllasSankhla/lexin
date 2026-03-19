from datetime import datetime
from pydantic import BaseModel


class PolicyDocumentResponse(BaseModel):
    id: int
    practice_area_id: int | None
    filename: str
    original_name: str
    mime_type: str
    size_bytes: int
    description: str | None
    enabled: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class PolicyDocumentUpdate(BaseModel):
    practice_area_id: int | None = None
    description: str | None = None
    enabled: bool | None = None
