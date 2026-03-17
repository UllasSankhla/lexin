from pydantic import BaseModel
from datetime import datetime


class CustomerKeyCreate(BaseModel):
    name: str
    enabled: bool = True


class CustomerKeyUpdate(BaseModel):
    name: str | None = None
    enabled: bool | None = None


class CustomerKeyResponse(BaseModel):
    id: int
    key: str
    name: str
    enabled: bool
    created_at: datetime

    class Config:
        from_attributes = True
