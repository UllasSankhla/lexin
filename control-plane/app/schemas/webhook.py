import json
from datetime import datetime
from pydantic import BaseModel, Field, HttpUrl, field_validator


class WebhookEndpointBase(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    url: str
    secret_header: str | None = None
    secret_value: str | None = None
    events: list[str] = ["call.completed"]
    enabled: bool = True
    timeout_sec: int = Field(default=10, ge=1, le=60)
    retry_count: int = Field(default=3, ge=0, le=10)


class WebhookEndpointCreate(WebhookEndpointBase):
    pass


class WebhookEndpointUpdate(BaseModel):
    name: str | None = None
    url: str | None = None
    secret_header: str | None = None
    secret_value: str | None = None
    events: list[str] | None = None
    enabled: bool | None = None
    timeout_sec: int | None = None
    retry_count: int | None = None


class WebhookEndpointResponse(WebhookEndpointBase):
    id: int
    secret_value: str | None = None  # masked in responses
    created_at: datetime

    model_config = {"from_attributes": True}

    @field_validator("events", mode="before")
    @classmethod
    def parse_events(cls, v):
        if isinstance(v, str):
            return json.loads(v)
        return v

    @field_validator("secret_value", mode="before")
    @classmethod
    def mask_secret(cls, v):
        if v:
            return "***"
        return v
