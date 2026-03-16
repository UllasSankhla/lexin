from datetime import datetime
from pydantic import BaseModel, Field


class AssistantConfigBase(BaseModel):
    persona_name: str = Field(default="Aria", max_length=100)
    persona_voice: str = Field(default="aura-2-thalia-en")
    nature: str = Field(default="friendly", max_length=50)
    system_prompt: str = Field(min_length=10)
    system_prompt_file: str | None = None
    greeting_message: str = Field(min_length=5)
    max_call_duration_sec: int = Field(default=600, ge=60, le=3600)
    silence_timeout_sec: int = Field(default=10, ge=3, le=60)
    language: str = Field(default="en-US")


class AssistantConfigCreate(AssistantConfigBase):
    pass


class AssistantConfigUpdate(BaseModel):
    persona_name: str | None = None
    persona_voice: str | None = None
    nature: str | None = None
    system_prompt: str | None = None
    system_prompt_file: str | None = None
    greeting_message: str | None = None
    max_call_duration_sec: int | None = None
    silence_timeout_sec: int | None = None
    language: str | None = None


class AssistantConfigResponse(AssistantConfigBase):
    id: int
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
