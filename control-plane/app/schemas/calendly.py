from pydantic import BaseModel, Field, HttpUrl


# ── CalendlyConfig schemas ────────────────────────────────────────────────────

class CalendlyConfigBase(BaseModel):
    api_token: str = Field(min_length=1)
    user_uri: str | None = None
    organization_uri: str | None = None
    lookahead_days: int = Field(default=30, ge=1, le=365)
    timezone: str = Field(default="UTC")
    enabled: bool = True


class CalendlyConfigCreate(CalendlyConfigBase):
    pass


class CalendlyConfigUpdate(BaseModel):
    api_token: str | None = None
    user_uri: str | None = None
    organization_uri: str | None = None
    lookahead_days: int | None = Field(default=None, ge=1, le=365)
    timezone: str | None = None
    enabled: bool | None = None


class CalendlyConfigResponse(CalendlyConfigBase):
    id: int
    model_config = {"from_attributes": True}


# ── CalendlyEventType schemas ─────────────────────────────────────────────────

class CalendlyEventTypeBase(BaseModel):
    name: str = Field(min_length=1)
    description: str | None = None
    event_type_uri: str = Field(min_length=1)
    enabled: bool = True


class CalendlyEventTypeCreate(CalendlyEventTypeBase):
    pass


class CalendlyEventTypeUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    event_type_uri: str | None = None
    enabled: bool | None = None


class CalendlyEventTypeResponse(CalendlyEventTypeBase):
    id: int
    model_config = {"from_attributes": True}
