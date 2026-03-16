from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field


class SpellRuleBase(BaseModel):
    wrong_form: str = Field(min_length=1)
    correct_form: str = Field(min_length=1)
    rule_type: Literal["substitution", "append_context"] = "substitution"
    enabled: bool = True


class SpellRuleCreate(SpellRuleBase):
    pass


class SpellRuleUpdate(BaseModel):
    wrong_form: str | None = None
    correct_form: str | None = None
    rule_type: Literal["substitution", "append_context"] | None = None
    enabled: bool | None = None


class SpellRuleImport(BaseModel):
    rules: list[SpellRuleCreate]


class SpellRuleResponse(SpellRuleBase):
    id: int
    created_at: datetime

    model_config = {"from_attributes": True}
