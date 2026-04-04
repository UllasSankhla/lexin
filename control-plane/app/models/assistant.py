from datetime import datetime
from sqlalchemy import Integer, String, Text, DateTime, Boolean, func
from sqlalchemy.orm import Mapped, mapped_column
from app.database import Base


class AssistantConfig(Base):
    __tablename__ = "assistant_config"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    owner_id: Mapped[str] = mapped_column(String(64), nullable=False, default="", index=True)
    persona_name: Mapped[str] = mapped_column(Text, nullable=False, default="Aria")
    persona_voice: Mapped[str] = mapped_column(Text, nullable=False, default="aura-2-thalia-en")
    nature: Mapped[str] = mapped_column(Text, nullable=False, default="friendly")
    system_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    system_prompt_file: Mapped[str | None] = mapped_column(Text, nullable=True)
    greeting_message: Mapped[str] = mapped_column(Text, nullable=False)
    max_call_duration_sec: Mapped[int] = mapped_column(Integer, nullable=False, default=600)
    silence_timeout_sec: Mapped[int] = mapped_column(Integer, nullable=False, default=10)
    language: Mapped[str] = mapped_column(Text, nullable=False, default="en-US")
    enable_empathy_fillers: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())
