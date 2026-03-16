from datetime import datetime
from sqlalchemy import Integer, Text, DateTime, Float, ForeignKey, func
from sqlalchemy.orm import Mapped, mapped_column
from app.database import Base


class CallAnalytics(Base):
    __tablename__ = "call_analytics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    call_id: Mapped[str] = mapped_column(Text, ForeignKey("call_record.id"), nullable=False)
    event_name: Mapped[str] = mapped_column(Text, nullable=False)
    stage: Mapped[str] = mapped_column(Text, nullable=False)  # stt|llm|tts|total
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    recorded_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
