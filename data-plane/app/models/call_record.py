from datetime import datetime
from sqlalchemy import Integer, Text, DateTime, Float, func
from sqlalchemy.orm import Mapped, mapped_column
from app.database import Base


class CallRecord(Base):
    __tablename__ = "call_record"

    id: Mapped[str] = mapped_column(Text, primary_key=True)  # UUID v4
    session_token: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    assistant_config_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    state: Mapped[str] = mapped_column(Text, nullable=False, default="connecting")
    origin_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    user_agent: Mapped[str | None] = mapped_column(Text, nullable=True)
    transcript_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    recording_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    connected_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    duration_sec: Mapped[float | None] = mapped_column(Float, nullable=True)
    caller_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    ai_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    booking_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    booking_status: Mapped[str | None] = mapped_column(Text, nullable=True)
    end_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
