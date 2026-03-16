from datetime import datetime
from sqlalchemy import Integer, Text, DateTime, Boolean, func
from sqlalchemy.orm import Mapped, mapped_column
from app.database import Base


class WebhookEndpoint(Base):
    __tablename__ = "webhook_endpoint"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    secret_header: Mapped[str | None] = mapped_column(Text, nullable=True)
    secret_value: Mapped[str | None] = mapped_column(Text, nullable=True)
    events: Mapped[str] = mapped_column(Text, nullable=False, default='["call.completed"]')
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    timeout_sec: Mapped[int] = mapped_column(Integer, nullable=False, default=10)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
