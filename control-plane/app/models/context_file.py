from datetime import datetime
from sqlalchemy import Integer, String, Text, DateTime, Boolean, func
from sqlalchemy.orm import Mapped, mapped_column
from app.database import Base


class ContextFile(Base):
    __tablename__ = "context_file"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    owner_id: Mapped[str] = mapped_column(String(64), nullable=False, default="", index=True)
    filename: Mapped[str] = mapped_column(Text, nullable=False)  # UUID-based stored name
    original_name: Mapped[str] = mapped_column(Text, nullable=False)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    mime_type: Mapped[str] = mapped_column(Text, nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
