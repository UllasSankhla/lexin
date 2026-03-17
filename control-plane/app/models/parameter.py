from datetime import datetime
from sqlalchemy import Integer, String, Text, DateTime, Boolean, func
from sqlalchemy.orm import Mapped, mapped_column
from app.database import Base


class CollectionParameter(Base):
    __tablename__ = "collection_parameter"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    owner_id: Mapped[str] = mapped_column(String(64), nullable=False, default="", index=True)
    name: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    display_label: Mapped[str] = mapped_column(Text, nullable=False)
    data_type: Mapped[str] = mapped_column(Text, nullable=False, default="text")
    required: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    collection_order: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    validation_regex: Mapped[str | None] = mapped_column(Text, nullable=True)
    validation_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    extraction_hints: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON list
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())
