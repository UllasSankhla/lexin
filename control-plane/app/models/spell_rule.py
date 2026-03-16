from datetime import datetime
from sqlalchemy import Integer, Text, DateTime, Boolean, func
from sqlalchemy.orm import Mapped, mapped_column
from app.database import Base


class SpellRule(Base):
    __tablename__ = "spell_rule"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    wrong_form: Mapped[str] = mapped_column(Text, nullable=False)
    correct_form: Mapped[str] = mapped_column(Text, nullable=False)
    rule_type: Mapped[str] = mapped_column(Text, nullable=False, default="substitution")
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
