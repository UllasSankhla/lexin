from sqlalchemy import Integer, Text, Boolean
from sqlalchemy.orm import Mapped, mapped_column
from app.database import Base


class CalendlyEventType(Base):
    """A Calendly event type that can be matched to caller intent via LLM."""

    __tablename__ = "calendly_event_type"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # Human-readable name shown in the admin UI, e.g. "Initial Consultation"
    name: Mapped[str] = mapped_column(Text, nullable=False)
    # Optional description used as LLM matching context, e.g. "First-time meeting to discuss needs"
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Full Calendly event type URI — e.g. "https://api.calendly.com/event_types/{uuid}"
    event_type_uri: Mapped[str] = mapped_column(Text, nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
