from sqlalchemy import Integer, Text, Boolean
from sqlalchemy.orm import Mapped, mapped_column
from app.database import Base


class CalendlyConfig(Base):
    """Singleton Calendly integration settings."""

    __tablename__ = "calendly_config"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # Calendly Personal Access Token or OAuth token
    api_token: Mapped[str] = mapped_column(Text, nullable=False)
    # Full Calendly user URI — e.g. "https://api.calendly.com/users/{uuid}"
    user_uri: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Full Calendly organization URI — e.g. "https://api.calendly.com/organizations/{uuid}"
    organization_uri: Mapped[str | None] = mapped_column(Text, nullable=True)
    # How many days ahead to search for open slots (min 1, sensible default 30)
    lookahead_days: Mapped[int] = mapped_column(Integer, nullable=False, default=30)
    # IANA timezone name used when formatting slot descriptions for voice (e.g. "America/New_York")
    timezone: Mapped[str] = mapped_column(Text, nullable=False, default="UTC")
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
