"""Calendly integration configuration endpoints."""
import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.calendly_config import CalendlyConfig
from app.models.calendly_event_type import CalendlyEventType
from app.schemas.calendly import (
    CalendlyConfigCreate,
    CalendlyConfigUpdate,
    CalendlyConfigResponse,
    CalendlyEventTypeCreate,
    CalendlyEventTypeUpdate,
    CalendlyEventTypeResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/calendly", tags=["calendly"])


# ── CalendlyConfig (singleton) ────────────────────────────────────────────────

@router.get("/config", response_model=CalendlyConfigResponse | None)
def get_calendly_config(db: Session = Depends(get_db)):
    return db.query(CalendlyConfig).first()


@router.put("/config", response_model=CalendlyConfigResponse)
def upsert_calendly_config(body: CalendlyConfigCreate, db: Session = Depends(get_db)):
    """Create or fully replace the Calendly configuration (singleton)."""
    config = db.query(CalendlyConfig).first()
    if config:
        for key, value in body.model_dump().items():
            setattr(config, key, value)
    else:
        config = CalendlyConfig(**body.model_dump())
        db.add(config)
    db.commit()
    db.refresh(config)
    logger.info("Calendly config saved | lookahead_days=%d timezone=%s", config.lookahead_days, config.timezone)
    return config


@router.patch("/config", response_model=CalendlyConfigResponse)
def patch_calendly_config(body: CalendlyConfigUpdate, db: Session = Depends(get_db)):
    config = db.query(CalendlyConfig).first()
    if not config:
        raise HTTPException(status_code=404, detail="No Calendly config found; use PUT to create one")
    for key, value in body.model_dump(exclude_none=True).items():
        setattr(config, key, value)
    db.commit()
    db.refresh(config)
    return config


# ── CalendlyEventType (list) ──────────────────────────────────────────────────

@router.get("/event-types", response_model=list[CalendlyEventTypeResponse])
def list_event_types(enabled: bool | None = None, db: Session = Depends(get_db)):
    q = db.query(CalendlyEventType)
    if enabled is not None:
        q = q.filter(CalendlyEventType.enabled == enabled)
    return q.all()


@router.post("/event-types", response_model=CalendlyEventTypeResponse, status_code=201)
def create_event_type(body: CalendlyEventTypeCreate, db: Session = Depends(get_db)):
    et = CalendlyEventType(**body.model_dump())
    db.add(et)
    db.commit()
    db.refresh(et)
    logger.info("Calendly event type added: %s (%s)", et.name, et.event_type_uri)
    return et


@router.patch("/event-types/{event_type_id}", response_model=CalendlyEventTypeResponse)
def update_event_type(event_type_id: int, body: CalendlyEventTypeUpdate, db: Session = Depends(get_db)):
    et = db.get(CalendlyEventType, event_type_id)
    if not et:
        raise HTTPException(status_code=404, detail="Event type not found")
    for key, value in body.model_dump(exclude_none=True).items():
        setattr(et, key, value)
    db.commit()
    db.refresh(et)
    return et


@router.delete("/event-types/{event_type_id}", status_code=204)
def delete_event_type(event_type_id: int, db: Session = Depends(get_db)):
    et = db.get(CalendlyEventType, event_type_id)
    if not et:
        raise HTTPException(status_code=404, detail="Event type not found")
    db.delete(et)
    db.commit()
    logger.info("Calendly event type deleted: id=%d", event_type_id)
