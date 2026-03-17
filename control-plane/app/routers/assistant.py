import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.middleware.auth import get_current_user
from app.models.assistant import AssistantConfig
from app.schemas.assistant import AssistantConfigCreate, AssistantConfigUpdate, AssistantConfigResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/assistant", tags=["assistant"])


@router.get("", response_model=AssistantConfigResponse)
def get_assistant(owner_id: str = Depends(get_current_user), db: Session = Depends(get_db)):
    config = db.query(AssistantConfig).filter(AssistantConfig.owner_id == owner_id).first()
    if not config:
        raise HTTPException(status_code=404, detail="No assistant configuration found")
    return config


@router.put("", response_model=AssistantConfigResponse)
def upsert_assistant(payload: AssistantConfigCreate, owner_id: str = Depends(get_current_user), db: Session = Depends(get_db)):
    config = db.query(AssistantConfig).filter(AssistantConfig.owner_id == owner_id).first()
    if config:
        for field, value in payload.model_dump().items():
            setattr(config, field, value)
    else:
        config = AssistantConfig(owner_id=owner_id, **payload.model_dump())
        db.add(config)
    db.commit()
    db.refresh(config)
    logger.info("Assistant config upserted (id=%d)", config.id)
    return config


@router.patch("", response_model=AssistantConfigResponse)
def patch_assistant(payload: AssistantConfigUpdate, owner_id: str = Depends(get_current_user), db: Session = Depends(get_db)):
    config = db.query(AssistantConfig).filter(AssistantConfig.owner_id == owner_id).first()
    if not config:
        raise HTTPException(status_code=404, detail="No assistant configuration found")
    for field, value in payload.model_dump(exclude_none=True).items():
        setattr(config, field, value)
    db.commit()
    db.refresh(config)
    return config
