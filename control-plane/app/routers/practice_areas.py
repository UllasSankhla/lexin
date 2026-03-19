import logging
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.middleware.auth import get_current_user
from app.models.practice_area import PracticeArea
from app.schemas.practice_area import PracticeAreaCreate, PracticeAreaUpdate, PracticeAreaResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/practice-areas", tags=["practice-areas"])


@router.get("", response_model=list[PracticeAreaResponse])
def list_practice_areas(
    enabled: bool | None = Query(None),
    owner_id: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    q = db.query(PracticeArea).filter(PracticeArea.owner_id == owner_id)
    if enabled is not None:
        q = q.filter(PracticeArea.enabled == enabled)
    return q.order_by(PracticeArea.display_order).all()


@router.post("", response_model=PracticeAreaResponse, status_code=201)
def create_practice_area(
    payload: PracticeAreaCreate,
    owner_id: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    area = PracticeArea(owner_id=owner_id, **payload.model_dump())
    db.add(area)
    db.commit()
    db.refresh(area)
    logger.info("Created practice area '%s' (id=%d)", area.name, area.id)
    return area


@router.get("/{area_id}", response_model=PracticeAreaResponse)
def get_practice_area(
    area_id: int,
    owner_id: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    area = db.get(PracticeArea, area_id)
    if not area or area.owner_id != owner_id:
        raise HTTPException(status_code=404, detail="Practice area not found")
    return area


@router.patch("/{area_id}", response_model=PracticeAreaResponse)
def patch_practice_area(
    area_id: int,
    payload: PracticeAreaUpdate,
    owner_id: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    area = db.get(PracticeArea, area_id)
    if not area or area.owner_id != owner_id:
        raise HTTPException(status_code=404, detail="Practice area not found")
    for field, value in payload.model_dump(exclude_none=True).items():
        setattr(area, field, value)
    db.commit()
    db.refresh(area)
    return area


@router.delete("/{area_id}", status_code=204)
def delete_practice_area(
    area_id: int,
    owner_id: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    area = db.get(PracticeArea, area_id)
    if not area or area.owner_id != owner_id:
        raise HTTPException(status_code=404, detail="Practice area not found")
    db.delete(area)
    db.commit()
