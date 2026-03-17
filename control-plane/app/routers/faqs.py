import logging
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.middleware.auth import get_current_user
from app.models.faq import FAQ
from app.schemas.faq import FAQCreate, FAQUpdate, FAQResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/faqs", tags=["faqs"])


@router.get("", response_model=list[FAQResponse])
def list_faqs(enabled: bool | None = Query(None), owner_id: str = Depends(get_current_user), db: Session = Depends(get_db)):
    q = db.query(FAQ).filter(FAQ.owner_id == owner_id)
    if enabled is not None:
        q = q.filter(FAQ.enabled == enabled)
    return q.all()


@router.post("", response_model=FAQResponse, status_code=201)
def create_faq(payload: FAQCreate, owner_id: str = Depends(get_current_user), db: Session = Depends(get_db)):
    faq = FAQ(owner_id=owner_id, **payload.model_dump())
    db.add(faq)
    db.commit()
    db.refresh(faq)
    logger.info("Created FAQ id=%d", faq.id)
    return faq


@router.get("/{faq_id}", response_model=FAQResponse)
def get_faq(faq_id: int, owner_id: str = Depends(get_current_user), db: Session = Depends(get_db)):
    faq = db.get(FAQ, faq_id)
    if not faq or faq.owner_id != owner_id:
        raise HTTPException(status_code=404, detail="FAQ not found")
    return faq


@router.put("/{faq_id}", response_model=FAQResponse)
def replace_faq(faq_id: int, payload: FAQCreate, owner_id: str = Depends(get_current_user), db: Session = Depends(get_db)):
    faq = db.get(FAQ, faq_id)
    if not faq or faq.owner_id != owner_id:
        raise HTTPException(status_code=404, detail="FAQ not found")
    for field, value in payload.model_dump().items():
        setattr(faq, field, value)
    db.commit()
    db.refresh(faq)
    return faq


@router.patch("/{faq_id}", response_model=FAQResponse)
def patch_faq(faq_id: int, payload: FAQUpdate, owner_id: str = Depends(get_current_user), db: Session = Depends(get_db)):
    faq = db.get(FAQ, faq_id)
    if not faq or faq.owner_id != owner_id:
        raise HTTPException(status_code=404, detail="FAQ not found")
    for field, value in payload.model_dump(exclude_none=True).items():
        setattr(faq, field, value)
    db.commit()
    db.refresh(faq)
    return faq


@router.delete("/{faq_id}", status_code=204)
def delete_faq(faq_id: int, owner_id: str = Depends(get_current_user), db: Session = Depends(get_db)):
    faq = db.get(FAQ, faq_id)
    if not faq or faq.owner_id != owner_id:
        raise HTTPException(status_code=404, detail="FAQ not found")
    db.delete(faq)
    db.commit()
