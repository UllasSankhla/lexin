"""Customer key management — admin CRUD + service lookup endpoint."""
import secrets
import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.middleware.auth import get_current_user, require_service_key
from app.models.customer_key import CustomerKey
from app.schemas.customer_key import CustomerKeyCreate, CustomerKeyUpdate, CustomerKeyResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/customer-keys", tags=["customer-keys"])


# ── Admin endpoints (Supabase JWT auth) ──────────────────────────────────────

@router.get("", response_model=list[CustomerKeyResponse])
def list_keys(owner_id: str = Depends(get_current_user), db: Session = Depends(get_db)):
    return db.query(CustomerKey).filter(CustomerKey.owner_id == owner_id).all()


@router.post("", response_model=CustomerKeyResponse, status_code=201)
def create_key(
    body: CustomerKeyCreate,
    owner_id: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    key_value = "ck_live_" + secrets.token_hex(20)
    ck = CustomerKey(owner_id=owner_id, key=key_value, name=body.name, enabled=body.enabled)
    db.add(ck)
    db.commit()
    db.refresh(ck)
    logger.info("Customer key created: %s for owner %s", ck.name, owner_id)
    return ck


@router.patch("/{key_id}", response_model=CustomerKeyResponse)
def update_key(
    key_id: int,
    body: CustomerKeyUpdate,
    owner_id: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    ck = db.get(CustomerKey, key_id)
    if not ck or ck.owner_id != owner_id:
        raise HTTPException(status_code=404, detail="Key not found")
    for field, value in body.model_dump(exclude_none=True).items():
        setattr(ck, field, value)
    db.commit()
    db.refresh(ck)
    return ck


@router.delete("/{key_id}", status_code=204)
def delete_key(
    key_id: int,
    owner_id: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    ck = db.get(CustomerKey, key_id)
    if not ck or ck.owner_id != owner_id:
        raise HTTPException(status_code=404, detail="Key not found")
    db.delete(ck)
    db.commit()
    logger.info("Customer key deleted: id=%d", key_id)


# ── Service endpoint (data plane → control plane, API key auth) ───────────────

@router.get("/lookup")
def lookup_key(
    key: str,
    _: None = Depends(require_service_key),
    db: Session = Depends(get_db),
):
    """Look up a customer key and return its owner_id. Used by the data plane."""
    ck = db.query(CustomerKey).filter(CustomerKey.key == key).first()
    if not ck:
        raise HTTPException(status_code=401, detail="Invalid customer key")
    if not ck.enabled:
        raise HTTPException(status_code=403, detail="Customer key is disabled")
    return {"owner_id": ck.owner_id, "name": ck.name}
