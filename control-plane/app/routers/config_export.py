from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.middleware.auth import require_service_key
from app.services.config_export import build_config_export

router = APIRouter(prefix="/api/v1/config", tags=["config"])


@router.get("/export")
def export_config(
    owner_id: str = Query(...),
    db: Session = Depends(get_db),
    _: None = Depends(require_service_key),
):
    """Full configuration export consumed by the data plane at call start."""
    return build_config_export(db, owner_id)
