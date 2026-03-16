from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.services.config_export import build_config_export

router = APIRouter(prefix="/api/v1/config", tags=["config"])


@router.get("/export")
def export_config(db: Session = Depends(get_db)):
    """Full configuration export consumed by the data plane at call start."""
    return build_config_export(db)
