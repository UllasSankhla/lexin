import logging
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from pathlib import Path

from app.database import get_db
from app.models.context_file import ContextFile
from app.schemas.context_file import ContextFileResponse, ContextFileUpdate
from app.services.file_storage import save_context_file, delete_context_file

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/context-files", tags=["context-files"])


@router.get("", response_model=list[ContextFileResponse])
def list_context_files(db: Session = Depends(get_db)):
    return db.query(ContextFile).all()


@router.post("", response_model=ContextFileResponse, status_code=201)
async def upload_context_file(
    file: UploadFile = File(...),
    description: str | None = None,
    db: Session = Depends(get_db),
):
    metadata = await save_context_file(file)
    cf = ContextFile(**metadata, description=description)
    db.add(cf)
    db.commit()
    db.refresh(cf)
    logger.info("Uploaded context file '%s' (id=%d)", cf.original_name, cf.id)
    return cf


@router.get("/{file_id}", response_model=ContextFileResponse)
def get_context_file(file_id: int, db: Session = Depends(get_db)):
    cf = db.get(ContextFile, file_id)
    if not cf:
        raise HTTPException(status_code=404, detail="Context file not found")
    return cf


@router.get("/{file_id}/download")
def download_context_file(file_id: int, db: Session = Depends(get_db)):
    cf = db.get(ContextFile, file_id)
    if not cf:
        raise HTTPException(status_code=404, detail="Context file not found")
    path = Path(cf.file_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found on disk")
    return FileResponse(path, filename=cf.original_name, media_type=cf.mime_type)


@router.patch("/{file_id}", response_model=ContextFileResponse)
def update_context_file(file_id: int, payload: ContextFileUpdate, db: Session = Depends(get_db)):
    cf = db.get(ContextFile, file_id)
    if not cf:
        raise HTTPException(status_code=404, detail="Context file not found")
    for field, value in payload.model_dump(exclude_none=True).items():
        setattr(cf, field, value)
    db.commit()
    db.refresh(cf)
    return cf


@router.delete("/{file_id}", status_code=204)
def delete_context_file_endpoint(file_id: int, db: Session = Depends(get_db)):
    cf = db.get(ContextFile, file_id)
    if not cf:
        raise HTTPException(status_code=404, detail="Context file not found")
    file_path = cf.file_path
    db.delete(cf)
    db.commit()
    delete_context_file(file_path)
