import logging
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from pathlib import Path

from app.database import get_db
from app.middleware.auth import get_current_user
from app.models.policy_document import PolicyDocument
from app.schemas.policy_document import PolicyDocumentResponse, PolicyDocumentUpdate
from app.services.file_storage import save_policy_document, delete_policy_document

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/policy-documents", tags=["policy-documents"])


@router.get("", response_model=list[PolicyDocumentResponse])
def list_policy_documents(
    practice_area_id: int | None = Query(None),
    owner_id: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    q = db.query(PolicyDocument).filter(PolicyDocument.owner_id == owner_id)
    if practice_area_id is not None:
        q = q.filter(PolicyDocument.practice_area_id == practice_area_id)
    return q.all()


@router.post("", response_model=PolicyDocumentResponse, status_code=201)
async def upload_policy_document(
    file: UploadFile = File(...),
    practice_area_id: int | None = Query(None),
    description: str | None = None,
    owner_id: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    metadata = await save_policy_document(file)
    pd = PolicyDocument(
        owner_id=owner_id,
        practice_area_id=practice_area_id,
        description=description,
        **metadata,
    )
    db.add(pd)
    db.commit()
    db.refresh(pd)
    logger.info("Uploaded policy document '%s' (id=%d)", pd.original_name, pd.id)
    return pd


@router.get("/{doc_id}", response_model=PolicyDocumentResponse)
def get_policy_document(
    doc_id: int,
    owner_id: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    pd = db.get(PolicyDocument, doc_id)
    if not pd or pd.owner_id != owner_id:
        raise HTTPException(status_code=404, detail="Policy document not found")
    return pd


@router.get("/{doc_id}/download")
def download_policy_document(
    doc_id: int,
    owner_id: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    pd = db.get(PolicyDocument, doc_id)
    if not pd or pd.owner_id != owner_id:
        raise HTTPException(status_code=404, detail="Policy document not found")
    path = Path(pd.file_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found on disk")
    return FileResponse(path, filename=pd.original_name, media_type=pd.mime_type)


@router.patch("/{doc_id}", response_model=PolicyDocumentResponse)
def update_policy_document(
    doc_id: int,
    payload: PolicyDocumentUpdate,
    owner_id: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    pd = db.get(PolicyDocument, doc_id)
    if not pd or pd.owner_id != owner_id:
        raise HTTPException(status_code=404, detail="Policy document not found")
    for field, value in payload.model_dump(exclude_none=True).items():
        setattr(pd, field, value)
    db.commit()
    db.refresh(pd)
    return pd


@router.delete("/{doc_id}", status_code=204)
def delete_policy_document_endpoint(
    doc_id: int,
    owner_id: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    pd = db.get(PolicyDocument, doc_id)
    if not pd or pd.owner_id != owner_id:
        raise HTTPException(status_code=404, detail="Policy document not found")
    file_path = pd.file_path
    db.delete(pd)
    db.commit()
    delete_policy_document(file_path)
