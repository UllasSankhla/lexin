import json
import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.parameter import CollectionParameter
from app.schemas.parameter import (
    CollectionParameterCreate, CollectionParameterUpdate,
    CollectionParameterResponse, CollectionParameterReorder,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/parameters", tags=["parameters"])


@router.get("", response_model=list[CollectionParameterResponse])
def list_parameters(db: Session = Depends(get_db)):
    return db.query(CollectionParameter).order_by(CollectionParameter.collection_order).all()


@router.post("", response_model=CollectionParameterResponse, status_code=201)
def create_parameter(payload: CollectionParameterCreate, db: Session = Depends(get_db)):
    existing = db.query(CollectionParameter).filter_by(name=payload.name).first()
    if existing:
        raise HTTPException(status_code=409, detail=f"Parameter '{payload.name}' already exists")
    data = payload.model_dump()
    if data.get("extraction_hints"):
        data["extraction_hints"] = json.dumps(data["extraction_hints"])
    param = CollectionParameter(**data)
    db.add(param)
    db.commit()
    db.refresh(param)
    logger.info("Created parameter '%s' (id=%d)", param.name, param.id)
    return param


@router.get("/{param_id}", response_model=CollectionParameterResponse)
def get_parameter(param_id: int, db: Session = Depends(get_db)):
    param = db.get(CollectionParameter, param_id)
    if not param:
        raise HTTPException(status_code=404, detail="Parameter not found")
    return param


@router.put("/{param_id}", response_model=CollectionParameterResponse)
def replace_parameter(param_id: int, payload: CollectionParameterCreate, db: Session = Depends(get_db)):
    param = db.get(CollectionParameter, param_id)
    if not param:
        raise HTTPException(status_code=404, detail="Parameter not found")
    data = payload.model_dump()
    if data.get("extraction_hints"):
        data["extraction_hints"] = json.dumps(data["extraction_hints"])
    for field, value in data.items():
        setattr(param, field, value)
    db.commit()
    db.refresh(param)
    return param


@router.patch("/{param_id}", response_model=CollectionParameterResponse)
def patch_parameter(param_id: int, payload: CollectionParameterUpdate, db: Session = Depends(get_db)):
    param = db.get(CollectionParameter, param_id)
    if not param:
        raise HTTPException(status_code=404, detail="Parameter not found")
    data = payload.model_dump(exclude_none=True)
    if "extraction_hints" in data:
        data["extraction_hints"] = json.dumps(data["extraction_hints"])
    for field, value in data.items():
        setattr(param, field, value)
    db.commit()
    db.refresh(param)
    return param


@router.delete("/{param_id}", status_code=204)
def delete_parameter(param_id: int, db: Session = Depends(get_db)):
    param = db.get(CollectionParameter, param_id)
    if not param:
        raise HTTPException(status_code=404, detail="Parameter not found")
    db.delete(param)
    db.commit()


@router.post("/reorder", status_code=200)
def reorder_parameters(payload: CollectionParameterReorder, db: Session = Depends(get_db)):
    for item in payload.items:
        param = db.get(CollectionParameter, item["id"])
        if param:
            param.collection_order = item["collection_order"]
    db.commit()
    return {"message": "Reordered successfully"}
