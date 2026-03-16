import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.spell_rule import SpellRule
from app.schemas.spell_rule import SpellRuleCreate, SpellRuleUpdate, SpellRuleImport, SpellRuleResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/spell-rules", tags=["spell-rules"])


@router.get("", response_model=list[SpellRuleResponse])
def list_spell_rules(db: Session = Depends(get_db)):
    return db.query(SpellRule).all()


@router.post("", response_model=SpellRuleResponse, status_code=201)
def create_spell_rule(payload: SpellRuleCreate, db: Session = Depends(get_db)):
    rule = SpellRule(**payload.model_dump())
    db.add(rule)
    db.commit()
    db.refresh(rule)
    return rule


@router.get("/{rule_id}", response_model=SpellRuleResponse)
def get_spell_rule(rule_id: int, db: Session = Depends(get_db)):
    rule = db.get(SpellRule, rule_id)
    if not rule:
        raise HTTPException(status_code=404, detail="Spell rule not found")
    return rule


@router.put("/{rule_id}", response_model=SpellRuleResponse)
def replace_spell_rule(rule_id: int, payload: SpellRuleCreate, db: Session = Depends(get_db)):
    rule = db.get(SpellRule, rule_id)
    if not rule:
        raise HTTPException(status_code=404, detail="Spell rule not found")
    for field, value in payload.model_dump().items():
        setattr(rule, field, value)
    db.commit()
    db.refresh(rule)
    return rule


@router.delete("/{rule_id}", status_code=204)
def delete_spell_rule(rule_id: int, db: Session = Depends(get_db)):
    rule = db.get(SpellRule, rule_id)
    if not rule:
        raise HTTPException(status_code=404, detail="Spell rule not found")
    db.delete(rule)
    db.commit()


@router.post("/import", response_model=list[SpellRuleResponse], status_code=201)
def import_spell_rules(payload: SpellRuleImport, db: Session = Depends(get_db)):
    created = []
    for rule_data in payload.rules:
        rule = SpellRule(**rule_data.model_dump())
        db.add(rule)
        created.append(rule)
    db.commit()
    for rule in created:
        db.refresh(rule)
    logger.info("Imported %d spell rules", len(created))
    return created
