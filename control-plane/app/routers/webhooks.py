import json
import logging
import httpx
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.middleware.auth import get_current_user
from app.models.webhook import WebhookEndpoint
from app.schemas.webhook import WebhookEndpointCreate, WebhookEndpointUpdate, WebhookEndpointResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/webhooks", tags=["webhooks"])


@router.get("", response_model=list[WebhookEndpointResponse])
def list_webhooks(owner_id: str = Depends(get_current_user), db: Session = Depends(get_db)):
    return db.query(WebhookEndpoint).filter(WebhookEndpoint.owner_id == owner_id).all()


@router.post("", response_model=WebhookEndpointResponse, status_code=201)
def create_webhook(payload: WebhookEndpointCreate, owner_id: str = Depends(get_current_user), db: Session = Depends(get_db)):
    data = payload.model_dump()
    data["events"] = json.dumps(data["events"])
    webhook = WebhookEndpoint(owner_id=owner_id, **data)
    db.add(webhook)
    db.commit()
    db.refresh(webhook)
    logger.info("Created webhook '%s' (id=%d)", webhook.name, webhook.id)
    return webhook


@router.get("/{webhook_id}", response_model=WebhookEndpointResponse)
def get_webhook(webhook_id: int, owner_id: str = Depends(get_current_user), db: Session = Depends(get_db)):
    webhook = db.get(WebhookEndpoint, webhook_id)
    if not webhook or webhook.owner_id != owner_id:
        raise HTTPException(status_code=404, detail="Webhook not found")
    return webhook


@router.put("/{webhook_id}", response_model=WebhookEndpointResponse)
def replace_webhook(webhook_id: int, payload: WebhookEndpointCreate, owner_id: str = Depends(get_current_user), db: Session = Depends(get_db)):
    webhook = db.get(WebhookEndpoint, webhook_id)
    if not webhook or webhook.owner_id != owner_id:
        raise HTTPException(status_code=404, detail="Webhook not found")
    data = payload.model_dump()
    data["events"] = json.dumps(data["events"])
    for field, value in data.items():
        setattr(webhook, field, value)
    db.commit()
    db.refresh(webhook)
    return webhook


@router.patch("/{webhook_id}", response_model=WebhookEndpointResponse)
def patch_webhook(webhook_id: int, payload: WebhookEndpointUpdate, owner_id: str = Depends(get_current_user), db: Session = Depends(get_db)):
    webhook = db.get(WebhookEndpoint, webhook_id)
    if not webhook or webhook.owner_id != owner_id:
        raise HTTPException(status_code=404, detail="Webhook not found")
    data = payload.model_dump(exclude_none=True)
    if "events" in data:
        data["events"] = json.dumps(data["events"])
    for field, value in data.items():
        setattr(webhook, field, value)
    db.commit()
    db.refresh(webhook)
    return webhook


@router.delete("/{webhook_id}", status_code=204)
def delete_webhook(webhook_id: int, owner_id: str = Depends(get_current_user), db: Session = Depends(get_db)):
    webhook = db.get(WebhookEndpoint, webhook_id)
    if not webhook or webhook.owner_id != owner_id:
        raise HTTPException(status_code=404, detail="Webhook not found")
    db.delete(webhook)
    db.commit()


@router.post("/{webhook_id}/test")
async def test_webhook(webhook_id: int, owner_id: str = Depends(get_current_user), db: Session = Depends(get_db)):
    webhook = db.get(WebhookEndpoint, webhook_id)
    if not webhook or webhook.owner_id != owner_id:
        raise HTTPException(status_code=404, detail="Webhook not found")

    test_payload = {
        "event": "webhook.test",
        "call_id": "test-call-id",
        "timestamp": "2026-01-01T00:00:00Z",
        "message": "This is a test webhook from the control plane.",
    }
    headers = {"Content-Type": "application/json"}
    if webhook.secret_header and webhook.secret_value:
        headers[webhook.secret_header] = webhook.secret_value

    try:
        async with httpx.AsyncClient(timeout=webhook.timeout_sec) as client:
            resp = await client.post(webhook.url, json=test_payload, headers=headers)
        return {"status": resp.status_code, "response": resp.text[:500]}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Webhook test failed: {e}")
