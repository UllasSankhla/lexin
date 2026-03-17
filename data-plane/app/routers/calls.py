"""Call management REST endpoints."""
import logging
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.middleware.customer_auth import require_customer_key
from app.models.call_record import CallRecord
from app.models.gathered_parameter import GatheredParameter
from app.models.call_analytics import CallAnalytics
from app.services.config_client import fetch_config_from_control_plane, set_cached_config

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/calls", tags=["calls"])


def _format_call(record: CallRecord) -> dict:
    """Serialize a CallRecord to a JSON-safe dict."""
    return {
        "id": record.id,
        "state": record.state,
        "caller_name": record.caller_name or "Unknown Caller",
        "ai_summary": record.ai_summary or "",
        "origin_url": record.origin_url,
        "duration_sec": record.duration_sec,
        "started_at": record.started_at.isoformat() if record.started_at else None,
        "connected_at": record.connected_at.isoformat() if record.connected_at else None,
        "completed_at": record.completed_at.isoformat() if record.completed_at else None,
        "has_transcript": bool(record.transcript_path),
        "error_message": record.error_message,
        "assistant_config_id": record.assistant_config_id,
    }


@router.post("/initiate")
async def initiate_call(
    request: Request,
    db: Session = Depends(get_db),
    customer_auth: tuple[str, str] = Depends(require_customer_key),
):
    customer_name, owner_id = customer_auth

    try:
        config = await fetch_config_from_control_plane(owner_id)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Configuration unavailable: {e}")

    call_id = str(uuid.uuid4())
    session_token = str(uuid.uuid4())

    origin_url = request.headers.get("Referer", str(request.url))
    user_agent = request.headers.get("User-Agent", "")

    record = CallRecord(
        id=call_id,
        session_token=session_token,
        assistant_config_id=config.get("assistant", {}).get("id"),
        state="connecting",
        origin_url=origin_url,
        user_agent=user_agent,
    )
    db.add(record)
    db.commit()

    await set_cached_config(session_token, config)

    forwarded_proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    forwarded_host = request.headers.get("x-forwarded-host", request.headers.get("host", request.url.hostname))
    ws_scheme = "wss" if forwarded_proto == "https" else "ws"
    ws_url = f"{ws_scheme}://{forwarded_host}/ws/call?token={session_token}"

    logger.info("Call initiated: call_id=%s customer=%r owner=%s token=%s...", call_id, customer_name, owner_id, session_token[:8])
    return {
        "call_id": call_id,
        "session_token": session_token,
        "ws_url": ws_url,
    }


@router.get("/stats")
def get_stats(db: Session = Depends(get_db)):
    """Dashboard statistics for the admin UI."""
    total = db.query(func.count(CallRecord.id)).scalar()
    completed = db.query(func.count(CallRecord.id)).filter(CallRecord.state == "done").scalar()
    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    today = db.query(func.count(CallRecord.id)).filter(CallRecord.started_at >= today_start).scalar()
    avg_duration = db.query(func.avg(CallRecord.duration_sec)).filter(
        CallRecord.duration_sec.isnot(None)
    ).scalar()
    errors = db.query(func.count(CallRecord.id)).filter(CallRecord.state == "error").scalar()

    return {
        "total_calls": total,
        "completed_calls": completed,
        "calls_today": today,
        "error_calls": errors,
        "avg_duration_sec": round(avg_duration, 1) if avg_duration else 0,
        "completion_rate": round((completed / total * 100) if total else 0, 1),
    }


@router.get("")
def list_calls(
    state: str | None = Query(None),
    search: str | None = Query(None),
    limit: int = Query(100, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    q = db.query(CallRecord)
    if state:
        q = q.filter(CallRecord.state == state)
    if search:
        q = q.filter(
            CallRecord.caller_name.ilike(f"%{search}%")
        )
    total = q.count()
    records = q.order_by(CallRecord.started_at.desc()).offset(offset).limit(limit).all()
    return {
        "total": total,
        "calls": [_format_call(r) for r in records],
    }


@router.get("/{call_id}")
def get_call(call_id: str, db: Session = Depends(get_db)):
    record = db.get(CallRecord, call_id)
    if not record:
        raise HTTPException(status_code=404, detail="Call not found")

    params = db.query(GatheredParameter).filter_by(call_id=call_id).all()
    analytics = db.query(CallAnalytics).filter_by(call_id=call_id).order_by(CallAnalytics.recorded_at).all()

    return {
        "call": _format_call(record),
        "gathered_parameters": [
            {
                "parameter_name": p.parameter_name,
                "raw_value": p.raw_value,
                "normalized_value": p.normalized_value,
                "validated": p.validated,
                "collected_at": p.collected_at.isoformat() if p.collected_at else None,
            }
            for p in params
        ],
        "analytics": [
            {
                "event_name": a.event_name,
                "stage": a.stage,
                "latency_ms": round(a.latency_ms, 1),
                "token_count": a.token_count,
            }
            for a in analytics
        ],
    }


@router.get("/{call_id}/transcript")
def get_transcript(call_id: str, db: Session = Depends(get_db)):
    record = db.get(CallRecord, call_id)
    if not record:
        raise HTTPException(status_code=404, detail="Call not found")
    if not record.transcript_path:
        raise HTTPException(status_code=404, detail="Transcript not available")
    path = Path(record.transcript_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Transcript file not found on disk")
    return {"call_id": call_id, "transcript": path.read_text(encoding="utf-8")}


@router.get("/{call_id}/analytics")
def get_analytics(call_id: str, db: Session = Depends(get_db)):
    record = db.get(CallRecord, call_id)
    if not record:
        raise HTTPException(status_code=404, detail="Call not found")
    events = db.query(CallAnalytics).filter_by(call_id=call_id).order_by(CallAnalytics.recorded_at).all()
    return {
        "call_id": call_id,
        "duration_sec": record.duration_sec,
        "events": [
            {
                "event_name": e.event_name,
                "stage": e.stage,
                "latency_ms": round(e.latency_ms, 1),
                "token_count": e.token_count,
            }
            for e in events
        ],
    }
