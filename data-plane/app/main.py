import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import init_db, SessionLocal
from app.routers import calls
from app.services.config_client import get_cached_config, invalidate_token
from app.websocket.handler import handle_call
from app.websocket.session import CallSession

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
# basicConfig is a no-op when uvicorn has already installed root handlers.
# Explicitly set the level on the app namespace so all app.* loggers are
# guaranteed to emit at the configured level regardless of who set up first.
_app_log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
logging.getLogger("app").setLevel(_app_log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.ensure_directories()
    init_db()
    logger.info("Data plane started on %s:%d", settings.app_host, settings.app_port)
    yield
    logger.info("Data plane shutting down")


app = FastAPI(
    title="Voice Booking Data Plane",
    description="Real-time voice call handling for the appointment booking system",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(calls.router)


@app.get("/api/v1/health")
def health():
    return {"status": "ok", "service": "data-plane"}


@app.websocket("/ws/call")
async def websocket_call(
    websocket: WebSocket,
    token: str = Query(...),
):
    """Main WebSocket endpoint for voice and text calls. Mode is determined by the call record."""
    await websocket.accept()

    config = await get_cached_config(token)
    if not config:
        logger.warning("WebSocket connection with invalid/expired token: %s...", token[:8])
        await websocket.send_text('{"type":"server.error","payload":{"code":"invalid_token","fatal":true}}')
        await websocket.close(code=4001)
        return

    db = SessionLocal()
    try:
        from app.models.call_record import CallRecord
        record = db.query(CallRecord).filter_by(session_token=token).first()
        if not record:
            await websocket.close(code=4002)
            return

        call_id = record.id
        mode = record.mode or "voice"
        session = CallSession(
            call_id=call_id,
            session_token=token,
            config=config,
        )

        logger.info("WebSocket opened for call %s (mode=%s)", call_id, mode)
        await handle_call(websocket, session, db, mode=mode)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for token %s...", token[:8])
    except Exception as e:
        logger.error("WebSocket handler error: %s", e, exc_info=True)
    finally:
        await invalidate_token(token)
        db.close()
        logger.info("WebSocket closed for call_id=%s", call_id if "call_id" in locals() else "unknown")
