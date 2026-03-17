import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import init_db
from app.routers import assistant, parameters, faqs, context_files, spell_rules, webhooks, config_export, calendly, customer_keys

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.ensure_directories()
    init_db()
    logger.info("Control plane started on %s:%d", settings.app_host, settings.app_port)
    yield
    logger.info("Control plane shutting down")


app = FastAPI(
    title="Voice Booking Control Plane",
    description="Configuration management for the voice appointment booking system",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(assistant.router)
app.include_router(parameters.router)
app.include_router(faqs.router)
app.include_router(context_files.router)
app.include_router(spell_rules.router)
app.include_router(webhooks.router)
app.include_router(config_export.router)
app.include_router(calendly.router)
app.include_router(customer_keys.router)


@app.get("/api/v1/health")
def health():
    return {"status": "ok", "service": "control-plane"}
