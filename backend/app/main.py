"""FastAPI application entry point.

Model artifacts are loaded once during startup so no request pays that cost.
Startup does not fail if artifacts are missing - /health reports "degraded" and
the data endpoints return 503 with an actionable message, which is far easier to
diagnose on Render than a crash loop.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import settings
from app.services import data_service as ds

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        art = ds.load_artifacts()
        log.info(
            "startup complete: model=%s horizon=%dh machines=%d",
            art.model_name,
            art.horizon_hours,
            len(art.machine_ids),
        )
    except ds.ArtifactsMissingError as exc:
        log.error("startup: %s", exc)
    if not settings.anthropic_api_key:
        log.warning(
            "ANTHROPIC_API_KEY is not set - AI explanations and Q&A will use "
            "the evidence-only fallback path."
        )
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Failure-risk monitoring for plant machinery. Serves calibrated risk "
        "scores, sensor evidence, model-derived risk drivers, runtime AI "
        "explanations and grounded Q&A."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/", include_in_schema=False)
def root() -> dict:
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health",
    }
