"""HTTP routes.

All ML, data and LLM work happens behind these handlers. Nothing here returns
secrets, model artifacts, or raw internal configuration.
"""
from __future__ import annotations

import logging

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.deps import as_of_param, require_machine
from app.config import settings
from app.schemas.machine import (
    Dashboard,
    Explanation,
    MachineDetail,
    MachineSummary,
    QARequest,
    QAResponse,
)
from app.services import data_service as ds
from app.services import explanation_service, llm_client
from app.services import machine_service as ms
from app.services import qa_service

log = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", tags=["system"], summary="Liveness and readiness")
def health() -> dict:
    """Always 200 if the process is up; reports whether artifacts are loaded."""
    out = {
        "status": "ok",
        "app": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "model_loaded": False,
        "llm_configured": llm_client.is_available(),
    }
    try:
        art = ds.load_artifacts()
        out.update(
            model_loaded=True,
            model_name=art.model_name,
            prediction_horizon_hours=art.horizon_hours,
            machines=len(art.machine_ids),
            data_end=art.latest_timestamp.isoformat(),
        )
    except ds.ArtifactsMissingError as exc:
        out["status"] = "degraded"
        out["detail"] = str(exc)
    return out


@router.get("/api/metadata", tags=["system"], summary="Dataset and app metadata")
def metadata() -> dict:
    art = _artifacts()
    cold = art.metadata.get("cold_start", {})
    return {
        "data_range": {
            "start": art.earliest_timestamp.isoformat(),
            "end": art.latest_timestamp.isoformat(),
        },
        "default_as_of": art.latest_timestamp.isoformat(),
        "machines": art.machine_ids,
        "lines": sorted(art.history["line"].dropna().unique().tolist()),
        "prediction_horizon_hours": art.horizon_hours,
        "risk_bands": art.bands.to_dict(),
        "cold_start_machines": cold.get("machines", []),
        "llm_configured": llm_client.is_available(),
        "suggested_questions": qa_service.SUGGESTED_QUESTIONS,
    }


@router.get("/api/model-info", tags=["system"], summary="Model card")
def model_info() -> dict:
    """What the model is, how it was validated, and how well it actually did.

    Deliberately excludes the artifact itself and any credentials.
    """
    art = _artifacts()
    m = art.metadata
    return {
        "model_name": m.get("model_name"),
        "model_version": m.get("model_version"),
        "trained_at": m.get("trained_at"),
        "prediction_horizon_hours": m.get("horizon_hours"),
        "horizon_rationale": m.get("horizon_rationale"),
        "n_features": m.get("n_features"),
        "risk_bands": m.get("risk_bands"),
        "threshold": m.get("threshold"),
        "calibration": m.get("calibration"),
        "validation": m.get("validation"),
        "headline_metrics": m.get("headline_metrics"),
        "model_comparison": m.get("model_comparison"),
        "group_importance": m.get("group_importance"),
        "cold_start": m.get("cold_start"),
        "dataset": m.get("dataset"),
        "llm": {
            "configured": llm_client.is_available(),
            "model": settings.model_name if llm_client.is_available() else None,
        },
    }


@router.get("/api/dashboard", response_model=Dashboard, tags=["fleet"])
def dashboard(as_of: pd.Timestamp = Depends(as_of_param)) -> dict:
    _artifacts()
    return ms.dashboard(as_of)


@router.get("/api/machines", response_model=list[MachineSummary], tags=["fleet"])
def machines(
    as_of: pd.Timestamp = Depends(as_of_param),
    risk_level: str | None = Query(None, description="CRITICAL | HIGH | MEDIUM | LOW"),
    trend: str | None = Query(None, description="RISING | STABLE | FALLING"),
    cohort: str | None = Query(None, description="new | established"),
    search: str | None = Query(None, description="Substring match on machine ID"),
) -> list:
    _artifacts()
    rows = ms.list_machines(as_of)
    if risk_level:
        wanted = {r.strip().upper() for r in risk_level.split(",")}
        rows = [r for r in rows if r["risk_level"] in wanted]
    if trend:
        wanted = {t.strip().upper() for t in trend.split(",")}
        rows = [r for r in rows if r["trend"]["direction"] in wanted]
    if cohort:
        want_new = cohort.strip().lower() == "new"
        rows = [r for r in rows if r["cold_start"]["is_cold_start"] == want_new]
    if search:
        s = search.strip().lower()
        rows = [r for r in rows if s in r["machine_id"].lower()]
    return rows


@router.get("/api/machines/{machine_id}", response_model=MachineDetail, tags=["fleet"])
def machine_detail(
    machine_id: str = Depends(require_machine),
    as_of: pd.Timestamp = Depends(as_of_param),
    history_hours: int = Query(336, ge=24, le=24 * 130, description="Chart window"),
) -> dict:
    try:
        return ms.machine_detail(machine_id, as_of, history_hours=history_hours)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No readings for {machine_id} at or before {as_of.isoformat()}",
        ) from exc


@router.get(
    "/api/machines/{machine_id}/explanation",
    response_model=Explanation,
    tags=["ai"],
    summary="Runtime, evidence-grounded AI explanation",
)
def explanation(
    machine_id: str = Depends(require_machine),
    as_of: pd.Timestamp = Depends(as_of_param),
) -> dict:
    try:
        return explanation_service.explain(machine_id, as_of)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No readings for {machine_id} at or before {as_of.isoformat()}",
        ) from exc


@router.post("/api/qa", response_model=QAResponse, tags=["ai"], summary="Grounded Q&A")
def qa(payload: QARequest) -> dict:
    _artifacts()
    try:
        as_of = ds.resolve_as_of(payload.as_of)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return qa_service.answer(payload.question.strip(), as_of)


def _artifacts():
    try:
        return ds.load_artifacts()
    except ds.ArtifactsMissingError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
