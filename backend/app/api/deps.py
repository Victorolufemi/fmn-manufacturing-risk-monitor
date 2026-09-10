"""Shared dependencies and error translation for the API layer."""
from __future__ import annotations

import pandas as pd
from fastapi import HTTPException, Query, status

from app.services import data_service as ds


def as_of_param(
    as_of: str | None = Query(
        None,
        description=(
            "Review timestamp (ISO 8601). Defaults to the latest hour in the "
            "dataset. Values outside the dataset range are clamped."
        ),
        examples=["2026-04-29T12:00:00"],
    ),
) -> pd.Timestamp:
    try:
        return ds.resolve_as_of(as_of)
    except ds.ArtifactsMissingError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc


def require_machine(machine_id: str) -> str:
    try:
        art = ds.load_artifacts()
    except ds.ArtifactsMissingError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    if machine_id not in art.machine_ids:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown machine {machine_id!r}. Known machines: "
            + ", ".join(art.machine_ids),
        )
    return machine_id
