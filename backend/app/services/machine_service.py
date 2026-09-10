"""Turns scored history into the payloads the dashboard and detail pages need."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from app.ml.constants import COL_FAILURE, COL_MACHINE, COL_TIMESTAMP
from app.ml.explainability import local_drivers
from app.services import data_service as ds

log = logging.getLogger(__name__)

TREND_LOOKBACK_HOURS = 24
DEFAULT_HISTORY_HOURS = 14 * 24


def _f(v):
    """Convert to a JSON-safe float, mapping NaN/inf to None."""
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if (np.isnan(x) or np.isinf(x)) else round(x, 4)


def cold_start_info(machine_id: str, row: pd.Series, as_of: pd.Timestamp) -> dict:
    hours = ds.history_hours(machine_id, as_of)
    cold = bool(row.get("is_cold_start", False)) or ds.is_cold_start(machine_id)
    source = str(row.get("baseline_source", "machine"))
    message = None
    if cold:
        message = (
            f"Limited history - this machine has only {hours} hours of "
            f"observations and has not yet experienced a failure. Its risk "
            f"estimate uses the fleet-wide model with a "
            f"{'production-line' if source == 'line_pooled' else 'machine'} "
            f"baseline, and is less reliable than for established machines."
        )
    elif source == "line_pooled":
        message = (
            "This machine does not yet have enough history for its own "
            "baseline, so a production-line baseline is being used."
        )
    return {
        "is_cold_start": cold,
        "history_hours": hours,
        "baseline_source": source,
        "message": message,
    }


def _main_driver(machine_id: str, row: pd.Series) -> str | None:
    """Cheap headline driver for the table - full attribution is on the detail page."""
    art = ds.load_artifacts()
    if float(row.get("risk_score", 0)) < art.bands.medium:
        return None
    # Whichever sensor sits furthest above this machine's own baseline, measured
    # in standard deviations so the two are comparable.
    candidates = [
        ("Temperature above machine baseline", _f(row.get("temp_z_vs_baseline"))),
        ("Vibration above machine baseline", _f(row.get("vib_z_vs_baseline"))),
    ]
    scored = [(label, v) for label, v in candidates if v is not None]
    if not scored:
        return None
    label, value = max(scored, key=lambda t: t[1])
    if value < 1.0:
        # Neither sensor is far from normal; the model is reacting to a
        # combination of trend and instability rather than to a level.
        return "Combined sensor pattern"
    return label


def machine_summary(row: pd.Series, as_of: pd.Timestamp) -> dict:
    art = ds.load_artifacts()
    machine_id = row[COL_MACHINE]
    trend = ds.risk_trend(machine_id, as_of, TREND_LOOKBACK_HOURS)
    return {
        "machine_id": machine_id,
        "line": row.get("line", ""),
        "timestamp": pd.Timestamp(row[COL_TIMESTAMP]).isoformat(),
        "risk_score": _f(row["risk_score"]) or 0.0,
        "risk_level": row["risk_level"],
        "prediction_horizon_hours": art.horizon_hours,
        "temperature_c": _f(row.get("temperature_c")),
        "vibration_mm_s": _f(row.get("vibration_mm_s")),
        "run_hours_since_maintenance": _f(row.get("run_hours_since_maintenance")),
        "trend": trend,
        "main_driver": _main_driver(machine_id, row),
        "cold_start": cold_start_info(machine_id, row, as_of),
    }


def list_machines(as_of: pd.Timestamp) -> list:
    snap = ds.snapshot(as_of)
    out = [machine_summary(r, as_of) for _, r in snap.iterrows()]
    out.sort(key=lambda m: (-m["risk_score"], m["machine_id"]))
    return out


# A degradation episode can dip below the alert threshold for a few hours and
# come back. Treating each dip as a separate alert would tell a supervisor that
# one deteriorating machine raised four alarms, so runs separated by less than
# this many hours are merged into a single event.
ALERT_MERGE_GAP_HOURS = 12


def recent_risk_events(as_of: pd.Timestamp, days: int = 14, limit: int = 8) -> list:
    """Stretches where a machine sat at or above the alert threshold.

    Gives the dashboard something truthful to show when the fleet happens to be
    all-clear at the review time: what the model flagged recently, and whether a
    failure followed.
    """
    art = ds.load_artifacts()
    start = as_of - pd.Timedelta(days=days)
    h = art.history[
        (art.history[COL_TIMESTAMP] > start) & (art.history[COL_TIMESTAMP] <= as_of)
    ]
    events = []
    for machine, g in h.groupby(COL_MACHINE, sort=False):
        g = g.sort_values(COL_TIMESTAMP)
        alert = (g["risk_score"] >= art.bands.high).to_numpy()
        if not alert.any():
            continue
        ts = g[COL_TIMESTAMP].to_numpy()
        scores = g["risk_score"].to_numpy()
        levels = g["risk_level"].to_numpy()

        # Bridge short below-threshold dips so one episode stays one event.
        alert_ix = np.flatnonzero(alert)
        merged = alert.copy()
        for a, b in zip(alert_ix, alert_ix[1:]):
            gap = (pd.Timestamp(ts[b]) - pd.Timestamp(ts[a])).total_seconds() / 3600.0
            if gap <= ALERT_MERGE_GAP_HOURS:
                merged[a : b + 1] = True
        alert = merged

        i = 0
        while i < len(alert):
            if not alert[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(alert) and alert[j + 1]:
                j += 1
            seg_scores = scores[i : j + 1]
            peak_ix = i + int(np.argmax(seg_scores))
            started, ended = pd.Timestamp(ts[i]), pd.Timestamp(ts[j])
            fails = g[
                (g[COL_FAILURE] == 1)
                & (g[COL_TIMESTAMP] >= started)
                & (g[COL_TIMESTAMP] <= ended + pd.Timedelta(hours=art.horizon_hours))
            ]
            failure_at = pd.Timestamp(fails[COL_TIMESTAMP].iloc[0]) if len(fails) else None
            events.append(
                {
                    "machine_id": machine,
                    "started_at": started.isoformat(),
                    "ended_at": ended.isoformat(),
                    "peak_risk_score": _f(scores[peak_ix]) or 0.0,
                    "peak_risk_level": levels[peak_ix],
                    "resulted_in_failure": failure_at is not None,
                    "failure_at": failure_at.isoformat() if failure_at is not None else None,
                    "warning_hours": (
                        round((failure_at - started).total_seconds() / 3600.0, 1)
                        if failure_at is not None
                        else None
                    ),
                }
            )
            i = j + 1
    events.sort(key=lambda e: e["started_at"], reverse=True)
    return events[:limit]


def dashboard(as_of: pd.Timestamp) -> dict:
    art = ds.load_artifacts()
    machines = list_machines(as_of)
    counts = {
        "total_machines": len(machines),
        "critical": sum(1 for m in machines if m["risk_level"] == "CRITICAL"),
        "high": sum(1 for m in machines if m["risk_level"] == "HIGH"),
        "medium": sum(1 for m in machines if m["risk_level"] == "MEDIUM"),
        "low": sum(1 for m in machines if m["risk_level"] == "LOW"),
        "newly_commissioned": sum(1 for m in machines if m["cold_start"]["is_cold_start"]),
        "trending_up": sum(1 for m in machines if m["trend"]["direction"] == "RISING"),
    }
    attention = [m for m in machines if m["risk_level"] in ("CRITICAL", "HIGH", "MEDIUM")]
    if not attention:
        # Nothing is elevated: show the three highest-ranked machines anyway so
        # the table is never empty and the ranking stays visible.
        attention = machines[:3]
    return {
        "as_of": as_of.isoformat(),
        "data_range": {
            "start": art.earliest_timestamp.isoformat(),
            "end": art.latest_timestamp.isoformat(),
        },
        "prediction_horizon_hours": art.horizon_hours,
        "counts": counts,
        "machines": machines,
        "attention_required": attention,
        "recent_risk_events": recent_risk_events(as_of),
        "risk_bands": art.bands.to_dict(),
    }


def _sensor_block(row: pd.Series, kind: str) -> dict:
    if kind == "temperature":
        return {
            "current": _f(row.get("temperature_c")),
            "rolling_mean_24h": _f(row.get("temp_roll_mean_24h")),
            "change_24h": _f(row.get("temp_change_24h")),
            "baseline": _f(row.get("baseline_temperature")),
            "deviation_from_baseline": _f(row.get("temp_dev_from_baseline")),
            "z_vs_baseline": _f(row.get("temp_z_vs_baseline")),
            "unit": "degC",
        }
    return {
        "current": _f(row.get("vibration_mm_s")),
        "rolling_mean_24h": _f(row.get("vib_roll_mean_24h")),
        "change_24h": _f(row.get("vib_change_24h")),
        "baseline": _f(row.get("baseline_vibration")),
        "deviation_from_baseline": _f(row.get("vib_dev_from_baseline")),
        "z_vs_baseline": _f(row.get("vib_z_vs_baseline")),
        "unit": "mm/s",
    }


def drivers_for(machine_id: str, row: pd.Series, top_n: int = 4) -> list:
    """Counterfactual driver attribution from the actual fitted model."""
    art = ds.load_artifacts()
    try:
        return local_drivers(
            art.scorer,
            row,
            ds.machine_normal_row(machine_id),
            art.feature_columns,
            top_n=top_n,
        )
    except Exception:  # pragma: no cover - never break the page on attribution
        log.exception("driver attribution failed for %s", machine_id)
        return []


def machine_detail(machine_id: str, as_of: pd.Timestamp, history_hours: int | None = None) -> dict:
    art = ds.load_artifacts()
    row = ds.latest_row(machine_id, as_of)
    if row is None:
        raise KeyError(machine_id)

    hours = history_hours or DEFAULT_HISTORY_HOURS
    hist = ds.machine_history(machine_id, as_of, hours=hours)
    points = [
        {
            "timestamp": pd.Timestamp(r[COL_TIMESTAMP]).isoformat(),
            "temperature_c": _f(r["temperature_c"]),
            "vibration_mm_s": _f(r["vibration_mm_s"]),
            "run_hours_since_maintenance": _f(r["run_hours_since_maintenance"]),
            "risk_score": _f(r["risk_score"]),
            "risk_level": r["risk_level"],
            "failure_event": int(r[COL_FAILURE]),
        }
        for _, r in hist.iterrows()
    ]

    failures = ds.failure_events(machine_id, as_of)
    return {
        "machine_id": machine_id,
        "line": row.get("line", ""),
        "timestamp": pd.Timestamp(row[COL_TIMESTAMP]).isoformat(),
        "risk_score": _f(row["risk_score"]) or 0.0,
        "risk_level": row["risk_level"],
        "prediction_horizon_hours": art.horizon_hours,
        "trend": ds.risk_trend(machine_id, as_of, TREND_LOOKBACK_HOURS),
        "temperature": _sensor_block(row, "temperature"),
        "vibration": _sensor_block(row, "vibration"),
        "run_hours_since_maintenance": _f(row.get("run_hours_since_maintenance")),
        "drivers": drivers_for(machine_id, row),
        "history": points,
        "failure_history": [pd.Timestamp(t).isoformat() for t in failures[COL_TIMESTAMP]],
        "maintenance_history": ds.maintenance_events(machine_id, as_of),
        "cold_start": cold_start_info(machine_id, row, as_of),
        "score_is_out_of_sample": bool(row.get("score_is_out_of_sample", False)),
    }
