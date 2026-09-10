"""Artifact loading and read-only data access.

Everything expensive happens once, at process start:
  * model.joblib           the fitted scorer, calibrator and feature list
  * metadata.json          horizon, thresholds, bands, headline metrics
  * scored_history.parquet one walk-forward risk score per machine-hour

No endpoint re-reads the CSV, rebuilds features or refits anything. The raw CSV
is only touched by the offline training job.

"As of" time
------------
The dataset is a fixed historical extract ending 2026-04-30 23:00. The service
therefore treats "now" as an explicit review timestamp that defaults to the
latest hour in the data. Every read is filtered to at-or-before that instant, so
the app never shows a plant user information from after the moment they are
looking at.
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from app.config import settings
from app.ml.constants import COL_FAILURE, COL_MACHINE, COL_TIMESTAMP
from app.ml.risk import RiskBands

log = logging.getLogger(__name__)

_LOCK = threading.Lock()


class ArtifactsMissingError(RuntimeError):
    """Raised when the model artifacts have not been built yet."""


@dataclass
class Artifacts:
    scorer: object
    serving_scorer: object
    feature_columns: list
    horizon_hours: int
    bands: RiskBands
    model_name: str
    feature_importance: list
    group_importance: list
    machine_normals: pd.DataFrame
    metadata: dict
    history: pd.DataFrame

    # ---------------- derived, computed once ----------------
    @property
    def latest_timestamp(self) -> pd.Timestamp:
        return self._latest

    def __post_init__(self) -> None:
        self.history = self.history.sort_values([COL_MACHINE, COL_TIMESTAMP]).reset_index(
            drop=True
        )
        self._latest = pd.Timestamp(self.history[COL_TIMESTAMP].max())
        self._earliest = pd.Timestamp(self.history[COL_TIMESTAMP].min())
        self._machines = sorted(self.history[COL_MACHINE].unique().tolist())
        self._by_machine = {m: g for m, g in self.history.groupby(COL_MACHINE, sort=False)}

    @property
    def earliest_timestamp(self) -> pd.Timestamp:
        return self._earliest

    @property
    def machine_ids(self) -> list:
        return list(self._machines)

    def machine_frame(self, machine_id: str) -> pd.DataFrame:
        try:
            return self._by_machine[machine_id]
        except KeyError as exc:
            raise KeyError(machine_id) from exc


def _read_history(model_dir: Path, metadata: dict) -> pd.DataFrame:
    name = metadata.get("scored_history_file", "scored_history.parquet")
    path = model_dir / name
    if not path.exists():
        for alt in ("scored_history.parquet", "scored_history.csv.gz"):
            if (model_dir / alt).exists():
                path = model_dir / alt
                break
    if not path.exists():
        raise ArtifactsMissingError(f"scored history not found in {model_dir}")
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, compression="gzip")
    df[COL_TIMESTAMP] = pd.to_datetime(df[COL_TIMESTAMP])
    return df


@lru_cache(maxsize=1)
def load_artifacts() -> Artifacts:
    """Load every artifact once. Thread-safe via lru_cache + module lock."""
    with _LOCK:
        model_dir = Path(settings.model_dir)
        model_path = model_dir / "model.joblib"
        meta_path = model_dir / "metadata.json"
        if not model_path.exists() or not meta_path.exists():
            raise ArtifactsMissingError(
                f"Model artifacts not found in {model_dir}. "
                "Run: python -m app.ml.training (from the backend directory)."
            )
        bundle = joblib.load(model_path)
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        history = _read_history(model_dir, metadata)

        art = Artifacts(
            scorer=bundle["scorer"],
            serving_scorer=bundle.get("serving_scorer", bundle["scorer"]),
            feature_columns=bundle["feature_columns"],
            horizon_hours=int(bundle["horizon_hours"]),
            bands=RiskBands.from_dict(bundle["risk_bands"]),
            model_name=bundle["model_name"],
            feature_importance=bundle.get("feature_importance", []),
            group_importance=bundle.get("group_importance", []),
            machine_normals=bundle.get("machine_normals", pd.DataFrame()),
            metadata=metadata,
            history=history,
        )
        log.info(
            "artifacts loaded: model=%s horizon=%dh machines=%d rows=%d",
            art.model_name,
            art.horizon_hours,
            len(art.machine_ids),
            len(art.history),
        )
        return art


# --------------------------------------------------------------------------
# Query helpers
# --------------------------------------------------------------------------
def resolve_as_of(as_of: str | None) -> pd.Timestamp:
    """Clamp a requested review time into the dataset range."""
    art = load_artifacts()
    if not as_of:
        return art.latest_timestamp
    try:
        ts = pd.Timestamp(as_of)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid timestamp: {as_of!r}") from exc
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return min(max(ts, art.earliest_timestamp), art.latest_timestamp)


def machine_history(machine_id: str, as_of: pd.Timestamp, hours: int | None = None) -> pd.DataFrame:
    art = load_artifacts()
    g = art.machine_frame(machine_id)
    g = g[g[COL_TIMESTAMP] <= as_of]
    if hours is not None and len(g):
        g = g[g[COL_TIMESTAMP] > as_of - pd.Timedelta(hours=hours)]
    return g


def latest_row(machine_id: str, as_of: pd.Timestamp) -> pd.Series | None:
    g = machine_history(machine_id, as_of)
    return None if g.empty else g.iloc[-1]


def snapshot(as_of: pd.Timestamp) -> pd.DataFrame:
    """One row per machine: its most recent observation at or before as_of."""
    art = load_artifacts()
    h = art.history[art.history[COL_TIMESTAMP] <= as_of]
    if h.empty:
        return h
    return h.sort_values(COL_TIMESTAMP).groupby(COL_MACHINE, as_index=False).tail(1)


def risk_trend(machine_id: str, as_of: pd.Timestamp, lookback_hours: int = 24) -> dict:
    """Change in risk score over a lookback window, and a direction label."""
    g = machine_history(machine_id, as_of, hours=lookback_hours + 1)
    if g.empty:
        return {"delta": 0.0, "direction": "STABLE", "previous_score": None}
    current = float(g["risk_score"].iloc[-1])
    previous = float(g["risk_score"].iloc[0])
    delta = current - previous
    # 0.02 absolute is roughly the width of the LOW band; below that the score
    # is not meaningfully moving.
    if delta > 0.02:
        direction = "RISING"
    elif delta < -0.02:
        direction = "FALLING"
    else:
        direction = "STABLE"
    return {
        "delta": round(delta, 4),
        "direction": direction,
        "previous_score": round(previous, 4),
        "lookback_hours": lookback_hours,
    }


def failure_events(machine_id: str | None = None, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    art = load_artifacts()
    f = art.history[art.history[COL_FAILURE] == 1]
    if machine_id:
        f = f[f[COL_MACHINE] == machine_id]
    if as_of is not None:
        f = f[f[COL_TIMESTAMP] <= as_of]
    return f[[COL_MACHINE, COL_TIMESTAMP]].reset_index(drop=True)


def maintenance_events(machine_id: str, as_of: pd.Timestamp) -> list:
    """Every run-hours reset, classified as a failure repair or planned work."""
    g = machine_history(machine_id, as_of)
    if g.empty:
        return []
    rh = g["run_hours_since_maintenance"].to_numpy()
    reset = np.zeros(len(g), dtype=bool)
    reset[1:] = np.diff(rh) < 0
    out = []
    for ts, is_fail, r in zip(g[COL_TIMESTAMP], g[COL_FAILURE], reset):
        if not r:
            continue
        out.append(
            {
                "timestamp": ts.isoformat(),
                "type": "FAILURE_REPAIR" if is_fail == 1 else "PLANNED_MAINTENANCE",
            }
        )
    return out


def machine_normal_row(machine_id: str) -> pd.Series:
    """The machine's typical feature values, used for counterfactual drivers."""
    art = load_artifacts()
    normals = art.machine_normals
    if isinstance(normals, pd.DataFrame) and machine_id in normals.index:
        return normals.loc[machine_id]
    # Fall back to the fleet median if a machine is unknown to the artifact.
    if isinstance(normals, pd.DataFrame) and len(normals):
        return normals.median()
    return pd.Series(dtype=float)


def is_cold_start(machine_id: str) -> bool:
    art = load_artifacts()
    cold = art.metadata.get("cold_start", {}).get("machines", [])
    return machine_id in cold


def history_hours(machine_id: str, as_of: pd.Timestamp) -> int:
    g = machine_history(machine_id, as_of)
    return int(len(g))
