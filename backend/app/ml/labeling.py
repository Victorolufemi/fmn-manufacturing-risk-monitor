"""Target construction and censoring rules.

Business question
-----------------
"Is this machine at elevated risk of failure soon?"

Target
------
``failure_within_horizon`` = 1 when at least one ``failure_event`` occurs in the
window ``(t, t + H]`` for that machine, where H is the prediction horizon in
hours. The window is strictly forward-looking and *excludes* t itself, so no
information from the prediction timestamp's own failure flag leaks in.

Censoring
---------
The dataset contains two kinds of ``run_hours_since_maintenance`` resets:

1. resets that coincide with ``failure_event == 1``  -> unplanned failure
2. resets with no failure flag                       -> planned/preventive
   maintenance (5 occurrences)

The preventive-maintenance machines show sustained elevated vibration in the
hours leading up to the intervention. We do not know whether they *would* have
failed, so labelling those hours as negatives would actively teach the model
that rising vibration is harmless. Instead they are treated as **censored**:
rows in the H hours before a preventive reset are excluded from training and
evaluation.

The failure hour itself is also excluded - at that instant the machine is
already down, which is an observation, not a prediction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import COL_FAILURE, COL_MACHINE, COL_RUN_HOURS, COL_TIMESTAMP


def _forward_window_has_failure(failures: np.ndarray, horizon: int) -> np.ndarray:
    """1 if any failure occurs in the half-open window (i, i+horizon].

    The window excludes row i itself, so a row can never be labelled positive by
    its own failure flag, and includes i+horizon, so a horizon of H really does
    cover H hours of look-ahead.

    Assumes a gap-free hourly index, which the profiling step verified.
    """
    n = len(failures)
    out = np.zeros(n, dtype=np.int8)
    csum = np.concatenate([[0], np.cumsum(failures)])
    for i in range(n):
        start = min(i + 1, n)
        end = min(i + horizon + 1, n)
        out[i] = 1 if (csum[end] - csum[start]) > 0 else 0
    return out


def add_target(df: pd.DataFrame, horizon_hours: int) -> pd.DataFrame:
    """Attach `failure_within_horizon` and censoring/eligibility flags."""
    df = df.sort_values([COL_MACHINE, COL_TIMESTAMP]).reset_index(drop=True)
    parts = []
    for _, g in df.groupby(COL_MACHINE, sort=False):
        g = g.copy()
        fail = g[COL_FAILURE].to_numpy(dtype=np.int64)
        g["failure_within_horizon"] = _forward_window_has_failure(fail, horizon_hours)

        # Preventive maintenance = run-hours reset without a failure flag.
        run_h = g[COL_RUN_HOURS].to_numpy(dtype=np.float64)
        reset = np.zeros(len(g), dtype=np.int64)
        reset[1:] = (np.diff(run_h) < 0).astype(np.int64)
        preventive = ((reset == 1) & (fail == 0)).astype(np.int64)
        g["preventive_maintenance"] = preventive

        # Censor the H hours preceding a preventive intervention.
        censored = _forward_window_has_failure(preventive, horizon_hours).astype(bool)
        # ...and the intervention/failure hour itself.
        censored |= (fail == 1)
        censored |= (preventive == 1)
        g["censored"] = censored

        # Rows usable for supervised learning / scoring evaluation.
        g["eligible"] = ~censored | (g["failure_within_horizon"] == 1)
        # A row that is both censored and positive keeps its positive label
        # (a real failure is about to happen); pure censoring only removes
        # ambiguous negatives.
        g.loc[g[COL_FAILURE] == 1, "eligible"] = False
        g.loc[g["preventive_maintenance"] == 1, "eligible"] = False
        parts.append(g)

    return pd.concat(parts, ignore_index=True)


def horizon_label_summary(df: pd.DataFrame) -> dict:
    """Diagnostics used by the horizon-selection report."""
    elig = df[df["eligible"]]
    return {
        "rows_total": int(len(df)),
        "rows_eligible": int(len(elig)),
        "rows_censored": int((~df["eligible"]).sum()),
        "positives": int(elig["failure_within_horizon"].sum()),
        "positive_rate": float(elig["failure_within_horizon"].mean()),
        "failure_episodes": int(df[COL_FAILURE].sum()),
    }
