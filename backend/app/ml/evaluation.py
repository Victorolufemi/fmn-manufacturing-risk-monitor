"""Evaluation utilities.

Row-level metrics alone are misleading here: with only 17 failure episodes the
positive rows are heavily clustered, so a model that catches one long episode
can look better than one that catches three short ones. We therefore report
both:

* row-level: PR-AUC, ROC-AUC, precision, recall, F1, confusion matrix
* episode-level: did the model raise at least one alert during the horizon
  window preceding each real failure, how much warning did it give, and how
  many false alerts did the plant team absorb per machine-day to get there

The episode-level view is the one a plant manager actually cares about.

Note on comparing horizons: PR-AUC is bounded below by the positive rate, which
grows with the horizon, so raw PR-AUC is not comparable across horizons.
pr_auc_lift (PR-AUC divided by the positive rate) is.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .constants import COL_MACHINE, COL_TIMESTAMP


def row_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    rate = float(y_true.mean()) if len(y_true) else 0.0
    out = {
        "threshold": float(threshold),
        "n": int(len(y_true)),
        "positives": int(y_true.sum()),
        "positive_rate": rate,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if 0 < y_true.sum() < len(y_true):
        out["pr_auc"] = float(average_precision_score(y_true, y_prob))
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        out["pr_auc_lift"] = round(out["pr_auc"] / rate, 2) if rate else float("nan")
    else:
        out["pr_auc"] = float("nan")
        out["roc_auc"] = float("nan")
        out["pr_auc_lift"] = float("nan")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out["confusion_matrix"] = {
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }
    return out


def episode_metrics(
    frame: pd.DataFrame,
    failures: pd.DataFrame,
    y_prob: np.ndarray,
    threshold: float,
    horizon_hours: int,
) -> dict:
    """Alert coverage per real failure episode, plus operational alert load.

    frame  : scored rows, must carry machine_id and timestamp, aligned to y_prob
    failures: the real failure events (machine_id, timestamp) to be judged
    """
    df = frame[[COL_MACHINE, COL_TIMESTAMP]].copy()
    df["prob"] = np.asarray(y_prob)
    df["alert"] = df["prob"] >= threshold
    df = df.sort_values([COL_MACHINE, COL_TIMESTAMP]).reset_index(drop=True)

    episodes = []
    in_true_window = np.zeros(len(df), dtype=bool)
    machines = df[COL_MACHINE].to_numpy()
    times = df[COL_TIMESTAMP].to_numpy()

    for _, row in failures.iterrows():
        machine, ts = row[COL_MACHINE], pd.Timestamp(row[COL_TIMESTAMP])
        lo = np.datetime64(ts - pd.Timedelta(hours=horizon_hours))
        hi = np.datetime64(ts)
        mask = (machines == machine) & (times > lo) & (times < hi)
        in_true_window |= mask
        if not mask.any():
            # The pre-failure window lies outside the scored period.
            continue
        sub = df[mask]
        caught = bool(sub["alert"].any())
        lead = None
        if caught:
            first = sub.loc[sub["alert"], COL_TIMESTAMP].min()
            lead = round((ts - first).total_seconds() / 3600.0, 1)
        episodes.append(
            {
                "machine_id": machine,
                "failure_time": ts.isoformat(),
                "detected": caught,
                "lead_time_hours": lead,
                "max_score_in_window": round(float(sub["prob"].max()), 4),
            }
        )

    df["in_true_window"] = in_true_window

    # Count contiguous runs of alerts that sit outside any true pre-failure
    # window - i.e. how many separate false call-outs the team receives.
    false_alert_runs = 0
    for _, g in df.groupby(COL_MACHINE, sort=False):
        prev = False
        for a, w in zip(g["alert"].to_numpy(), g["in_true_window"].to_numpy()):
            cur = bool(a and not w)
            if cur and not prev:
                false_alert_runs += 1
            prev = cur

    n_ep = len(episodes)
    n_caught = sum(1 for e in episodes if e["detected"])
    leads = [e["lead_time_hours"] for e in episodes if e["lead_time_hours"] is not None]
    machine_days = len(df) / 24.0
    return {
        "episodes_total": n_ep,
        "episodes_detected": n_caught,
        "episode_recall": round(n_caught / n_ep, 4) if n_ep else None,
        "median_lead_time_hours": float(np.median(leads)) if leads else None,
        "min_lead_time_hours": float(np.min(leads)) if leads else None,
        "false_alert_events": int(false_alert_runs),
        "machine_days_scored": round(machine_days, 1),
        "false_alerts_per_machine_day": (
            round(false_alert_runs / machine_days, 4) if machine_days else None
        ),
        "episodes": episodes,
    }


def evaluate(
    frame: pd.DataFrame,
    failures: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    horizon_hours: int,
) -> dict:
    out = row_metrics(y_true, y_prob, threshold)
    out["episode"] = episode_metrics(frame, failures, y_prob, threshold, horizon_hours)
    return out


def threshold_grid(y_prob: np.ndarray, n: int = 200) -> np.ndarray:
    """Grid drawn from the actual score distribution.

    A fixed 0.01 grid is useless when a well-separating model puts almost all
    of its mass below 0.05, so we take quantiles of the observed scores.
    """
    p = np.asarray(y_prob, dtype=float)
    qs = np.unique(np.quantile(p, np.linspace(0.50, 0.9999, n)))
    qs = np.unique(np.round(qs, 6))
    return qs[(qs > 0) & (qs < 1)]


def sweep_thresholds(y_true: np.ndarray, y_prob: np.ndarray, grid=None) -> pd.DataFrame:
    """Precision/recall/F-beta across a threshold grid."""
    y_true = np.asarray(y_true).astype(int)
    if grid is None:
        grid = threshold_grid(y_prob)
    rows = []
    for t in grid:
        pred = (y_prob >= t).astype(int)
        p = precision_score(y_true, pred, zero_division=0)
        r = recall_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)
        # F2 weights recall 2x - a missed failure costs far more than a check.
        f2 = (5 * p * r / (4 * p + r)) if (p + r) > 0 else 0.0
        rows.append(
            {
                "threshold": float(t),
                "precision": float(p),
                "recall": float(r),
                "f1": float(f1),
                "f2": float(f2),
                "alerts": int(pred.sum()),
            }
        )
    return pd.DataFrame(rows)
