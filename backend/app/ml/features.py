"""Leak-free feature engineering.

Every feature at timestamp t is computed from observations at t or earlier, for
that machine only. Nothing in this module ever looks forward, and nothing is
computed across the full dataset in a way that would let the test period
influence the training period.

Machine-specific baselines
--------------------------
Machines run at genuinely different operating points (machine medians in this
dataset span 56.5 degC to 70.1 degC, and 0.30 to 1.26 mm/s), so a global
"temperature above X" rule would systematically over-flag hot machines and
under-flag cool ones. Each machine therefore gets its own trailing baseline:

    baseline(t) = expanding median of that machine's readings up to t - 24h

The 24h lag stops an in-progress degradation ramp from inflating the machine's
own notion of "normal". Machines with fewer than 72 usable historical hours
(the newly commissioned ones) fall back to a pooled production-line baseline
learned from established machines during training. That is the cold-start path,
and it is recorded per row in the baseline_source column.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .constants import (
    BASELINE_LAG_HOURS,
    COLD_START_HOURS,
    COL_FAILURE,
    COL_LINE,
    COL_MACHINE,
    COL_RUN_HOURS,
    COL_TEMP,
    COL_TIMESTAMP,
    COL_VIB,
    MAX_IMPUTE_GAP_HOURS,
    MIN_HOURS_FOR_OWN_BASELINE,
    ROLLING_WINDOWS,
)

EPS = 1e-6


@dataclass
class BaselineBook:
    """Pooled fallback baselines, fitted on the training window only."""

    per_line: dict = field(default_factory=dict)
    global_: dict = field(default_factory=dict)

    def lookup(self, line: str) -> dict:
        return self.per_line.get(line, self.global_)

    def to_dict(self) -> dict:
        return {"per_line": self.per_line, "global": self.global_}

    @classmethod
    def from_dict(cls, d: dict) -> "BaselineBook":
        return cls(per_line=d.get("per_line", {}), global_=d.get("global", {}))


def fit_baseline_book(train_df: pd.DataFrame) -> BaselineBook:
    """Pooled per-line baselines from established machines in the training window.

    Only machines with a full history (at least COLD_START_HOURS) contribute, so
    the newly commissioned machines never define the fallback they depend on.
    """
    counts = train_df.groupby(COL_MACHINE)[COL_TIMESTAMP].size()
    established = counts[counts >= COLD_START_HOURS].index
    est = train_df[train_df[COL_MACHINE].isin(established)]

    def stats(frame: pd.DataFrame) -> dict:
        return {
            "temp_median": float(frame[COL_TEMP].median()),
            "temp_std": float(max(frame[COL_TEMP].std(), EPS)),
            "vib_median": float(frame[COL_VIB].median()),
            "vib_std": float(max(frame[COL_VIB].std(), EPS)),
        }

    per_line = {str(line): stats(g) for line, g in est.groupby(COL_LINE)}
    return BaselineBook(per_line=per_line, global_=stats(est))


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------
def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate, sort, and impute short sensor gaps (flagging what we filled)."""
    df = df.copy()
    df[COL_TIMESTAMP] = pd.to_datetime(df[COL_TIMESTAMP])
    before = len(df)
    df = df.drop_duplicates(subset=[COL_MACHINE, COL_TIMESTAMP], keep="first")
    removed = before - len(df)
    df = df.sort_values([COL_MACHINE, COL_TIMESTAMP]).reset_index(drop=True)

    df["temp_imputed"] = df[COL_TEMP].isna().astype(np.int8)
    df["vib_imputed"] = df[COL_VIB].isna().astype(np.int8)

    for col in (COL_TEMP, COL_VIB):
        df[col] = df.groupby(COL_MACHINE, sort=False)[col].transform(
            lambda s: s.interpolate(limit=MAX_IMPUTE_GAP_HOURS, limit_direction="both")
        )
        df[col] = df.groupby(COL_MACHINE, sort=False)[col].ffill()
        df[col] = df.groupby(COL_MACHINE, sort=False)[col].bfill()

    df[COL_RUN_HOURS] = df[COL_RUN_HOURS].astype(float)
    df[COL_FAILURE] = df[COL_FAILURE].astype(int)
    df.attrs["duplicates_removed"] = removed
    return df


# --------------------------------------------------------------------------
# Feature building
# --------------------------------------------------------------------------
def _slope(series: pd.Series, window: int) -> pd.Series:
    """Least-squares slope (units per hour) over a trailing window."""
    x = np.arange(window, dtype=float)
    x_centered = x - x.mean()

    def _f(vals):
        n = len(vals)
        xc = x_centered[-n:]
        xc = xc - xc.mean()
        denom = max(float((xc ** 2).sum()), EPS)
        return float(np.dot(vals - vals.mean(), xc) / denom)

    return series.rolling(window, min_periods=max(3, window // 3)).apply(_f, raw=True)


def _machine_features(g: pd.DataFrame, book: BaselineBook) -> pd.DataFrame:
    g = g.sort_values(COL_TIMESTAMP).copy()
    temp, vib = g[COL_TEMP], g[COL_VIB]

    out = pd.DataFrame(index=g.index)
    out["temperature_c"] = temp
    out["vibration_mm_s"] = vib
    out["run_hours_since_maintenance"] = g[COL_RUN_HOURS]
    out["log_run_hours"] = np.log1p(g[COL_RUN_HOURS])

    # --- trailing self-baseline (lagged so the current event cannot inflate it)
    hist_temp = temp.shift(BASELINE_LAG_HOURS)
    hist_vib = vib.shift(BASELINE_LAG_HOURS)
    mp = MIN_HOURS_FOR_OWN_BASELINE
    own_t_med = hist_temp.expanding(min_periods=mp).median()
    own_t_std = hist_temp.expanding(min_periods=mp).std()
    own_v_med = hist_vib.expanding(min_periods=mp).median()
    own_v_std = hist_vib.expanding(min_periods=mp).std()

    fb = book.lookup(str(g[COL_LINE].iloc[0]))
    has_own = own_t_med.notna() & own_v_med.notna()

    t_med = own_t_med.fillna(fb["temp_median"])
    t_std = own_t_std.fillna(fb["temp_std"]).clip(lower=EPS)
    v_med = own_v_med.fillna(fb["vib_median"])
    v_std = own_v_std.fillna(fb["vib_std"]).clip(lower=EPS)

    out["baseline_temperature"] = t_med
    out["baseline_vibration"] = v_med
    out["temp_dev_from_baseline"] = temp - t_med
    out["vib_dev_from_baseline"] = vib - v_med
    out["temp_z_vs_baseline"] = (temp - t_med) / t_std
    out["vib_z_vs_baseline"] = (vib - v_med) / v_std
    out["vib_ratio_to_baseline"] = vib / (v_med + EPS)

    # --- rolling statistics
    for w in ROLLING_WINDOWS:
        m = max(2, w // 3)
        out["temp_roll_mean_%dh" % w] = temp.rolling(w, min_periods=m).mean()
        out["vib_roll_mean_%dh" % w] = vib.rolling(w, min_periods=m).mean()
        if w <= 24:
            out["temp_roll_std_%dh" % w] = temp.rolling(w, min_periods=m).std()
            out["vib_roll_std_%dh" % w] = vib.rolling(w, min_periods=m).std()
            out["temp_roll_max_%dh" % w] = temp.rolling(w, min_periods=m).max()
            out["vib_roll_max_%dh" % w] = vib.rolling(w, min_periods=m).max()

    out["temp_roll_mean_24h_z"] = (out["temp_roll_mean_24h"] - t_med) / t_std
    out["vib_roll_mean_24h_z"] = (out["vib_roll_mean_24h"] - v_med) / v_std

    # --- change / trend
    for h in (1, 6, 24):
        out["temp_change_%dh" % h] = temp.diff(h)
        out["vib_change_%dh" % h] = vib.diff(h)

    out["temp_slope_24h"] = _slope(temp, 24)
    out["vib_slope_24h"] = _slope(vib, 24)
    out["temp_slope_6h"] = _slope(temp, 6)
    out["vib_slope_6h"] = _slope(vib, 6)
    # Acceleration: is the trend itself steepening?
    out["temp_accel"] = out["temp_slope_6h"] - out["temp_slope_24h"]
    out["vib_accel"] = out["vib_slope_6h"] - out["vib_slope_24h"]
    # Short-term vs long-term average.
    out["temp_short_vs_long"] = out["temp_roll_mean_6h"] - out["temp_roll_mean_72h"]
    out["vib_short_vs_long"] = out["vib_roll_mean_6h"] - out["vib_roll_mean_72h"]

    # --- maintenance / failure history
    fail = g[COL_FAILURE].to_numpy()
    n = len(g)
    since = np.full(n, np.nan)
    last = -1
    for i in range(n):
        if last >= 0:
            since[i] = i - last
        if fail[i] == 1:
            last = i
    out["has_prior_failure"] = (~np.isnan(since)).astype(np.int8)
    out["hours_since_last_failure"] = pd.Series(since, index=g.index).fillna(9999.0)
    out["prior_failure_count"] = np.concatenate([[0], np.cumsum(fail)[:-1]]).astype(float)

    # --- temporal
    hod = g[COL_TIMESTAMP].dt.hour
    out["hour_sin"] = np.sin(2 * np.pi * hod / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hod / 24)
    out["day_of_week"] = g[COL_TIMESTAMP].dt.dayofweek.astype(float)

    # --- data quality
    out["temp_imputed"] = g["temp_imputed"].astype(float)
    out["vib_imputed"] = g["vib_imputed"].astype(float)

    # --- bookkeeping (never fed to the model)
    out[COL_MACHINE] = g[COL_MACHINE]
    out[COL_LINE] = g[COL_LINE]
    out[COL_TIMESTAMP] = g[COL_TIMESTAMP]
    out[COL_FAILURE] = g[COL_FAILURE]
    out["hours_of_history"] = np.arange(1, n + 1)
    out["baseline_source"] = np.where(has_own.to_numpy(), "machine", "line_pooled")
    return out


NON_FEATURE_COLUMNS = [
    COL_MACHINE,
    COL_LINE,
    COL_TIMESTAMP,
    COL_FAILURE,
    "hours_of_history",
    "baseline_source",
    "baseline_temperature",
    "baseline_vibration",
    "failure_within_horizon",
    "eligible",
    "censored",
    "preventive_maintenance",
]


def build_features(df: pd.DataFrame, book: BaselineBook) -> pd.DataFrame:
    """Build the full per-row feature frame for every machine."""
    parts = [_machine_features(g, book) for _, g in df.groupby(COL_MACHINE, sort=False)]
    return (
        pd.concat(parts)
        .sort_values([COL_MACHINE, COL_TIMESTAMP])
        .reset_index(drop=True)
    )


def feature_columns(feats: pd.DataFrame) -> list:
    """Model input columns, in a stable order."""
    return [c for c in feats.columns if c not in NON_FEATURE_COLUMNS]
