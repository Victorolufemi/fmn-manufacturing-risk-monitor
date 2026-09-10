"""Explainability: what is driving a machine's risk score.

Two complementary views, both computed from the actual fitted model.

1. Global, grouped importance (training time)
   Per-feature permutation importance is misleading here because the 47
   features are heavily correlated by construction - permuting
   vib_roll_mean_6h barely moves the score while vib_roll_mean_24h carries the
   same information. We therefore permute whole feature families together, so
   the reported number answers "how much does the model rely on vibration
   variability at all", which is both statistically sound and the question a
   plant engineer is actually asking.

2. Local counterfactual drivers (request time)
   For a specific machine at a specific hour we ask, for each feature family:
   "if this were sitting at the machine's own normal value, what would the risk
   score be?" The drop in risk is the driver's contribution. This is a real
   model evaluation, not a heuristic, and it translates directly into plain
   English: "vibration variability is the reason - at normal variability this
   machine would score 0.08 instead of 0.86".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Feature families. Order matters only for stable display.
FEATURE_GROUPS: dict[str, dict] = {
    "vibration_level": {
        "label": "Vibration level",
        "description": "How high vibration is right now versus this machine's normal.",
        "prefixes": [
            "vibration_mm_s",
            "vib_roll_mean_",
            "vib_roll_max_",
            "vib_dev_from_baseline",
            "vib_z_vs_baseline",
            "vib_ratio_to_baseline",
        ],
    },
    "vibration_instability": {
        "label": "Vibration instability",
        "description": "How erratic and fast-rising vibration has been.",
        "prefixes": [
            "vib_roll_std_",
            "vib_change_",
            "vib_slope_",
            "vib_accel",
            "vib_short_vs_long",
        ],
    },
    "temperature_level": {
        "label": "Temperature level",
        "description": "How hot the machine is running versus its own normal.",
        "prefixes": [
            "temperature_c",
            "temp_roll_mean_",
            "temp_roll_max_",
            "temp_dev_from_baseline",
            "temp_z_vs_baseline",
        ],
    },
    "temperature_instability": {
        "label": "Temperature instability",
        "description": "How erratic and fast-rising temperature has been.",
        "prefixes": [
            "temp_roll_std_",
            "temp_change_",
            "temp_slope_",
            "temp_accel",
            "temp_short_vs_long",
        ],
    },
    "maintenance_history": {
        "label": "Maintenance and run hours",
        "description": "Hours run since the last maintenance, and past failures.",
        "prefixes": [
            "run_hours_since_maintenance",
            "log_run_hours",
            "hours_since_last_failure",
            "has_prior_failure",
            "prior_failure_count",
        ],
    },
    "time_of_day": {
        "label": "Time pattern",
        "description": "Hour of day and day of week.",
        "prefixes": ["hour_sin", "hour_cos", "day_of_week"],
    },
    "data_quality": {
        "label": "Sensor data quality",
        "description": "Whether the current readings had to be filled in.",
        "prefixes": ["temp_imputed", "vib_imputed"],
    },
}

# Human-readable names for individual features, used in evidence payloads.
FEATURE_LABELS = {
    "temperature_c": "Temperature",
    "vibration_mm_s": "Vibration",
    "run_hours_since_maintenance": "Run hours since maintenance",
    "temp_z_vs_baseline": "Temperature vs machine baseline (std devs)",
    "vib_z_vs_baseline": "Vibration vs machine baseline (std devs)",
    "temp_dev_from_baseline": "Temperature above machine baseline",
    "vib_dev_from_baseline": "Vibration above machine baseline",
    "temp_roll_mean_24h": "Average temperature, last 24h",
    "vib_roll_mean_24h": "Average vibration, last 24h",
    "temp_roll_std_24h": "Temperature variability, last 24h",
    "vib_roll_std_24h": "Vibration variability, last 24h",
    "temp_change_24h": "Temperature change over 24h",
    "vib_change_24h": "Vibration change over 24h",
    "temp_slope_24h": "Temperature trend, last 24h",
    "vib_slope_24h": "Vibration trend, last 24h",
}


def group_of(feature: str) -> str | None:
    for key, spec in FEATURE_GROUPS.items():
        for p in spec["prefixes"]:
            if feature == p or feature.startswith(p):
                return key
    return None


def resolve_groups(feature_columns: list) -> dict:
    """feature family -> the columns belonging to it, for a given feature set."""
    out: dict = {k: [] for k in FEATURE_GROUPS}
    for c in feature_columns:
        g = group_of(c)
        if g:
            out[g].append(c)
    return {k: v for k, v in out.items() if v}


def grouped_permutation_importance(
    model,
    X: pd.DataFrame,
    y: np.ndarray,
    feature_columns: list,
    n_repeats: int = 5,
    random_state: int = 42,
    max_negatives: int = 6000,
) -> list:
    """Permute whole feature families and measure the drop in average precision."""
    from sklearn.metrics import average_precision_score

    rng = np.random.default_rng(random_state)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    neg = rng.choice(neg, size=min(len(neg), max_negatives), replace=False)
    sel = np.concatenate([pos, neg])
    Xs, ys = X.iloc[sel].reset_index(drop=True), y[sel]

    base = average_precision_score(ys, model.predict_proba(Xs)[:, 1])
    groups = resolve_groups(feature_columns)

    raw = {}
    for key, cols in groups.items():
        drops = []
        for _ in range(n_repeats):
            Xp = Xs.copy()
            perm = rng.permutation(len(Xp))
            for c in cols:
                Xp[c] = Xs[c].to_numpy()[perm]
            drops.append(base - average_precision_score(ys, model.predict_proba(Xp)[:, 1]))
        raw[key] = max(float(np.mean(drops)), 0.0)

    total = sum(raw.values())
    if total <= 0:
        total = 1.0
    return sorted(
        (
            {
                "group": k,
                "label": FEATURE_GROUPS[k]["label"],
                "description": FEATURE_GROUPS[k]["description"],
                "features": groups[k],
                "importance": round(v / total, 5),
                "average_precision_drop": round(v, 5),
            }
            for k, v in raw.items()
        ),
        key=lambda d: -d["importance"],
    )


def local_drivers(
    scorer,
    row: pd.Series,
    normal_row: pd.Series,
    feature_columns: list,
    top_n: int = 4,
) -> list:
    """Counterfactual attribution for one machine-hour.

    For each feature family, replace its features with the machine's own
    "normal" values and re-score. The resulting drop is that family's
    contribution to the current risk score.

    scorer      : RiskScorer (estimator + calibrator)
    row         : the feature row being explained
    normal_row  : the same machine's typical values (median over its history)
    """
    groups = resolve_groups(feature_columns)
    base_frame = pd.DataFrame([row[feature_columns].to_dict()])
    variants, keys = [], []
    for key, cols in groups.items():
        v = base_frame.iloc[0].copy()
        for c in cols:
            v[c] = normal_row.get(c, np.nan)
        variants.append(v)
        keys.append(key)

    batch = pd.DataFrame([base_frame.iloc[0].to_dict()] + [v.to_dict() for v in variants])
    batch = batch[feature_columns].astype(float)
    scores = scorer.risk_score(batch)
    current, counterfactual = float(scores[0]), scores[1:]

    out = []
    for key, cf in zip(keys, counterfactual):
        out.append(
            {
                "group": key,
                "label": FEATURE_GROUPS[key]["label"],
                "description": FEATURE_GROUPS[key]["description"],
                "contribution": round(float(current - cf), 5),
                "risk_if_normal": round(float(cf), 5),
                "evidence": [
                    {
                        "feature": c,
                        "label": FEATURE_LABELS.get(c, c.replace("_", " ").capitalize()),
                        "value": _num(row.get(c)),
                        "typical_value": _num(normal_row.get(c)),
                    }
                    for c in _headline_features(groups[key])
                ],
            }
        )

    out.sort(key=lambda d: -d["contribution"])
    total = sum(max(d["contribution"], 0.0) for d in out) or 1.0
    for d in out:
        d["importance"] = round(max(d["contribution"], 0.0) / total, 4)
    return out[:top_n]


# The one or two most legible features per family, for the evidence panel.
_HEADLINE = {
    "vibration_level": ["vibration_mm_s", "vib_roll_mean_24h", "vib_z_vs_baseline"],
    "vibration_instability": ["vib_roll_std_24h", "vib_change_24h", "vib_slope_24h"],
    "temperature_level": ["temperature_c", "temp_roll_mean_24h", "temp_z_vs_baseline"],
    "temperature_instability": ["temp_roll_std_24h", "temp_change_24h", "temp_slope_24h"],
    "maintenance_history": ["run_hours_since_maintenance", "hours_since_last_failure"],
    "time_of_day": ["day_of_week"],
    "data_quality": ["temp_imputed", "vib_imputed"],
}


def _headline_features(cols: list) -> list:
    for key, wanted in _HEADLINE.items():
        if set(wanted) & set(cols):
            return [c for c in wanted if c in cols]
    return cols[:2]


def _num(v):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if np.isnan(f) or np.isinf(f):
        return None
    return round(f, 4)
