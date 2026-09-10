"""Model, scoring, calibration and risk-banding tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.ml.constants import COL_MACHINE, COL_TIMESTAMP
from app.ml.evaluation import episode_metrics, row_metrics, sweep_thresholds
from app.ml.labeling import add_target
from app.ml.models import MajorityBaseline, SensorRuleBaseline, make_hist_gbm, make_logistic
from app.ml.risk import PlattCalibrator, RiskBands, RiskScorer, choose_bands, fit_calibrator
from app.services import data_service as ds

from .conftest import requires_artifacts


# --------------------------------------------------------------------------
# Training / inference
# --------------------------------------------------------------------------
def test_model_trains_and_predicts_valid_probabilities(clean_df, features_df, feature_cols):
    lab = add_target(clean_df, 72)
    df = features_df.merge(
        lab[[COL_MACHINE, COL_TIMESTAMP, "failure_within_horizon", "eligible"]],
        on=[COL_MACHINE, COL_TIMESTAMP],
    )
    df = df[df["eligible"]]
    train = df[df[COL_TIMESTAMP] < "2026-03-01"]
    test = df[df[COL_TIMESTAMP] >= "2026-03-01"]

    model = make_hist_gbm()
    model.fit(train[feature_cols], train["failure_within_horizon"])
    p = model.predict_proba(test[feature_cols])[:, 1]

    assert p.shape == (len(test),)
    assert np.all((p >= 0) & (p <= 1))
    assert not np.isnan(p).any()
    # A model with any signal must beat the base rate substantially.
    m = row_metrics(test["failure_within_horizon"].to_numpy(), p, 0.5)
    assert m["pr_auc"] > m["positive_rate"] * 5


def test_logistic_handles_warmup_nans(clean_df, features_df, feature_cols):
    """Warm-up rows carry NaNs; the pipeline must impute rather than crash."""
    lab = add_target(clean_df, 72)
    df = features_df.merge(
        lab[[COL_MACHINE, COL_TIMESTAMP, "failure_within_horizon", "eligible"]],
        on=[COL_MACHINE, COL_TIMESTAMP],
    )
    df = df[df["eligible"]].head(20000)
    assert df[feature_cols].isna().any().any()
    model = make_logistic()
    model.fit(df[feature_cols], df["failure_within_horizon"])
    p = model.predict_proba(df[feature_cols])[:, 1]
    assert np.all((p >= 0) & (p <= 1))


def test_majority_baseline_is_useless_but_accurate(clean_df, features_df, feature_cols):
    lab = add_target(clean_df, 72)
    df = features_df.merge(
        lab[[COL_MACHINE, COL_TIMESTAMP, "failure_within_horizon", "eligible"]],
        on=[COL_MACHINE, COL_TIMESTAMP],
    )
    df = df[df["eligible"]]
    y = df["failure_within_horizon"].to_numpy()
    m = MajorityBaseline().fit(df[feature_cols], y)
    p = m.predict_proba(df[feature_cols])[:, 1]
    # Predicting the base rate everywhere: it never raises an alert at 0.5,
    # which is exactly why accuracy is not the headline metric.
    assert (p >= 0.5).sum() == 0
    assert (1 - y).mean() > 0.97  # "always no failure" would be 97%+ accurate


def test_sensor_rule_baseline_ranks_better_than_chance(clean_df, features_df, feature_cols):
    lab = add_target(clean_df, 72)
    df = features_df.merge(
        lab[[COL_MACHINE, COL_TIMESTAMP, "failure_within_horizon", "eligible"]],
        on=[COL_MACHINE, COL_TIMESTAMP],
    )
    df = df[df["eligible"]]
    y = df["failure_within_horizon"].to_numpy()
    rule = SensorRuleBaseline().fit(df[feature_cols], y)
    p = rule.predict_proba(df[feature_cols])[:, 1]
    m = row_metrics(y, p, 0.5)
    assert m["roc_auc"] > 0.8


# --------------------------------------------------------------------------
# Calibration and bands
# --------------------------------------------------------------------------
def test_platt_calibration_is_monotone_and_unsaturated():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.03, 5000)
    raw = np.clip(rng.beta(1, 30, 5000) + y * 0.4, 1e-4, 1 - 1e-4)
    cal = fit_calibrator(raw, y)
    p = cal.predict(raw)
    order_raw = np.argsort(raw)
    # Monotone: calibration cannot reorder machines.
    assert np.all(np.diff(p[order_raw]) >= -1e-9)
    # Never claims absolute certainty.
    assert p.max() < 1.0 and p.min() > 0.0


def test_calibration_does_not_change_ranking_metrics():
    rng = np.random.default_rng(1)
    y = rng.binomial(1, 0.05, 3000)
    raw = np.clip(rng.beta(1, 20, 3000) + y * 0.3, 1e-4, 1 - 1e-4)
    cal = fit_calibrator(raw, y)
    before = row_metrics(y, raw, 0.5)
    after = row_metrics(y, cal.predict(raw), 0.5)
    assert before["roc_auc"] == pytest.approx(after["roc_auc"], abs=1e-9)
    assert before["pr_auc"] == pytest.approx(after["pr_auc"], abs=1e-9)


def test_risk_bands_are_ordered_and_classify_correctly():
    bands = RiskBands(medium=0.05, high=0.2, critical=0.7)
    assert bands.medium < bands.high < bands.critical
    assert bands.level(0.01) == "LOW"
    assert bands.level(0.05) == "MEDIUM"
    assert bands.level(0.19) == "MEDIUM"
    assert bands.level(0.2) == "HIGH"
    assert bands.level(0.69) == "HIGH"
    assert bands.level(0.7) == "CRITICAL"
    assert bands.level(1.0) == "CRITICAL"


def test_choose_bands_keeps_strict_ordering():
    sweep = sweep_thresholds(
        np.array([0] * 900 + [1] * 100),
        np.concatenate([np.linspace(0.001, 0.4, 900), np.linspace(0.3, 0.99, 100)]),
    )
    bands = choose_bands(sweep, float(sweep.loc[sweep["f2"].idxmax(), "threshold"]))
    assert 0 < bands.medium < bands.high < bands.critical <= 1.0


def test_risk_scorer_applies_calibration():
    class Stub:
        def predict_proba(self, X):
            p = np.asarray(X["x"], dtype=float)
            return np.column_stack([1 - p, p])

    cal = PlattCalibrator().fit(
        np.array([0.01, 0.02, 0.9, 0.95]), np.array([0, 0, 1, 1])
    )
    scorer = RiskScorer(Stub(), cal)
    out = scorer.risk_score(pd.DataFrame({"x": [0.01, 0.95]}))
    assert out[0] < out[1]
    assert np.all((out >= 0) & (out <= 1))


# --------------------------------------------------------------------------
# Evaluation helpers
# --------------------------------------------------------------------------
def test_episode_metrics_counts_detection_and_lead_time():
    ts = pd.date_range("2026-01-01", periods=100, freq="h")
    frame = pd.DataFrame({COL_MACHINE: "M1", COL_TIMESTAMP: ts})
    prob = np.zeros(100)
    prob[60:80] = 0.9  # alerting from hour 60
    failures = pd.DataFrame({COL_MACHINE: ["M1"], COL_TIMESTAMP: [ts[80]]})

    out = episode_metrics(frame, failures, prob, threshold=0.5, horizon_hours=24)
    assert out["episodes_total"] == 1
    assert out["episodes_detected"] == 1
    assert out["episode_recall"] == 1.0
    assert out["median_lead_time_hours"] == 20.0
    assert out["false_alert_events"] == 0


def test_episode_metrics_counts_false_alerts_outside_the_window():
    ts = pd.date_range("2026-01-01", periods=100, freq="h")
    frame = pd.DataFrame({COL_MACHINE: "M1", COL_TIMESTAMP: ts})
    prob = np.zeros(100)
    prob[5:9] = 0.9  # nowhere near a failure
    failures = pd.DataFrame({COL_MACHINE: ["M1"], COL_TIMESTAMP: [ts[80]]})
    out = episode_metrics(frame, failures, prob, threshold=0.5, horizon_hours=24)
    assert out["episodes_detected"] == 0
    assert out["false_alert_events"] == 1


def test_threshold_sweep_is_monotone_in_recall():
    rng = np.random.default_rng(3)
    y = rng.binomial(1, 0.1, 2000)
    p = np.clip(rng.random(2000) * 0.5 + y * 0.4, 0, 1)
    sweep = sweep_thresholds(y, p)
    assert sweep["recall"].is_monotonic_decreasing
    assert (sweep["threshold"].diff().dropna() > 0).all()


# --------------------------------------------------------------------------
# Served artifacts
# --------------------------------------------------------------------------
@requires_artifacts
def test_served_scores_are_in_range_and_banded_consistently():
    art = ds.load_artifacts()
    h = art.history
    assert h["risk_score"].between(0, 1).all()
    for level, group in h.groupby("risk_level"):
        recomputed = {art.bands.level(p) for p in group["risk_score"]}
        assert recomputed == {level}


@requires_artifacts
def test_ranking_puts_the_failing_machine_on_top():
    """MCH-203 failed on 2026-04-30 04:00; the day before it must rank first."""
    as_of = pd.Timestamp("2026-04-29 12:00:00")
    snap = ds.snapshot(as_of).sort_values("risk_score", ascending=False)
    assert snap.iloc[0][COL_MACHINE] == "MCH-203"
    assert snap.iloc[0]["risk_level"] == "CRITICAL"


@requires_artifacts
def test_risk_trend_detects_a_rise():
    trend = ds.risk_trend("MCH-203", pd.Timestamp("2026-04-27 12:00:00"), 24)
    assert trend["direction"] == "RISING"
    assert trend["delta"] > 0


@requires_artifacts
def test_cold_start_machines_are_flagged():
    art = ds.load_artifacts()
    cold = art.metadata["cold_start"]["machines"]
    assert sorted(cold) == ["MCH-300", "MCH-301"]
    for m in cold:
        assert ds.is_cold_start(m)
        assert ds.history_hours(m, art.latest_timestamp) == 72
    assert not ds.is_cold_start("MCH-200")


@requires_artifacts
def test_as_of_never_returns_future_rows():
    as_of = pd.Timestamp("2026-03-15 06:00:00")
    snap = ds.snapshot(as_of)
    assert (snap[COL_TIMESTAMP] <= as_of).all()
    hist = ds.machine_history("MCH-200", as_of)
    assert hist[COL_TIMESTAMP].max() <= as_of


@requires_artifacts
def test_as_of_is_clamped_to_the_dataset_range():
    art = ds.load_artifacts()
    assert ds.resolve_as_of(None) == art.latest_timestamp
    assert ds.resolve_as_of("2099-01-01") == art.latest_timestamp
    assert ds.resolve_as_of("1990-01-01") == art.earliest_timestamp
    with pytest.raises(ValueError):
        ds.resolve_as_of("not-a-date")


@requires_artifacts
def test_reported_metrics_are_present_and_plausible():
    """Guards against shipping a model card with missing or absurd numbers."""
    art = ds.load_artifacts()
    m = art.metadata["headline_metrics"]
    for key in ("test_pr_auc", "test_precision", "test_recall", "test_roc_auc"):
        assert 0.0 <= m[key] <= 1.0
    assert m["test_episodes_total"] == 7
    assert m["test_episodes_detected"] <= m["test_episodes_total"]
    cm = m["test_confusion_matrix"]
    assert sum(cm.values()) == art.metadata["validation"]["test_rows"]
