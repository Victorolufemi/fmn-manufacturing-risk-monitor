"""Candidate models and baselines.

The dataset is small (43k hourly rows, 17 failure episodes), so the bar for
adding model complexity is high. We start from two honest baselines and only
adopt a learned model if it beats them on the metric that matters.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class MajorityBaseline(BaseEstimator, ClassifierMixin):
    """Predicts the training prevalence for every row.

    This is the "do nothing" reference: it has high accuracy and zero business
    value, which is exactly why accuracy is not our headline metric.
    """

    def fit(self, X, y):
        self.rate_ = float(np.mean(y))
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = np.full(len(X), self.rate_)
        return np.column_stack([1 - p, p])


class SensorRuleBaseline(BaseEstimator, ClassifierMixin):
    """The rule a maintenance engineer would write without any ML.

    Flag a machine when temperature or vibration sits more than k standard
    deviations above that machine's own trailing baseline. Scores are squashed
    with a logistic so the output is comparable to a probability and can share
    the same threshold sweep as the learned models.
    """

    def __init__(self, k: float = 2.0, scale: float = 1.0):
        self.k = k
        self.scale = scale

    def fit(self, X, y=None):
        cols = list(X.columns)
        self.temp_ix_ = cols.index("temp_z_vs_baseline")
        self.vib_ix_ = cols.index("vib_z_vs_baseline")
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        arr = np.asarray(X, dtype=float)
        tz = np.nan_to_num(arr[:, self.temp_ix_], nan=0.0)
        vz = np.nan_to_num(arr[:, self.vib_ix_], nan=0.0)
        z = np.maximum(tz, vz)
        p = 1.0 / (1.0 + np.exp(-(z - self.k) / self.scale))
        return np.column_stack([1 - p, p])


def make_logistic(balanced: bool = True) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    C=0.5,
                    class_weight="balanced" if balanced else None,
                    random_state=42,
                ),
            ),
        ]
    )


def make_random_forest(balanced: bool = True) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=400,
                    max_depth=8,
                    min_samples_leaf=20,
                    class_weight="balanced_subsample" if balanced else None,
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )


def make_hist_gbm() -> HistGradientBoostingClassifier:
    """Handles NaN natively, so warm-up rows need no imputation."""
    return HistGradientBoostingClassifier(
        max_iter=250,
        learning_rate=0.06,
        max_depth=4,
        min_samples_leaf=30,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=42,
    )


def candidate_models() -> dict:
    """name -> (estimator factory, supports_sample_weight)."""
    return {
        "majority_baseline": (MajorityBaseline, False),
        "sensor_rule_baseline": (SensorRuleBaseline, False),
        "logistic_regression": (lambda: make_logistic(balanced=True), False),
        "random_forest": (lambda: make_random_forest(balanced=True), False),
        "hist_gbm": (make_hist_gbm, False),
        "hist_gbm_balanced": (make_hist_gbm, True),
    }
