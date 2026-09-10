"""Risk score calibration and banding.

The selected classifier separates well but is not calibrated: at the tuned
operating point a genuinely alarming machine can score 0.03, which is
meaningless to a plant supervisor. We therefore fit a calibrator on the pooled
out-of-fold predictions from the rolling-origin CV, so the number shown in the
UI is an estimate of "probability this machine fails within the horizon", not
an arbitrary model output.

Why Platt (sigmoid) rather than isotonic
----------------------------------------
There are only 17 failure episodes in the whole dataset and 10 in the
development period. The positive *rows* number in the hundreds, but they are
almost perfectly autocorrelated within an episode, so the effective sample size
is the episode count. Isotonic regression is non-parametric and, on data this
well separated with an effective n of 10, collapses to a step function that
emits exactly 0.00 and exactly 1.00 - it claims certainty it has not earned and
is useless for ranking machines inside a band. Platt scaling fits two
parameters, stays smooth, and cannot saturate, which is the right bias/variance
trade at this sample size.

Calibration is monotone either way, so it changes none of the ranking metrics
(PR-AUC, ROC-AUC) - only the numbers people read.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression


@dataclass
class RiskBands:
    """Probability cut-points that turn a score into a plant-floor label."""

    medium: float
    high: float
    critical: float

    def level(self, p: float) -> str:
        if p >= self.critical:
            return "CRITICAL"
        if p >= self.high:
            return "HIGH"
        if p >= self.medium:
            return "MEDIUM"
        return "LOW"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RiskBands":
        return cls(medium=d["medium"], high=d["high"], critical=d["critical"])


PROB_CLIP = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), PROB_CLIP, 1 - PROB_CLIP)
    return np.log(p / (1 - p))


class PlattCalibrator:
    """One-dimensional logistic map from raw score to calibrated probability."""

    def __init__(self) -> None:
        self._lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)

    def fit(self, raw_prob, y) -> "PlattCalibrator":
        self._lr.fit(_logit(raw_prob).reshape(-1, 1), np.asarray(y).astype(int))
        return self

    def predict(self, raw_prob) -> np.ndarray:
        z = _logit(raw_prob).reshape(-1, 1)
        return self._lr.predict_proba(z)[:, 1]


class RiskScorer:
    """Estimator + calibrator, serialised together for serving."""

    def __init__(self, estimator, calibrator: PlattCalibrator | None = None):
        self.estimator = estimator
        self.calibrator = calibrator

    def raw_proba(self, X) -> np.ndarray:
        return self.estimator.predict_proba(X)[:, 1]

    def calibrate(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        if self.calibrator is None:
            return p
        return np.clip(self.calibrator.predict(p), 0.0, 1.0)

    def risk_score(self, X) -> np.ndarray:
        return self.calibrate(self.raw_proba(X))


def fit_calibrator(oof_prob: np.ndarray, y_oof: np.ndarray) -> PlattCalibrator:
    return PlattCalibrator().fit(oof_prob, y_oof)


def choose_bands(sweep, operating_threshold: float) -> RiskBands:
    """Derive the four risk bands from the tuned operating threshold.

    HIGH     the F2-optimal operating threshold - act this shift
    CRITICAL the lowest threshold at or above HIGH whose out-of-fold precision
             reaches 0.90; falls back to the 99.5th percentile of scores
    MEDIUM   the highest threshold below HIGH that still recovers at least 98%
             of positive hours - a watch list, not a call-out
    """
    high = float(operating_threshold)

    crit_rows = sweep[(sweep["threshold"] >= high) & (sweep["precision"] >= 0.90)]
    if len(crit_rows):
        critical = float(crit_rows["threshold"].iloc[0])
    else:
        critical = float(sweep["threshold"].quantile(0.995))
    if critical <= high:
        critical = float(min(0.99, high + max(0.05, high * 0.5)))

    med_rows = sweep[(sweep["threshold"] < high) & (sweep["recall"] >= 0.98)]
    if len(med_rows):
        medium = float(med_rows["threshold"].iloc[-1])
    else:
        medium = high * 0.5
    if medium >= high:
        medium = high * 0.5

    return RiskBands(
        medium=round(medium, 4), high=round(high, 4), critical=round(critical, 4)
    )
