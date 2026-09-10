"""Shared, documented constants for the manufacturing risk model.

Every value here is a deliberate engineering decision; the rationale lives in
`backend/reports/model_summary.md`.
"""
from __future__ import annotations

# Raw dataset column names (verified against the supplied CSV).
COL_TIMESTAMP = "timestamp"
COL_MACHINE = "machine_id"
COL_LINE = "line"
COL_TEMP = "temperature_c"
COL_VIB = "vibration_mm_s"
COL_RUN_HOURS = "run_hours_since_maintenance"
COL_FAILURE = "failure_event"

RAW_COLUMNS = [
    COL_TIMESTAMP,
    COL_MACHINE,
    COL_LINE,
    COL_TEMP,
    COL_VIB,
    COL_RUN_HOURS,
    COL_FAILURE,
]

# Prediction horizons compared during model development (hours).
CANDIDATE_HORIZONS = [12, 24, 48, 72, 96, 120]

# Minimum hours of observation before a machine gets its own statistical
# baseline. Below this we fall back to the production-line pooled baseline.
MIN_HOURS_FOR_OWN_BASELINE = 72

# The baseline deliberately ignores the most recent 24h so that an in-progress
# degradation event cannot contaminate the machine's own "normal".
BASELINE_LAG_HOURS = 24

# A machine with fewer than this many hours of history is treated as
# newly commissioned (cold start) and flagged in the UI.
COLD_START_HOURS = 14 * 24  # 336h / 14 days

# Rolling windows used for feature engineering (hours).
ROLLING_WINDOWS = [6, 24, 72]

# Maximum forward-fill span for missing sensor readings (hours).
MAX_IMPUTE_GAP_HOURS = 3

# Risk bands. Chosen relative to the tuned operating threshold; see
# `risk_bands_from_threshold`.
RISK_LEVELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

# Chronological split boundaries (inclusive start, exclusive end).
# Development period is used for rolling-origin CV + threshold selection.
# The test period is touched exactly once, at the end.
DEV_END = "2026-04-06 00:00:00"

# Rolling-origin CV fold boundaries inside the development period.
CV_FOLDS = [
    # (train_end, valid_end)
    ("2026-02-15 00:00:00", "2026-03-05 00:00:00"),
    ("2026-03-05 00:00:00", "2026-03-20 00:00:00"),
    ("2026-03-20 00:00:00", "2026-04-06 00:00:00"),
]
