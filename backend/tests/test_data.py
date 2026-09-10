"""Dataset contract and data-quality tests.

These encode what the profiling step actually found, so a swapped or corrupted
CSV fails loudly instead of silently producing a different model.
"""
from __future__ import annotations

import pandas as pd

from app.ml.constants import (
    COL_FAILURE,
    COL_LINE,
    COL_MACHINE,
    COL_RUN_HOURS,
    COL_TEMP,
    COL_TIMESTAMP,
    COL_VIB,
    RAW_COLUMNS,
)


def test_csv_loads_with_expected_columns(raw_df: pd.DataFrame):
    assert list(raw_df.columns) == RAW_COLUMNS
    assert len(raw_df) > 40_000


def test_timestamps_parse(raw_df: pd.DataFrame):
    ts = pd.to_datetime(raw_df[COL_TIMESTAMP], errors="coerce")
    assert ts.isna().sum() == 0


def test_machine_and_line_counts(clean_df: pd.DataFrame):
    assert clean_df[COL_MACHINE].nunique() == 17
    assert clean_df[COL_LINE].nunique() == 3


def test_failure_label_is_binary_and_rare(clean_df: pd.DataFrame):
    assert set(clean_df[COL_FAILURE].unique()) <= {0, 1}
    n_failures = int(clean_df[COL_FAILURE].sum())
    assert n_failures == 17
    # Extreme imbalance is the defining property of this problem.
    assert clean_df[COL_FAILURE].mean() < 0.001


def test_duplicates_are_removed(raw_df: pd.DataFrame, clean_df: pd.DataFrame):
    assert raw_df.duplicated(subset=[COL_MACHINE, COL_TIMESTAMP]).sum() == 10
    assert clean_df.duplicated(subset=[COL_MACHINE, COL_TIMESTAMP]).sum() == 0
    assert clean_df.attrs["duplicates_removed"] == 10


def test_missing_sensor_values_are_imputed_and_flagged(raw_df: pd.DataFrame, clean_df: pd.DataFrame):
    assert raw_df[COL_TEMP].isna().sum() > 0
    assert raw_df[COL_VIB].isna().sum() > 0
    # After cleaning nothing is missing, but every fill is recorded.
    assert clean_df[COL_TEMP].isna().sum() == 0
    assert clean_df[COL_VIB].isna().sum() == 0
    assert clean_df["temp_imputed"].sum() == raw_df[COL_TEMP].isna().sum()
    assert clean_df["vib_imputed"].sum() == raw_df[COL_VIB].isna().sum()


def test_hourly_series_is_gap_free_per_machine(clean_df: pd.DataFrame):
    for _, g in clean_df.groupby(COL_MACHINE):
        deltas = g.sort_values(COL_TIMESTAMP)[COL_TIMESTAMP].diff().dropna()
        assert (deltas == pd.Timedelta(hours=1)).all()


def test_two_machines_are_newly_commissioned(clean_df: pd.DataFrame):
    counts = clean_df.groupby(COL_MACHINE).size()
    short = counts[counts < 14 * 24]
    assert sorted(short.index.tolist()) == ["MCH-300", "MCH-301"]
    assert set(short.values) == {72}


def test_run_hours_reset_at_every_failure(clean_df: pd.DataFrame):
    """The run-hours counter is zeroed at each failure hour.

    This is why run-hours at time t is safe as a feature but the failure row
    itself must be excluded from training.
    """
    failures = clean_df[clean_df[COL_FAILURE] == 1]
    assert (failures[COL_RUN_HOURS] == 0).all()


def test_sensor_ranges_are_physically_plausible(clean_df: pd.DataFrame):
    assert clean_df[COL_TEMP].between(40, 100).all()
    assert clean_df[COL_VIB].between(0, 5).all()
    assert (clean_df[COL_RUN_HOURS] >= 0).all()
