"""Feature-engineering tests, above all the leakage tests.

The single most important property of this pipeline is that a feature computed
at time t never depends on anything after t. test_no_future_leakage checks that
directly rather than by inspection: features built from a truncated history must
be bit-for-bit identical to features built from the full history.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.ml.constants import COL_MACHINE, COL_TEMP, COL_TIMESTAMP, COL_VIB
from app.ml.features import build_features, feature_columns, fit_baseline_book
from app.ml.labeling import add_target

CUTOFF = "2026-03-01 00:00:00"


def test_no_future_leakage(clean_df: pd.DataFrame, baseline_book):
    """Truncating the future must not change any past feature value."""
    truncated = clean_df[clean_df[COL_TIMESTAMP] < CUTOFF].copy()
    full = build_features(clean_df, baseline_book)
    part = build_features(truncated, baseline_book)

    cols = feature_columns(part)
    keys = [COL_MACHINE, COL_TIMESTAMP]
    merged = part[keys + cols].merge(
        full[keys + cols], on=keys, suffixes=("_part", "_full")
    )
    assert len(merged) == len(part)

    for c in cols:
        a = merged[f"{c}_part"].to_numpy(dtype=float)
        b = merged[f"{c}_full"].to_numpy(dtype=float)
        both_nan = np.isnan(a) & np.isnan(b)
        assert np.allclose(a[~both_nan], b[~both_nan], equal_nan=False), (
            f"feature {c} changed when future data was added - this is leakage"
        )


def test_rolling_mean_matches_manual_trailing_window(clean_df: pd.DataFrame, features_df):
    machine = "MCH-200"
    raw = clean_df[clean_df[COL_MACHINE] == machine].sort_values(COL_TIMESTAMP)
    feat = features_df[features_df[COL_MACHINE] == machine].sort_values(COL_TIMESTAMP)
    i = 500
    expected = raw[COL_TEMP].iloc[i - 23 : i + 1].mean()
    assert feat["temp_roll_mean_24h"].iloc[i] == pytest.approx(expected)


def test_change_features_are_backward_differences(clean_df: pd.DataFrame, features_df):
    machine = "MCH-201"
    raw = clean_df[clean_df[COL_MACHINE] == machine].sort_values(COL_TIMESTAMP)
    feat = features_df[features_df[COL_MACHINE] == machine].sort_values(COL_TIMESTAMP)
    i = 300
    expected = raw[COL_VIB].iloc[i] - raw[COL_VIB].iloc[i - 24]
    assert feat["vib_change_24h"].iloc[i] == pytest.approx(expected)


def test_baseline_is_machine_specific(clean_df: pd.DataFrame, features_df):
    """Machines run at genuinely different operating points.

    This is the whole reason for machine-relative features: a global
    "temperature above X" rule would over-flag the hot machines and miss the
    cool ones entirely.
    """
    raw_temp = clean_df.groupby(COL_MACHINE)[COL_TEMP].median()
    raw_vib = clean_df.groupby(COL_MACHINE)[COL_VIB].median()
    assert raw_temp.max() - raw_temp.min() > 10.0
    assert raw_vib.max() / raw_vib.min() > 3.0

    # The learned baseline must track those real differences for machines that
    # have enough history to own one. The two cold-start machines deliberately
    # carry a pooled line baseline instead, so they are excluded here.
    established = [m for m in raw_temp.index if m not in ("MCH-300", "MCH-301")]
    base_temp = features_df.groupby(COL_MACHINE)["baseline_temperature"].median()
    assert base_temp.max() - base_temp.min() > 8.0
    assert base_temp.loc[established].corr(raw_temp.loc[established]) > 0.99


def test_baseline_excludes_the_most_recent_24h(clean_df: pd.DataFrame, baseline_book):
    """A spike in the last 24h must not move the machine's own baseline."""
    machine = "MCH-205"
    df = clean_df.copy()
    mask = df[COL_MACHINE] == machine
    idx = df[mask].sort_values(COL_TIMESTAMP).index
    spiked = df.copy()
    # Add a large spike to the final 12 hours only.
    spiked.loc[idx[-12:], COL_TEMP] = spiked.loc[idx[-12:], COL_TEMP] + 40

    base = build_features(df, baseline_book)
    with_spike = build_features(spiked, baseline_book)
    a = base[base[COL_MACHINE] == machine]["baseline_temperature"].iloc[-1]
    b = with_spike[with_spike[COL_MACHINE] == machine]["baseline_temperature"].iloc[-1]
    assert a == pytest.approx(b)


def test_cold_start_machines_fall_back_to_line_baseline(features_df):
    for machine in ("MCH-300", "MCH-301"):
        g = features_df[features_df[COL_MACHINE] == machine]
        assert (g["baseline_source"] == "line_pooled").all()


def test_established_machines_use_their_own_baseline_eventually(features_df):
    g = features_df[features_df[COL_MACHINE] == "MCH-200"].sort_values(COL_TIMESTAMP)
    assert g["baseline_source"].iloc[0] == "line_pooled"
    assert g["baseline_source"].iloc[-1] == "machine"


def test_baseline_book_excludes_new_machines(clean_df: pd.DataFrame):
    """New machines must never define the fallback they themselves depend on."""
    book = fit_baseline_book(clean_df)
    only_new = clean_df[clean_df[COL_MACHINE].isin(["MCH-300", "MCH-301"])]
    book_without_new = fit_baseline_book(clean_df[~clean_df[COL_MACHINE].isin(["MCH-300", "MCH-301"])])
    assert book.to_dict() == book_without_new.to_dict()
    assert len(only_new) == 144


def test_bookkeeping_columns_are_not_model_features(features_df):
    cols = feature_columns(features_df)
    for banned in (COL_MACHINE, COL_TIMESTAMP, "failure_event", "hours_of_history", "baseline_source"):
        assert banned not in cols


def test_history_features_do_not_use_the_current_failure(clean_df: pd.DataFrame, features_df):
    """hours_since_last_failure at a failure hour refers to the PREVIOUS failure."""
    g = features_df[features_df[COL_MACHINE] == "MCH-202"].sort_values(COL_TIMESTAMP)
    fail_rows = g[g["failure_event"] == 1]
    assert len(fail_rows) == 2
    # At the first failure there is no prior failure at all.
    assert fail_rows["prior_failure_count"].iloc[0] == 0
    assert fail_rows["has_prior_failure"].iloc[0] == 0
    # At the second, exactly one prior failure is known.
    assert fail_rows["prior_failure_count"].iloc[1] == 1


# --------------------------------------------------------------------------
# Labelling
# --------------------------------------------------------------------------
def test_target_is_strictly_forward_looking(clean_df: pd.DataFrame):
    labelled = add_target(clean_df, 24)
    g = labelled[labelled[COL_MACHINE] == "MCH-200"].sort_values(COL_TIMESTAMP).reset_index(drop=True)
    fail_ix = g.index[g["failure_event"] == 1][0]
    # The 24 hours before the failure are positive.
    assert g.loc[fail_ix - 1, "failure_within_horizon"] == 1
    assert g.loc[fail_ix - 24, "failure_within_horizon"] == 1
    # 25 hours before is outside the window.
    assert g.loc[fail_ix - 25, "failure_within_horizon"] == 0
    # The failure hour itself does not count its own event.
    assert g.loc[fail_ix, "failure_within_horizon"] == 0


def test_positive_rate_grows_with_horizon(clean_df: pd.DataFrame):
    rates = []
    for h in (12, 24, 48, 72):
        lab = add_target(clean_df, h)
        elig = lab[lab["eligible"]]
        rates.append(elig["failure_within_horizon"].mean())
    assert rates == sorted(rates)
    assert rates[0] < 0.01


def test_failure_and_maintenance_rows_are_excluded(clean_df: pd.DataFrame):
    lab = add_target(clean_df, 72)
    assert not lab.loc[lab["failure_event"] == 1, "eligible"].any()
    assert not lab.loc[lab["preventive_maintenance"] == 1, "eligible"].any()


def test_preventive_maintenance_windows_are_censored(clean_df: pd.DataFrame):
    """Hours before planned maintenance are neither positive nor negative."""
    lab = add_target(clean_df, 72)
    prev = lab[lab["preventive_maintenance"] == 1]
    assert len(prev) == 5, "the dataset has five planned-maintenance resets"
    for _, row in prev.iterrows():
        window = lab[
            (lab[COL_MACHINE] == row[COL_MACHINE])
            & (lab[COL_TIMESTAMP] > row[COL_TIMESTAMP] - pd.Timedelta(hours=72))
            & (lab[COL_TIMESTAMP] < row[COL_TIMESTAMP])
        ]
        # Any censored hour with no real failure ahead of it is dropped.
        no_failure_ahead = window[window["failure_within_horizon"] == 0]
        assert not no_failure_ahead["eligible"].any()
