from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.config import settings  # noqa: E402
from app.ml.features import build_features, clean_raw, feature_columns, fit_baseline_book  # noqa: E402

ARTIFACTS_PRESENT = (Path(settings.model_dir) / "model.joblib").exists()

requires_artifacts = pytest.mark.skipif(
    not ARTIFACTS_PRESENT,
    reason="Model artifacts not built. Run: python -m app.ml.training",
)


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    return pd.read_csv(settings.data_path)


@pytest.fixture(scope="session")
def clean_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    return clean_raw(raw_df)


@pytest.fixture(scope="session")
def baseline_book(clean_df: pd.DataFrame):
    return fit_baseline_book(clean_df)


@pytest.fixture(scope="session")
def features_df(clean_df: pd.DataFrame, baseline_book) -> pd.DataFrame:
    return build_features(clean_df, baseline_book)


@pytest.fixture(scope="session")
def feature_cols(features_df: pd.DataFrame) -> list:
    return feature_columns(features_df)


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        yield c
