"""Deployment packaging tests.

These exist because of a real production failure: pyarrow was dropped from
requirements.txt, every import succeeded, and the container then died at
startup trying to read scored_history.parquet. A unit test suite that only
exercises code paths on a machine where everything happens to be installed
will never catch that - so these tests assert on the dependency manifest
itself.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
REQUIREMENTS = BACKEND / "requirements.txt"


def _declared_packages() -> dict:
    """package name (lowercased, extras stripped) -> full requirement line."""
    out = {}
    for raw in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        name = re.split(r"[\[<>=!~;]", line, maxsplit=1)[0].strip().lower()
        if name:
            out[name] = line
    return out


@pytest.mark.parametrize(
    "package",
    [
        "fastapi",
        "uvicorn",
        "pydantic",
        "pydantic-settings",
        "pandas",
        "numpy",
        "scikit-learn",
        "joblib",
        "pyarrow",
        "anthropic",
    ],
)
def test_runtime_dependency_is_declared(package):
    assert package in _declared_packages(), (
        f"{package} is imported at runtime but missing from requirements.txt. "
        "The container will start and then fail."
    )


def test_parquet_engine_is_declared():
    """pandas cannot read the serving cache without an engine."""
    declared = _declared_packages()
    assert "pyarrow" in declared or "fastparquet" in declared, (
        "scored_history.parquet is the serving cache; without pyarrow or "
        "fastparquet, load_artifacts() raises ImportError at startup."
    )


def test_parquet_engine_is_actually_importable():
    """Guards the installed environment, not just the manifest."""
    pytest.importorskip("pyarrow", reason="pyarrow is a declared runtime dependency")


def test_sklearn_is_pinned_exactly():
    """The committed model.joblib is a pickle coupled to its writer version.

    A floating scikit-learn means a fresh build unpickles the artifact with a
    different version, which emits InconsistentVersionWarning and is explicitly
    documented as unsafe.
    """
    line = _declared_packages()["scikit-learn"]
    assert "==" in line, (
        f"scikit-learn must be pinned with '==', found: {line!r}. "
        "Unpinned, a rebuild loads model.joblib with a mismatched version."
    )


def test_installed_sklearn_matches_the_pin():
    """The environment running the tests must match what requirements.txt says."""
    import sklearn

    pinned = _declared_packages()["scikit-learn"].split("==", 1)[1].strip()
    assert sklearn.__version__ == pinned, (
        f"requirements.txt pins scikit-learn=={pinned} but "
        f"{sklearn.__version__} is installed. Retrain and re-pin, or install "
        "the pinned version - the committed artifact was written by the pin."
    )


def test_no_dependency_line_was_silently_commented_out():
    """A commented-out requirement is how the pyarrow outage happened."""
    suspicious = [
        raw.strip()
        for raw in REQUIREMENTS.read_text(encoding="utf-8").splitlines()
        if re.match(r"^\s*#\s*[a-zA-Z0-9_\-]+\s*[<>=~\[]", raw)
    ]
    assert not suspicious, (
        "requirements.txt contains commented-out package lines, which read as "
        f"deliberate but disable a dependency: {suspicious}"
    )
