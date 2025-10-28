# tests/conftest.py
"""
Central pytest fixtures for the Diabetes-Risk-Prediction-Project tests.

Organization:
- Section A: Shared pipeline fixtures (data, tiny DataFrame, trained pipeline, saved model)
- Section B: Dashboard / Flask fixtures (app, client)
- Section C: Utility fixtures (temporary reports dir, sample CSV writer wrapper)

Fixture naming uses clear prefixes to avoid collisions:
 - pipeline_* for pipeline-related fixtures
 - dashboard_* for Flask/dashboard fixtures
 - tmp_* for temporary filesystem directories

Scope choices:
 - session: expensive or stable setup (tiny dataset file, trained pipeline)
 - function/module: per-test isolation where appropriate (test client, tmp directories)
"""

import os
import sys
from pathlib import Path
from typing import Generator

import joblib
import numpy as np
import pandas as pd
import pytest
from flask import Flask
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Ensure repository root (one level up from tests/) is importable so tests can import src.*
ROOT = Path(__file__).resolve().parents[1]
ROOT_STR = str(ROOT)
if ROOT_STR not in sys.path:
    sys.path.insert(0, ROOT_STR)

# Export PYTHONPATH to subprocesses (useful for tests that spawn subprocesses)
os.environ.setdefault(
    "PYTHONPATH",
    os.pathsep.join(filter(None, [os.environ.get("PYTHONPATH", ""), ROOT_STR])),
)

# -------------------------------
# Section A: Shared pipeline fixtures
# -------------------------------


@pytest.fixture(scope="session")
def pipeline_tiny_dataset_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    """
    Create a small deterministic CSV dataset used across pipeline tests.
    Session-scoped to avoid re-creating the file repeatedly.
    """
    rng = np.random.RandomState(0)
    n = 40
    df = pd.DataFrame(
        {
            "Pregnancies": rng.randint(0, 10, size=n),
            "Glucose": rng.normal(100, 15, size=n).round(1),
            "BloodPressure": rng.normal(70, 10, size=n).round(1),
            "SkinThickness": rng.normal(20, 5, size=n).round(1),
            "Insulin": rng.normal(80, 30, size=n).round(1),
            "BMI": (rng.normal(30, 5, size=n)).round(2),
            "DiabetesPedigreeFunction": (rng.rand(n) * 1).round(3),
            "Age": rng.randint(21, 80, size=n),
            "Outcome": rng.randint(0, 2, size=n),
        }
    )
    d = tmp_path_factory.mktemp("data")
    p = d / "tiny_diabetes.csv"
    df.to_csv(p, index=False)
    return str(p)


@pytest.fixture
def pipeline_tiny_df(pipeline_tiny_dataset_path: str) -> pd.DataFrame:
    """Load the tiny CSV into a DataFrame for tests that need it as a DataFrame."""
    return pd.read_csv(pipeline_tiny_dataset_path)


@pytest.fixture(scope="session")
def pipeline_trained_model(pipeline_tiny_df: pd.DataFrame) -> Pipeline:
    """
    Train and return a lightweight sklearn Pipeline (StandardScaler + LogisticRegression).
    Session scope because training is fast but deterministic and reused by many tests.
    """
    X = pipeline_tiny_df.drop(columns=["Outcome"])
    y = pipeline_tiny_df["Outcome"]
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(solver="liblinear", max_iter=1000, random_state=42),
            ),
        ]
    )
    pipeline.fit(X, y)
    return pipeline


@pytest.fixture(scope="session")
def pipeline_saved_model_path(
    pipeline_trained_model: Pipeline, tmp_path_factory: pytest.TempPathFactory
) -> str:
    """Dump the trained pipeline to a joblib file and return its path (session-scoped)."""
    d = tmp_path_factory.mktemp("models")
    p = d / "model.joblib"
    joblib.dump(pipeline_trained_model, str(p))
    return str(p)


# -------------------------------
# Section B: Dashboard / Flask fixtures
# -------------------------------


@pytest.fixture(scope="module")
def dashboard_app() -> "Flask":
    """
    Import and return the Flask app object from src/dashboard/app.py

    Scope: module (tests within a module can reuse the app instance).
    """
    # Ensure src/ is on sys.path (already inserted above) then import
    from dashboard.app import app as flask_app

    # Minimal test config overrides
    flask_app.config["TESTING"] = True
    flask_app.config["WTF_CSRF_ENABLED"] = False
    # Use an isolated upload folder under tmp for tests
    temp_uploads = Path(os.getenv("TMPDIR", "/tmp")) / "tests_dashboard_uploads"
    temp_uploads.mkdir(parents=True, exist_ok=True)
    flask_app.config.setdefault("UPLOAD_FOLDER", str(temp_uploads))
    return flask_app


@pytest.fixture
def dashboard_client(dashboard_app) -> Generator:
    """Provide Flask test client for route testing."""
    with dashboard_app.test_client() as client:
        yield client


# -------------------------------
# Section C: Utility / filesystem fixtures
# -------------------------------


@pytest.fixture
def tmp_reports_dir(tmp_path: Path) -> str:
    """A temporary reports directory for tests to write artifacts into."""
    d = tmp_path / "reports"
    d.mkdir(exist_ok=True)
    return str(d)


@pytest.fixture(scope="function")
def sample_csv_path(tmp_path: Path) -> str:
    """
    Small sample CSV for dashboard upload tests (function-scoped for isolation).
    Creates two rows matching dataset feature names (without Outcome).
    """
    data = {
        "Pregnancies": [1, 2],
        "Glucose": [85, 130],
        "BloodPressure": [66, 70],
        "SkinThickness": [29, 30],
        "Insulin": [0, 120],
        "BMI": [26.6, 32.5],
        "DiabetesPedigreeFunction": [0.351, 0.5],
        "Age": [22, 45],
    }
    df = pd.DataFrame(data)
    p = tmp_path / "sample_upload.csv"
    df.to_csv(p, index=False)
    return str(p)


@pytest.fixture
def client():
    # ensures the Flask app is importable
    import sys

    sys.path.insert(0, os.path.join(ROOT, "src"))
    from dashboard.app import app

    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# -------------------------------
# Small helper importable within tests
# -------------------------------
@pytest.fixture(scope="session")
def helpers_module():
    """
    Expose utility helpers (keeps heavy imports out of top-level test files).
    Tests can import from this fixture: e.g. use `helpers = request.getfixturevalue('helpers_module')`
    """
    import tests.helpers as helpers

    return helpers
