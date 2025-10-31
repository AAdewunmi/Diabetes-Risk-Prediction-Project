"""
tests/conftest.py

Unified fixtures for pipeline and dashboard tests.

Fixtures:
- tiny_dataset_path (session) -> path to small CSV
- tiny_df -> DataFrame loaded from the tiny CSV
- trained_pipeline (session) -> sklearn Pipeline (StandardScaler + LogisticRegression)
- saved_model_path (session) -> joblib path to saved trained pipeline
- sample_csv_path (function) -> small CSV for upload tests
- client (function) -> Flask test client (uses dashboard_app)
- dashboard_app (module) -> Flask app instance
"""

import os
import sys
from pathlib import Path

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


@pytest.fixture(scope="session")
def tiny_dataset_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Creates a small CSV dataset for quick training/testing."""
    rng = np.random.RandomState(0)
    n = 40
    df = pd.DataFrame(
        {
            "Pregnancies": rng.randint(0, 10, n),
            "Glucose": rng.randint(70, 200, n),
            "BloodPressure": rng.randint(50, 100, n),
            "SkinThickness": rng.randint(10, 50, n),
            "Insulin": rng.randint(0, 300, n),
            "BMI": rng.uniform(18.0, 45.0, n),
            "DiabetesPedigreeFunction": rng.uniform(0.05, 1.5, n),
            "Age": rng.randint(20, 70, n),
            "Outcome": rng.randint(0, 2, n),
        }
    )
    p = tmp_path_factory.mktemp("data") / "tiny_data.csv"
    df.to_csv(p, index=False)
    return str(p)


@pytest.fixture(scope="session")
def tiny_df(tiny_dataset_path: str) -> pd.DataFrame:
    """Loads the tiny CSV dataset into a DataFrame."""
    return pd.read_csv(tiny_dataset_path)


@pytest.fixture(scope="session")
def trained_pipeline(tiny_df: pd.DataFrame) -> Pipeline:
    """Trains a simple pipeline for session reuse."""
    X = tiny_df.drop("Outcome", axis=1)
    y = tiny_df["Outcome"]
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(random_state=0, max_iter=1000)),
        ]
    )
    pipeline.fit(X, y)
    return pipeline


@pytest.fixture(scope="session")
def saved_model_path(
    tmp_path_factory: pytest.TempPathFactory, trained_pipeline: Pipeline
) -> str:
    """Saves the trained pipeline to a temporary file path."""
    p = tmp_path_factory.mktemp("models") / "model_v1.joblib"
    joblib.dump(trained_pipeline, p)
    return str(p)


@pytest.fixture
def temp_reports_dir(tmp_path: Path) -> str:
    """A temporary reports directory for tests to write artifacts into."""
    d = tmp_path / "reports"
    d.mkdir(exist_ok=True)
    return str(d)


# NOTE: Keeping the function-scoped version of sample_csv_path
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


@pytest.fixture(scope="module")
def dashboard_app() -> "Flask":
    """
    Import and return the Flask app object from src/dashboard/app.py.
    The Flask object is needed for creating the client.
    """
    # ensure src import works
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from dashboard.app import app

    return app


# NOTE: Keeping the simpler version of client
@pytest.fixture
def client(saved_model_path: str, dashboard_app: Flask):
    """
    Creates a Flask test client configured for the dashboard app.
    It injects the path to the saved model as an environment variable.
    """
    # Set the model path environment variable before creating the client
    os.environ["DASHBOARD_MODEL"] = saved_model_path

    dashboard_app.config["TESTING"] = True

    with dashboard_app.test_client() as c:
        yield c

    # Clean up the environment variable
    del os.environ["DASHBOARD_MODEL"]


# -------------------------------
# Small helper import

# We explicitly need the Flask import above to satisfy the type hint
# def dashboard_app() -> "Flask":
# The use of the string "Flask" prevents a circular import in other fixtures.
