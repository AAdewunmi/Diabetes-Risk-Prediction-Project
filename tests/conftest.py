# tests/conftest.py
"""
pytest fixtures for Diabetes Risk Prediction Project tests.

Provides:
- tiny_dataset_path: path to a small CSV with a binary Outcome column
- tiny_df: the same dataset loaded as a pandas DataFrame
- reports_dir: temporary directory path for writing reports/artifacts
- trained_pipeline: a lightweight sklearn Pipeline (StandardScaler + LogisticRegression)
- saved_model_path: path to a joblib-dumped trained pipeline

These fixtures are intentionally small and deterministic so CI runs fast.
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import os
import sys
from pathlib import Path


# Add the project root (one level up from tests/) to sys.path so pytest can import `src`
ROOT = Path(__file__).resolve().parents[1]
ROOT_STR = str(ROOT)

if ROOT_STR not in sys.path:
    # Insert at position 0 so it takes precedence over installed site-packages
    sys.path.insert(0, ROOT_STR)

# Also export PYTHONPATH for any subprocesses spawned by tests (optional but handy)
os.environ.setdefault("PYTHONPATH", os.pathsep.join(filter(None, [os.environ.get("PYTHONPATH", ""), ROOT_STR])))


@pytest.fixture
def tiny_dataset_path(tmp_path: Path) -> str:
    """Create and return path to a tiny deterministic CSV dataset."""
    rng = np.random.RandomState(0)
    n = 40
    df = pd.DataFrame({
        "Pregnancies": rng.randint(0, 10, size=n),
        "Glucose": rng.normal(100, 15, size=n).round(1),
        "BloodPressure": rng.normal(70, 10, size=n).round(1),
        "SkinThickness": rng.normal(20, 5, size=n).round(1),
        "Insulin": rng.normal(80, 30, size=n).round(1),
        "BMI": (rng.normal(30, 5, size=n)).round(2),
        "DiabetesPedigreeFunction": (rng.rand(n) * 1).round(3),
        "Age": rng.randint(21, 80, size=n),
        "Outcome": rng.randint(0, 2, size=n)
    })
    path = tmp_path / "tiny_diabetes.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def tiny_df(tiny_dataset_path: str) -> pd.DataFrame:
    """Load the tiny CSV into a DataFrame and return it."""
    return pd.read_csv(tiny_dataset_path)


@pytest.fixture
def reports_dir(tmp_path: Path) -> str:
    """Directory for tests to write reports/artifacts into."""
    d = tmp_path / "reports"
    d.mkdir(exist_ok=True)
    return str(d)


@pytest.fixture
def trained_pipeline(tmp_path: Path, tiny_df: pd.DataFrame):
    """
    Train and return a lightweight sklearn Pipeline (StandardScaler + LogisticRegression).
    This is useful for evaluation/explainability tests that need a working model.
    """
    X = tiny_df.drop(columns=["Outcome"])
    y = tiny_df["Outcome"]
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(solver="liblinear", max_iter=1000, random_state=42))
    ])
    pipeline.fit(X, y)
    return pipeline


@pytest.fixture
def saved_model_path(trained_pipeline, tmp_path: Path) -> str:
    """Dump the trained_pipeline to a joblib file and return its path."""
    p = tmp_path / "model.joblib"
    joblib.dump(trained_pipeline, str(p))
    return str(p)
