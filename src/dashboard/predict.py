"""
src/dashboard/predict.py

Lightweight wrapper to load the trained model and run predictions.
This isolates model handling for tests and the Flask app.
"""

from __future__ import annotations

import glob
import logging
import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def find_model(
    models_dir: str = "reports/models", preferred: Optional[str] = None
) -> Optional[str]:
    """
    Find the first matching model artifact. If preferred is given, prefer that file.
    """
    if preferred and os.path.exists(preferred):
        return preferred

    if not os.path.isdir(models_dir):
        return None

    # try explicit patterns
    patterns = [
        os.path.join(models_dir, "*_best.joblib"),
        os.path.join(models_dir, "*_best.pkl"),
        os.path.join(models_dir, "*.joblib"),
        os.path.join(models_dir, "*.pkl"),
    ]
    for pat in patterns:
        matches = glob.glob(pat)
        if matches:
            return matches[0]
    return None


class ModelWrapper:
    """
    Responsible for loading the model artifact and exposing predict/predict_proba methods.
    Accepts sklearn Pipelines and raw estimators saved with joblib.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or find_model()
        if self.model_path is None:
            raise FileNotFoundError("No model artifact found in reports/models/")
        logger.info("Loading model from %s", self.model_path)
        self.model = joblib.load(self.model_path)

    def predict_single(self, X: pd.DataFrame) -> dict:
        """
        Predict a single-row DataFrame (or 1d array-like converted to DataFrame).
        Returns {"prediction": int, "probability": float}
        """
        if isinstance(X, (list, tuple, np.ndarray)):
            X = pd.DataFrame([X])
        elif isinstance(X, dict):
            X = pd.DataFrame([X])
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame, dict, list or array")

        pred = int(self.model.predict(X)[0])
        prob = None
        try:
            prob = float(self.model.predict_proba(X)[0, 1])
        except Exception:
            # some models don't support predict_proba
            prob = None
        return {"prediction": pred, "probability": prob}

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict on a DataFrame and return the DataFrame with added columns:
        'prediction' and 'probability' (if available).
        """
        df_copy = df.copy()
        preds = self.model.predict(df_copy)
        df_copy["prediction"] = preds
        try:
            probs = self.model.predict_proba(df_copy)[:, 1]
            df_copy["probability"] = probs
        except Exception:
            df_copy["probability"] = pd.NA
        return df_copy

    def get_model_info(self) -> dict:
        """
        Return meta info about loaded model (path, class name).
        """
        return {"model_path": self.model_path, "estimator": type(self.model).__name__}
