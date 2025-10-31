# src/dashboard/predict.py
"""
Model wrapper utilities used by the Flask dashboard.

Responsibilities:
- locate a persisted model artifact (find_model)
- list explainability artifacts (list_explain_files)
- provide ModelWrapper with predict_single and predict_batch

Key behavior fixes (2025-10-31):
- enforce model's feature ordering when possible (uses feature_names_in_ if present)
- return predict_batch that includes 'n_rows', 'mean_probability' and 'predictions'
- predict_single produces a friendly 'user_message' and returns probabilities if available
- uses Agg backend for matplotlib (safe for server)
"""

from __future__ import annotations

import glob
import os
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

# ensure safe server-side plotting backend (prevents macOS GUI windows)
os.environ.setdefault("MPLBACKEND", "Agg")


def find_model() -> Optional[str]:
    """Locate most-recent model artifact or respect DASHBOARD_MODEL env var."""
    env_path = os.environ.get("DASHBOARD_MODEL")
    if env_path:
        if os.path.isabs(env_path) and os.path.exists(env_path):
            return os.path.abspath(env_path)
        alt = os.path.abspath(os.path.join(os.getcwd(), env_path))
        if os.path.exists(alt):
            return alt

    models_dir = os.path.join(os.getcwd(), "reports", "models")
    if not os.path.isdir(models_dir):
        return None

    patterns = ["*_best.joblib", "*_best.pkl", "*.joblib", "*.pkl"]
    matches = []
    for p in patterns:
        matches.extend(glob.glob(os.path.join(models_dir, p)))
    if not matches:
        return None
    matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return os.path.abspath(matches[0])


def list_explain_files(extensions: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Return explainability artifacts in reports/explain as:
      [{"filename": "...", "mtime": 12345678}, ...]
    newest-first.
    """
    if extensions is None:
        extensions = [".png", ".jpg", ".jpeg", ".gif", ".svg", ".html"]

    explain_dir = os.path.join(os.getcwd(), "reports", "explain")
    out: List[Dict[str, Any]] = []
    if not os.path.isdir(explain_dir):
        return out

    for name in os.listdir(explain_dir):
        path = os.path.join(explain_dir, name)
        if not os.path.isfile(path):
            continue
        if extensions and not any(name.lower().endswith(ext) for ext in extensions):
            continue
        try:
            mtime = int(os.path.getmtime(path))
        except Exception:
            mtime = 0
        out.append({"filename": name, "mtime": mtime})
    out.sort(key=lambda r: r["mtime"], reverse=True)
    return out


class ModelWrapper:
    """
    Wrapper for persisted sklearn model (joblib).

    Methods:
      - predict_single(dict|Series|DataFrame) -> dict
      - predict_batch(DataFrame) -> dict with 'n_rows','mean_probability','predictions'
    """

    def __init__(self, model_path: Optional[str] = None) -> None:
        self.model_path = model_path or find_model()
        self.model = None
        if self.model_path:
            self._load_model(self.model_path)

    def _load_model(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        self.model = joblib.load(path)

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "estimator": type(self.model).__name__ if self.model is not None else None,
            "model_path": self.model_path,
        }

    def _expected_feature_names(self) -> Optional[List[str]]:
        """
        Attempt to extract the model's expected feature order.
        sklearn estimators/pipelines usually expose 'feature_names_in_'.
        """
        if self.model is None:
            return None
        # pipeline may have attribute directly or on the final estimator
        if hasattr(self.model, "feature_names_in_"):
            return list(getattr(self.model, "feature_names_in_"))
        # try final estimator if pipeline
        try:
            final = getattr(self.model, "named_steps", None)
            if final:
                last = list(self.model.named_steps.values())[-1]
                if hasattr(last, "feature_names_in_"):
                    return list(getattr(last, "feature_names_in_"))
        except Exception:
            pass
        return None

    def _validate_and_reorder(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        If the loaded model exposes feature_names_in_, reorder columns to match.
        Raise a clear ValueError listing missing features.
        """
        expected = self._expected_feature_names()
        if expected is None:
            return X  # cannot validate

        missing = [c for c in expected if c not in X.columns]
        if missing:
            raise ValueError(
                f"Feature mismatch: missing columns required by model: {missing}. "
                "Ensure you provide the same features used during training."
            )
        # reorder to exactly expected order
        return X[expected]

    def _explain_files(self) -> List[Dict[str, Any]]:
        return list_explain_files()

    def predict_single(self, df: Any) -> Dict[str, Any]:
        """
        Accept dict / pd.Series / one-row DataFrame.
        Returns:
          {
            "prediction": int,
            "probability": float|None,
            "user_message": str,
            "model_info": {...},
            "explanation_files": [...]
          }
        """
        if self.model is None:
            raise RuntimeError("No model loaded for predictions")

        if isinstance(df, dict):
            X = pd.DataFrame([df])
        elif isinstance(df, pd.Series):
            X = df.to_frame().T
        elif isinstance(df, pd.DataFrame):
            X = df.copy()
        else:
            raise TypeError(
                "predict_single expects dict, pandas.Series, or pandas.DataFrame"
            )

        # drop Outcome if present
        if "Outcome" in X.columns:
            X = X.drop(columns=["Outcome"])

        # coerce numeric-like columns to numeric where reasonable
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")

        # validate and reorder according to model (if available)
        try:
            X = self._validate_and_reorder(X)
        except ValueError:
            # re-raise with friendly message
            raise

        try:
            preds = self.model.predict(X)
        except Exception:
            # bubble up clearly for the app to log and return 500
            raise

        prob = None
        if hasattr(self.model, "predict_proba"):
            try:
                proba = self.model.predict_proba(X)
                if proba.shape[1] == 2:
                    prob = float(proba[0, 1])
                else:
                    idx = int(np.asarray(preds).ravel()[0])
                    prob = float(proba[0, idx])
            except Exception:
                prob = None
        elif hasattr(self.model, "decision_function"):
            try:
                dfun = self.model.decision_function(X)
                prob = float(1 / (1 + np.exp(-float(dfun[0]))))
            except Exception:
                prob = None

        pred_val = int(np.asarray(preds).ravel()[0])
        if prob is None:
            user_msg = f"The model predicts class {pred_val}. Probability details are not available for this model."
        else:
            user_msg = (
                f"Based on the details you provided, the model estimates a {prob * 100:.2f}% chance of diabetes. "
                "This is not a medical diagnosis — please consult a healthcare professional for clinical advice."
            )

        return {
            "prediction": int(pred_val),
            "probability": float(prob) if prob is not None else None,
            "user_message": user_msg,
            "model_info": self.get_model_info(),
            "explanation_files": self._explain_files(),
        }

    def predict_batch(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Batch predict on a pandas DataFrame.

        Behavior & return value:
          - Drops "Outcome" column if present (we don't send ground-truth to the model).
          - Coerces feature columns to numeric where possible.
          - Validates & reorders columns to match the model's expected feature order.
          - Computes predictions and (if available) per-row probabilities.
          - Returns a dict:
              {
                "n_rows": int,
                "mean_probability": float | None,
                "predictions": pd.DataFrame,   # original columns + 'prediction' + 'probability'
                "explanation_files": [...]
              }

        Notes:
          - The returned DataFrame (predictions) keeps the original columns (including Outcome if present)
            and appends 'prediction' and 'probability'.
          - The Flask route should convert the DataFrame to JSON-serializable format (e.g. .to_dict(orient='records'))
            before returning it to clients.
        """
        if self.model is None:
            raise RuntimeError("No model loaded for predictions")

        if not isinstance(df, pd.DataFrame):
            raise TypeError("predict_batch expects a pandas DataFrame")

        # Work on a copy so caller's df is untouched
        df_copy = df.copy()

        # Drop Outcome if present (we never send ground-truth to the model)
        if "Outcome" in df_copy.columns:
            df_copy = df_copy.drop(columns=["Outcome"])

        # Coerce to numeric where possible (safeguard)
        for col in df_copy.columns:
            df_copy[col] = pd.to_numeric(df_copy[col], errors="coerce")

        # Validate & reorder features to match model's expected input
        try:
            df_copy = self._validate_and_reorder(df_copy)
        except ValueError:
            # re-raise so callers can handle the feature-mismatch case
            raise

        # Get predictions
        try:
            preds = self.model.predict(df_copy)
        except Exception:
            raise

        # Get probabilities if available
        prob_col = None
        if hasattr(self.model, "predict_proba"):
            try:
                proba = self.model.predict_proba(df_copy)
                if proba is not None:
                    # Binary case: column 1 is positive class probability
                    if proba.ndim == 2 and proba.shape[1] == 2:
                        prob_col = proba[:, 1]
                    else:
                        # Multiclass: probability of the predicted class for each row
                        idxs = np.asarray(preds).ravel().astype(int)
                        prob_col = proba[np.arange(len(preds)), idxs]
            except Exception:
                prob_col = None

        # Build output DataFrame (keep original columns including Outcome if provided)
        out = df.copy()
        out["prediction"] = np.asarray(preds).ravel().astype(int)

        # Ensure probability column is a pandas Series of floats or NaNs
        if prob_col is not None:
            prob_series = pd.Series(np.asarray(prob_col).astype(float), index=out.index)
            out["probability"] = prob_series
            # compute mean skipping NaNs
            try:
                mean_prob = (
                    float(np.nanmean(prob_series.values))
                    if len(prob_series) > 0
                    else None
                )
            except Exception:
                mean_prob = None
        else:
            # create NaN column so downstream code can always expect a numeric column
            out["probability"] = pd.Series([np.nan] * len(out), index=out.index)
            mean_prob = None

        result = {
            "n_rows": int(len(out)),
            "mean_probability": mean_prob,
            "predictions": out,
            "explanation_files": self._explain_files(),  # Include explanation files
        }
        return result
