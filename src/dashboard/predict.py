"""
src/dashboard/predict.py

Model loading, single/batch prediction, and lightweight artifact generation.
The matplotlib usage is headless via Agg primitives (no GUI backend required).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

logger = logging.getLogger("dashboard.predict")

# ----- Paths (relative to repo layout: src/../reports) -----
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))  # .../src/dashboard
_SRC_ROOT = os.path.dirname(_THIS_DIR)  # .../src
_REPORTS_DIR = os.path.abspath(os.path.join(_SRC_ROOT, "..", "reports"))
REPORTS_EXPLAIN_DIR = os.path.join(_REPORTS_DIR, "explain")
REPORTS_MODELS_DIR = os.path.join(_REPORTS_DIR, "models")

os.makedirs(REPORTS_EXPLAIN_DIR, exist_ok=True)


def list_explain_files(
    extensions: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """
    List files in reports/explain with mtime (int seconds).
    """
    exts = (
        [".png", ".jpg", ".jpeg", ".gif", ".svg", ".html"]
        if not extensions
        else list(extensions)
    )
    out: List[Dict[str, Any]] = []
    if not os.path.isdir(REPORTS_EXPLAIN_DIR):
        return out
    for name in os.listdir(REPORTS_EXPLAIN_DIR):
        path = os.path.join(REPORTS_EXPLAIN_DIR, name)
        if not os.path.isfile(path):
            continue
        if not any(name.lower().endswith(ext) for ext in exts):
            continue
        try:
            mtime = int(os.path.getmtime(path))
        except Exception:
            mtime = 0
        out.append({"filename": name, "mtime": mtime})
    out.sort(key=lambda r: r["mtime"], reverse=True)
    return out


def find_model() -> Optional[str]:
    """
    Try to locate a .joblib model in reports/models.
    Preference: *_best.joblib, then first *.joblib.
    """
    if not os.path.isdir(REPORTS_MODELS_DIR):
        return None
    best: Optional[str] = None
    fallback: Optional[str] = None
    for name in sorted(os.listdir(REPORTS_MODELS_DIR)):
        if not name.lower().endswith(".joblib"):
            continue
        path = os.path.join(REPORTS_MODELS_DIR, name)
        if "_best" in name:
            best = path
            break
        if fallback is None:
            fallback = path
    return best or fallback


def _now_tag() -> str:
    """Timestamp tag for filenames (YYYYmmdd_HHMMSS)."""
    import datetime as _dt

    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _save_probability_bar(prob: float, out_path: str) -> None:
    """Save a simple horizontal bar indicating probability (0..1)."""
    fig = Figure(figsize=(4, 2))
    ax = fig.add_subplot(1, 1, 1)
    ax.barh(["Risk"], [prob * 100])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Probability (%)")
    ax.set_title("Estimated Risk")
    fig.tight_layout()
    # Canvas creation is implicit; just save
    fig.savefig(out_path)


def _save_probability_hist(
    counts: np.ndarray, edges: np.ndarray, out_path: str
) -> None:
    """Save a histogram figure from counts & bin edges."""
    fig = Figure(figsize=(6, 3))
    ax = fig.add_subplot(1, 1, 1)
    mids = (edges[:-1] + edges[1:]) / 2.0
    width = np.diff(edges)
    ax.bar(mids, counts, width=width, align="center")
    ax.set_xlabel("Probability")
    ax.set_ylabel("Count")
    ax.set_title("Batch Probability Distribution")
    fig.tight_layout()
    fig.savefig(out_path)


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce columns to numeric where possible."""
    return df.apply(pd.to_numeric)


@dataclass
class ModelWrapper:
    model_path: Optional[str] = None

    def __post_init__(self):
        path = self.model_path or os.environ.get("DASHBOARD_MODEL") or find_model()
        if not path or not os.path.exists(path):
            logger.warning("No model file found. Predictions will fail if called.")
            self.model = None
            self.feature_names_in_: Optional[List[str]] = None
            return
        self.model = joblib.load(path)
        # Try to capture feature order if available
        self.feature_names_in_ = getattr(self.model, "feature_names_in_", None)  # type: ignore[attr-defined]
        self.model_path = path

    # --------- helpers ---------
    def _validate_and_reorder(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure columns match training order if known.
        If feature_names_in_ is present, reorder and check.
        """
        if self.feature_names_in_ is None:
            return X
        X_cols = list(X.columns)
        expected = list(self.feature_names_in_)
        if set(X_cols) != set(expected):
            raise ValueError(
                "Feature mismatch: ensure you provide the same feature columns used during model training."
            )
        # Reorder to expected
        return X[expected]

    def get_model_info(self) -> Dict[str, Any]:
        est_name = type(self.model).__name__ if self.model is not None else None
        return {"estimator": est_name, "model_path": self.model_path}

    # --------- predictions ---------
    def predict_single(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Accept a one-row DataFrame or dict-like. Returns:
          {
            "prediction": int,
            "probability": float|None,
            "user_message": str,
            "explanation_files": [{"filename","mtime"}, ...]
          }
        """
        if isinstance(df, dict):
            df = pd.DataFrame([df])

        if not isinstance(df, pd.DataFrame):
            raise TypeError("predict_single expects a pandas DataFrame or dict")

        if df.shape[0] < 1:
            raise ValueError("Empty input")

        X = _coerce_numeric(df.copy())
        try:
            X = self._validate_and_reorder(X)
        except ValueError:
            raise

        if self.model is None:
            raise RuntimeError("No model loaded for predictions")

        try:
            preds = self.model.predict(X)
        except Exception as e:
            raise ValueError(
                "Feature mismatch: ensure you provide the same feature columns used during model training."
            ) from e

        prob_val: Optional[float] = None
        if hasattr(self.model, "predict_proba"):
            try:
                proba = self.model.predict_proba(X)
                if proba.ndim == 2 and proba.shape[1] >= 2:
                    prob_val = float(proba[0, 1])
                else:
                    # multiclass or unusual shape: take predicted class prob
                    idx = int(preds[0])
                    prob_val = float(proba[0, idx])
            except Exception:
                prob_val = None

        pred_int = int(np.asarray(preds).ravel()[0])
        user_msg = (
            f"Based on your inputs, estimated risk is {prob_val * 100:.2f}%."
            if prob_val is not None
            else "Prediction available, but probability could not be computed."
        )

        # Save a small artifact for the single prediction probability if available
        files: List[Dict[str, Any]] = []
        try:
            if prob_val is not None:
                fname = f"single_prob_{_now_tag()}.png"
                fpath = os.path.join(REPORTS_EXPLAIN_DIR, fname)
                _save_probability_bar(prob_val, fpath)
                files = list_explain_files()
        except Exception:
            logger.exception("Failed to save single prediction artifact")

        return {
            "prediction": pred_int,
            "probability": prob_val,
            "user_message": user_msg,
            "explanation_files": files,
        }

    def predict_batch(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Batch predict on DataFrame. Drops Outcome column if present.

        Returns:
          {
            "n_rows": int,
            "mean_probability": float|None,
            "histogram": {"counts": [...], "bin_edges": [...] } | None,
            "explanation_files": [{"filename","mtime"}, ...],
            "predictions": pd.DataFrame  # original columns + 'prediction' + 'probability'
          }
        """
        if self.model is None:
            raise RuntimeError("No model loaded for predictions")

        if not isinstance(df, pd.DataFrame):
            raise TypeError("predict_batch expects a pandas DataFrame")

        df_copy = df.copy()
        if "Outcome" in df_copy.columns:
            df_copy = df_copy.drop(columns=["Outcome"])

        df_copy = _coerce_numeric(df_copy)

        # validate & reorder
        try:
            df_copy = self._validate_and_reorder(df_copy)
        except ValueError:
            raise

        try:
            preds = self.model.predict(df_copy)
        except Exception:
            raise

        # probabilities if available
        prob_col = None
        if hasattr(self.model, "predict_proba"):
            try:
                proba = self.model.predict_proba(df_copy)
                if proba.ndim == 2 and proba.shape[1] == 2:
                    prob_col = proba[:, 1]
                else:
                    idxs = np.asarray(preds).ravel().astype(int)
                    prob_col = proba[np.arange(len(preds)), idxs]
            except Exception:
                prob_col = None

        out = df.copy()  # retain original columns
        out["prediction"] = np.asarray(preds).ravel().astype(int)
        if prob_col is not None:
            out["probability"] = np.asarray(prob_col).astype(float)
            mean_prob: Optional[float] = (
                float(np.nanmean(out["probability"].values)) if len(out) > 0 else None
            )
        else:
            out["probability"] = np.nan
            mean_prob = None

        histogram = None
        try:
            if prob_col is not None and len(out) > 0:
                counts, edges = np.histogram(
                    out["probability"].values, bins=10, range=(0.0, 1.0)
                )
                histogram = {"counts": counts.tolist(), "bin_edges": edges.tolist()}

                # save a batch histogram artifact
                fname = f"batch_hist_{_now_tag()}.png"
                fpath = os.path.join(REPORTS_EXPLAIN_DIR, fname)
                _save_probability_hist(counts, edges, fpath)
        except Exception:
            logger.exception("Failed to compute/save batch histogram")

        files = list_explain_files()

        return {
            "n_rows": int(len(out)),
            "mean_probability": mean_prob,
            "histogram": histogram,
            "explanation_files": files,
            "predictions": out,
        }
