"""
src/dashboard/predict.py

ModelWrapper: loads a saved joblib/sklearn pipeline and exposes:
- get_expected_features()
- predict_single(df)
- predict_batch(df)

The methods return rich dictionaries used by the dashboard endpoints.

Notes:
- If the saved model cannot be found, the wrapper will attempt to locate the
  first '*_best.joblib' under 'reports/models/'.
- For explainability, this module will generate a minimal visualization PNG for
  single and batch predictions (simple bar/histogram) and save into 'reports/explain/'.
- Optional SHAP support is used if installed, but not required.
"""

from __future__ import annotations

import glob
import logging
import os
from typing import Any, Dict, List, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import artifact_name, list_files_with_mtime, safe_prepare_df

# Optional SHAP
try:
    import shap  # type: ignore

    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

LOGGER = logging.getLogger(__name__)
BASE_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REPORTS_EXPLAIN_DIR = os.path.join(BASE_REPO, "reports", "explain")
REPORTS_MODELS_DIR = os.path.join(BASE_REPO, "reports", "models")
os.makedirs(REPORTS_EXPLAIN_DIR, exist_ok=True)


class ModelWrapper:
    def __init__(self, model_path: Optional[str] = None):
        """
        Load model. If model_path not provided, try environment var DASHBOARD_MODEL,
        then look for reports/models/*_best.joblib.
        """
        path = model_path or os.environ.get("DASHBOARD_MODEL")
        if path is None:
            # search reports/models
            candidates = glob.glob(os.path.join(REPORTS_MODELS_DIR, "*_best.*"))
            path = candidates[0] if candidates else None
        if path is None or not os.path.exists(path):
            raise FileNotFoundError(
                "Could not find model file. Provide path or set DASHBOARD_MODEL."
            )
        self.model_path = os.path.abspath(path)
        self.model = joblib.load(self.model_path)
        # determine expected features
        self.expected_features = self._extract_feature_names()
        LOGGER.info(
            "Loaded model %s expecting features: %s",
            self.model_path,
            self.expected_features,
        )

    def _extract_feature_names(self) -> List[str]:
        """
        Try multiple heuristics to extract feature names the model was trained on:
        - model.feature_names_in_
        - if Pipeline, final estimator.feature_names_in_
        - fall back to a common diabetes feature set if unknown
        """
        # common fallback for this project
        fallback = [
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age",
        ]

        # direct attr
        try:
            fn = getattr(self.model, "feature_names_in_", None)
            if fn is not None:
                return list(fn)
        except Exception:
            pass

        # pipeline final estimator
        try:
            # pipeline: named_steps or steps
            if hasattr(self.model, "named_steps"):
                final = list(self.model.named_steps.values())[-1]
                fn = getattr(final, "feature_names_in_", None)
                if fn is not None:
                    return list(fn)
            # some estimators store feature names in coef_ shapes etc; not reliable
        except Exception:
            pass

        return fallback

    def get_expected_features(self) -> List[str]:
        return list(self.expected_features)

    def get_model_info(self) -> Dict[str, Any]:
        return {"estimator": type(self.model).__name__, "model_path": self.model_path}

    def predict_single(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Accepts a one-row DataFrame or DataFrame with one record.
        Returns a dict with keys:
          - prediction (int)
          - probability (float 0..1)
          - user_message (string)
          - explanation_files: list of {filename, mtime}
        """
        if df.shape[0] < 1:
            raise ValueError("Empty input for single prediction")
        expected = self.get_expected_features()
        df_prepped = safe_prepare_df(df.iloc[[0]], expected)

        # predict
        pred = None
        prob = None
        try:
            pred_arr = self.model.predict(df_prepped)
            pred = int(pred_arr[0])
        except Exception:
            pred = None
        try:
            prob_arr = self.model.predict_proba(df_prepped)[:, 1]
            prob = float(prob_arr[0])
        except Exception:
            prob = None

        # friendly message
        pct = (prob * 100) if prob is not None else None
        if pct is None:
            user_message = "The model could not compute a probability for this input."
        else:
            user_message = (
                f"Based on the details you shared, our model estimates there’s about "
                f"{pct:.2f}% chance you may be at risk of developing diabetes. "
                "This isn’t a medical diagnosis — consult a healthcare professional for personalised advice."
            )

        # produce a small bar chart png for this single prediction
        pngname = artifact_name("shap_single_pred", "png")
        pngpath = os.path.join(REPORTS_EXPLAIN_DIR, pngname)
        try:
            plt.figure(figsize=(4, 2))
            val = pct if pct is not None else 0.0
            plt.barh([0], [val], height=0.6)
            plt.xlim(0, 100)
            plt.xlabel("Risk (%)")
            plt.yticks([])
            plt.title("Predicted risk (%)")
            plt.tight_layout()
            plt.savefig(pngpath, bbox_inches="tight")
            plt.close()
        except Exception:
            LOGGER.exception("Failed to create single prediction PNG")
            pngname = None

        files = []
        if pngname:
            try:
                m = int(os.path.getmtime(os.path.join(REPORTS_EXPLAIN_DIR, pngname)))
                files.append({"filename": pngname, "mtime": m})
            except Exception:
                files.append({"filename": pngname, "mtime": 0})

        # optional SHAP explanation (best-effort)
        if SHAP_AVAILABLE:
            try:
                expl = shap.Explainer(self.model, df_prepped)
                sv = expl(df_prepped)
                htmlname = artifact_name("shap_single_force", "html")
                htmlpath = os.path.join(REPORTS_EXPLAIN_DIR, htmlname)
                # try to produce a self-contained html via shap (best-effort)
                try:
                    force = shap.plots.force(sv, matplotlib=False)
                    with open(htmlpath, "w", encoding="utf-8") as fh:
                        fh.write(force.html())
                    m = int(os.path.getmtime(htmlpath))
                    files.append({"filename": htmlname, "mtime": m})
                except Exception:
                    # fallback: save a simple text file
                    pass
            except Exception:
                LOGGER.debug("SHAP explain not produced (optional)")

        return {
            "prediction": pred,
            "probability": prob if prob is not None else 0.0,
            "user_message": user_message,
            "explanation_files": files,
            "model_info": self.get_model_info(),
        }

    def predict_batch(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Accepts a DataFrame (possibly with Outcome). Drops Outcome if present,
        prepares DataFrame to expected features, and returns summary dict:
         - n_rows, n_positive, mean_probability, hist_bins, hist_counts, explanation_files, model_info
        Also saves a histogram PNG under reports/explain/.
        """
        if df.shape[0] < 1:
            raise ValueError("Empty DataFrame uploaded")
        expected = self.get_expected_features()
        # drop Outcome if present and ensure expected columns
        if "Outcome" in df.columns:
            df = df.drop(columns=["Outcome"])

        df_prepped = safe_prepare_df(df, expected)

        # predict probabilities if possible
        probs = None
        preds = None
        try:
            probs = self.model.predict_proba(df_prepped)[:, 1]
            preds = (probs >= 0.5).astype(int)
        except Exception:
            try:
                preds = self.model.predict(df_prepped)
                probs = np.zeros_like(preds, dtype=float)
            except Exception:
                raise

        n_rows = int(len(df_prepped))
        n_positive = int(int(np.sum(preds)))
        mean_prob = float(float(np.mean(probs))) if len(probs) > 0 else 0.0

        # histogram
        counts, bins = np.histogram(probs, bins=10, range=(0.0, 1.0))
        bin_labels = [
            f"{int(b*100)}-{int(bins[i+1]*100)}%" for i, b in enumerate(bins[:-1])
        ]

        # save histogram png
        pngname = artifact_name("batch_pred_hist", "png")
        pngpath = os.path.join(REPORTS_EXPLAIN_DIR, pngname)
        try:
            plt.figure(figsize=(6, 3))
            plt.bar(range(len(counts)), counts)
            plt.xticks(range(len(counts)), bin_labels, rotation=45, ha="right")
            plt.ylabel("Count")
            plt.title("Prediction probability distribution")
            plt.tight_layout()
            plt.savefig(pngpath, bbox_inches="tight")
            plt.close()
        except Exception:
            LOGGER.exception("Failed to save batch histogram")
            pngname = None

        files = []
        if pngname:
            try:
                m = int(os.path.getmtime(os.path.join(REPORTS_EXPLAIN_DIR, pngname)))
                files.append({"filename": pngname, "mtime": m})
            except Exception:
                files.append({"filename": pngname, "mtime": 0})

        # produce permutation importance csv (best-effort, may be slow)
        try:
            from sklearn.inspection import permutation_importance

            r = permutation_importance(
                self.model,
                df_prepped,
                np.asarray(preds),
                n_repeats=5,
                random_state=42,
                n_jobs=1,
            )
            imp_df = pd.DataFrame(
                {
                    "feature": expected,
                    "importance_mean": r.importances_mean,
                    "importance_std": r.importances_std,
                }
            )
            csvname = artifact_name("permutation_importance", "csv")
            csvpath = os.path.join(REPORTS_EXPLAIN_DIR, csvname)
            imp_df.to_csv(csvpath, index=False)
            m = int(os.path.getmtime(csvpath))
            files.append({"filename": csvname, "mtime": m})
        except Exception:
            LOGGER.debug("Permutation importance not generated (optional)")

        return {
            "n_rows": n_rows,
            "n_positive": n_positive,
            "mean_probability": mean_prob,
            "hist_counts": counts.tolist(),
            "hist_bins": bin_labels,
            "explanation_files": files,
            "model_info": self.get_model_info(),
        }


def find_model() -> Optional[str]:
    """
    Return currently configured model path (env DASHBOARD_MODEL) or first candidate.
    """
    path = os.environ.get("DASHBOARD_MODEL")
    if path and os.path.exists(path):
        return os.path.abspath(path)
    candidates = glob.glob(os.path.join(REPORTS_MODELS_DIR, "*_best.*"))
    return candidates[0] if candidates else None


# Convenience for listing explain files
def list_explain_files() -> List[Dict[str, Any]]:
    return list_files_with_mtime(REPORTS_EXPLAIN_DIR)
