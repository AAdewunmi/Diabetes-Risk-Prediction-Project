#!/usr/bin/env python3
"""
src/model_training.py

Train models for the Diabetes Risk Prediction Project.

Features:
- Select model via CLI: logreg (default), rf, gb, xgb (xgboost optional)
- RandomizedSearchCV hyperparameter tuning
- Cross-validation and test evaluation
- Optional SMOTE (imbalanced-learn)
- Uses joblib.parallel_backend("threading") for heavy parallel jobs to reduce macOS resource-tracker noise
- Saves artifacts to <out_dir>/models and metrics to <out_dir>

Outputs:
- <out_dir>/models/<model_name>_best.joblib
- <out_dir>/models/<model_name>_search.joblib (if search used)
- <out_dir>/<model_name>_metrics.json
"""
from typing import Tuple, Dict, Any, Optional
import argparse
import json
import logging
import os

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score

from joblib import parallel_backend

# Optional
try:
    import xgboost as xgb  # type: ignore
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE  # type: ignore
    from imblearn.pipeline import Pipeline as ImbPipeline  # type: ignore
    IMBLEARN_AVAILABLE = True
except Exception:
    IMBLEARN_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def load_data(path: str, target_col: str = "Outcome") -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in data.")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def build_pipeline(model_name: str = "logreg", use_scaler: bool = True, random_state: int = 42, use_smote: bool = False):
    steps = []
    if use_scaler:
        steps.append(("scaler", StandardScaler()))

    if model_name == "logreg":
        estimator = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=random_state)
        param_dist = {
            "estimator__C": [0.01, 0.1, 1, 10],
            "estimator__penalty": ["l1", "l2"]
        }
    elif model_name == "rf":
        estimator = RandomForestClassifier(random_state=random_state, n_jobs=-1)
        param_dist = {
            "estimator__n_estimators": [100, 200, 300],
            "estimator__max_depth": [None, 5, 10],
            "estimator__min_samples_split": [2, 5]
        }
    elif model_name == "gb":
        estimator = GradientBoostingClassifier(random_state=random_state)
        param_dist = {
            "estimator__n_estimators": [100, 200],
            "estimator__learning_rate": [0.01, 0.1],
            "estimator__max_depth": [3, 5]
        }
    elif model_name == "xgb" and XGBOOST_AVAILABLE:
        estimator = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=random_state, n_jobs=-1)
        param_dist = {
            "estimator__n_estimators": [100, 200],
            "estimator__learning_rate": [0.01, 0.1],
            "estimator__max_depth": [3, 6]
        }
    else:
        raise ValueError(f"Unsupported model: {model_name} (XGBoost available: {XGBOOST_AVAILABLE})")

    steps.append(("estimator", estimator))

    if use_smote and IMBLEARN_AVAILABLE:
        pl = ImbPipeline(steps)
    else:
        pl = Pipeline(steps)

    return pl, param_dist


def train_and_tune(X: pd.DataFrame, y: pd.Series,
                   model_name: str = "logreg",
                   test_size: float = 0.2,
                   random_state: int = 42,
                   use_smote: bool = False,
                   cv_folds: int = 5,
                   n_iter_search: int = 20,
                   out_dir: str = "./reports") -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    models_dir = os.path.join(out_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    stratify_arg = y if len(np.unique(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                        random_state=random_state,
                                                        stratify=stratify_arg)
    logging.info("Train/test split: %s / %s", X_train.shape, X_test.shape)

    pipeline, param_dist = build_pipeline(model_name=model_name, random_state=random_state, use_smote=use_smote)

    search = None
    if param_dist:
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_dist,
            n_iter=min(n_iter_search, max(1, len(param_dist))),
            scoring="roc_auc",
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state),
            verbose=1,
            n_jobs=-1,
            random_state=random_state
        )

    if search:
        logging.info("Starting hyperparameter search with threading backend...")
        with parallel_backend("threading"):
            search.fit(X_train, y_train)
        best = search.best_estimator_
        logging.info("Hyperparameter search completed. Best params: %s", search.best_params_)
        joblib.dump(search, os.path.join(models_dir, f"{model_name}_search.joblib"))
    else:
        logging.info("No parameter grid provided for model; fitting pipeline directly.")
        best = pipeline.fit(X_train, y_train)

    model_path = os.path.join(models_dir, f"{model_name}_best.joblib")
    joblib.dump(best, model_path)
    logging.info("Saved best model to %s", model_path)

    preds = best.predict(X_test)
    probas = None
    try:
        probas = best.predict_proba(X_test)[:, 1]
    except Exception:
        logging.debug("predict_proba not available for this model/estimator.")

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, probas)) if probas is not None else None,
        "f1": float(f1_score(y_test, preds, zero_division=0))
    }
    logging.info("Test ROC AUC: %s | F1: %s", metrics["roc_auc"], metrics["f1"])

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    try:
        logging.info("Starting cross-validation scoring with threading backend...")
        with parallel_backend("threading"):
            cv_scores = cross_val_score(best, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        metrics["cv_roc_auc_mean"] = float(np.mean(cv_scores))
        metrics["cv_roc_auc_std"] = float(np.std(cv_scores))
        logging.info("Cross-val ROC AUC mean: %.4f (std %.4f)", metrics["cv_roc_auc_mean"], metrics["cv_roc_auc_std"])
    except Exception as e:
        logging.warning("Cross-val scoring failed: %s", e)

    metrics_path = os.path.join(out_dir, f"{model_name}_metrics.json")
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    logging.info("Saved metrics to %s", metrics_path)

    return {"model": best, "metrics": metrics, "model_path": model_path, "search_obj": search}


def parse_args():
    parser = argparse.ArgumentParser(description="Train models for Diabetes Risk Prediction")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV dataset")
    parser.add_argument("--model", type=str, default="logreg", choices=["logreg", "rf", "gb", "xgb"], help="Model to train (default: logreg)")
    parser.add_argument("--out_dir", type=str, default="reports", help="Directory to save models and metrics")
    parser.add_argument("--smote", action="store_true", help="Apply SMOTE to training data (requires imbalanced-learn)")
    parser.add_argument("--cv", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--n_iter", type=int, default=20, help="Number of iterations for RandomizedSearchCV")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        X, y = load_data(args.data)
    except Exception as e:
        logging.error("Failed to load data: %s", e)
        raise

    result = train_and_tune(X, y,
                            model_name=args.model,
                            test_size=0.2,
                            random_state=42,
                            use_smote=args.smote,
                            cv_folds=args.cv,
                            n_iter_search=args.n_iter,
                            out_dir=args.out_dir)

    logging.info("Training complete. Results: %s", result["metrics"])
    print(json.dumps(result["metrics"], indent=2))
