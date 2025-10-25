#!/usr/bin/env python3
"""
model_training_refactored.py

Refactored model training script for the Diabetes Risk Prediction Project.

Features:
- Multiple model options: logistic regression (baseline), Random Forest, Gradient Boosting, XGBoost (optional)
- Hyperparameter tuning via RandomizedSearchCV
- k-fold cross-validation (StratifiedKFold)
- Optional SMOTE for handling class imbalance (requires imbalanced-learn)
- Uses joblib.parallel_backend("threading") to avoid macOS `resource_tracker` errors when using parallel backends
- Saves best model and training metrics to disk under an output directory

Usage:
    python src/model_training_refactored.py --data ./data/diabetes.csv --model rf --out_dir ./reports --smote

Outputs:
- Trained model saved to: <out_dir>/models/<model_name>_best.joblib
- Search object saved to: <out_dir>/models/<model_name>_search.joblib (if search used)
- Metrics JSON saved to: <out_dir>/<model_name>_metrics.json

Notes:
- XGBoost and imbalanced-learn are optional dependencies. The script falls back gracefully if they are absent.
- To avoid macOS multiprocessing/resource_tracker tracebacks, the heavy parallel work is executed with the "threading" backend.
"""

from typing import Tuple, Dict, Any, Optional
import os
import argparse
import logging
import json

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score

# Use threading backend to avoid macOS resource_tracker errors when using loky/processes
from joblib import parallel_backend

# Optional imports
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

# Logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def load_data(path: str, target_col: str = "Outcome") -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load CSV dataset and split into X (features) and y (target).

    Args:
        path: path to CSV file
        target_col: name of the target column in the CSV

    Returns:
        X (DataFrame), y (Series)
    """
    df = pd.read_csv(path)
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in data.")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def build_pipeline(model_name: str = "logreg", use_scaler: bool = True, random_state: int = 42, use_smote: bool = False):
    """
    Build a scikit-learn pipeline for preprocessing + estimator.

    Args:
        model_name: one of "logreg", "rf", "gb", "xgb"
        use_scaler: whether to include StandardScaler
        random_state: random seed
        use_smote: whether to build an imbalanced-learn pipeline (SMOTE) if available

    Returns:
        pipeline (Pipeline or ImbPipeline), param_dist (dict)
    """
    steps = []
    if use_scaler:
        steps.append(("scaler", StandardScaler()))

    if model_name == "logreg":
        estimator = LogisticRegression(max_iter=1000, solver="liblinear", random_state=random_state)
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
        # Build imbalanced-learn pipeline where SMOTE is applied before estimator
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
    """
    Train and tune a model. Uses RandomizedSearchCV for hyperparameter tuning, evaluates on a held-out test set,
    and returns the trained model and metrics.

    Important: heavy parallel calls (search.fit and cross_val_score) are executed under joblib.parallel_backend("threading")
    to avoid macOS resource_tracker warnings while retaining parallelism benefits.

    Args:
        X: features DataFrame
        y: target Series
        model_name: model choice
        test_size: test split proportion
        random_state: random seed
        use_smote: whether to use SMOTE (requires imbalanced-learn)
        cv_folds: number of CV folds
        n_iter_search: number of iterations for RandomizedSearchCV
        out_dir: directory to save artifacts

    Returns:
        dictionary with model, metrics, model_path, and search_obj (if applicable)
    """
    os.makedirs(out_dir, exist_ok=True)
    models_dir = os.path.join(out_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    # Stratified split when possible
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

    # Fit with threading backend to avoid macOS loky/resource_tracker issues
    if search:
        logging.info("Starting hyperparameter search with threading backend...")
        with parallel_backend("threading"):
            search.fit(X_train, y_train)
        best = search.best_estimator_
        logging.info("Hyperparameter search completed. Best params: %s", search.best_params_)
        # Save the search object
        joblib.dump(search, os.path.join(models_dir, f"{model_name}_search.joblib"))
    else:
        logging.info("No parameter grid provided for model; fitting pipeline directly.")
        best = pipeline.fit(X_train, y_train)

    # Save best estimator
    model_path = os.path.join(models_dir, f"{model_name}_best.joblib")
    joblib.dump(best, model_path)
    logging.info("Saved best model to %s", model_path)

    # Predictions and probabilites (if available)
    preds = best.predict(X_test)
    probas = None
    try:
        probas = best.predict_proba(X_test)[:, 1]
    except Exception:
        logging.debug("predict_proba not available for this model/estimator.")

    # Basic metrics
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, probas)) if probas is not None else None,
        "f1": float(f1_score(y_test, preds, zero_division=0))
    }
    logging.info("Test ROC AUC: %s | F1: %s", metrics["roc_auc"], metrics["f1"])

    # Cross-validation metrics using threading backend
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

    # Save metrics to JSON
    metrics_path = os.path.join(out_dir, f"{model_name}_metrics.json")
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    logging.info("Saved metrics to %s", metrics_path)

    return {"model": best, "metrics": metrics, "model_path": model_path, "search_obj": search}


def parse_args():
    parser = argparse.ArgumentParser(description="Train models for Diabetes Risk Prediction")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV dataset")
    parser.add_argument("--model", type=str, default="rf", choices=["logreg", "rf", "gb", "xgb"], help="Model to train")
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
    # Print metrics summary for quick CLI view
    print(json.dumps(result["metrics"], indent=2))
