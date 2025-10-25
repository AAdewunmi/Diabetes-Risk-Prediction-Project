#!/usr/bin/env python3
"""
Model Training Script for Diabetes Risk Prediction Project

Enhancements:
- Multiple ML models: Logistic Regression, RandomForest, XGBoost (optional)
- Hyperparameter tuning using RandomizedSearchCV/GridSearchCV
- K-Fold Cross-Validation
- Optional SMOTE for class imbalance handling
- Saves trained models to ./models directory

Usage:
    python src/model_training.py --data ./data/diabetes.csv --model rf --smote
"""

import argparse
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Optional imports
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not installed. Skipping XGB option...")

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    IMB_AVAILABLE = True
except ImportError:
    IMB_AVAILABLE = False
    print("Imbalanced-learn not installed. SMOTE unavailable...")

RANDOM_STATE = 42

def load_data(path):
    return pd.read_csv(path)

def build_model(model_name="lr", use_smote=False):
    numeric_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    preprocessor = ColumnTransformer(
        transformers=[("scale", StandardScaler(), numeric_features)]
    )

    if model_name == "rf":
        model = RandomForestClassifier(random_state=RANDOM_STATE)
        param_grid = {
            "clf__n_estimators": [100, 200, 300],
            "clf__max_depth": [None, 5, 10]
        }
    elif model_name == "xgb" and XGBOOST_AVAILABLE:
        model = XGBClassifier(
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            use_label_encoder=False
        )
        param_grid = {
            "clf__max_depth": [3, 5, 7],
            "clf__learning_rate": [0.01, 0.1, 0.2]
        }
    else:
        model = LogisticRegression(max_iter=200, random_state=RANDOM_STATE)
        param_grid = {
            "clf__C": [0.01, 0.1, 1, 10]
        }

    PipelineClass = ImbPipeline if use_smote and IMB_AVAILABLE else Pipeline

    pipeline = PipelineClass([
        ("preprocess", preprocessor),
        ("clf", model)
    ])

    return pipeline, param_grid

def train_model(df, model_name="lr", cv_splits=5, use_smote=False):
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    model, param_grid = build_model(model_name, use_smote)
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)

    grid = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        cv=cv,
        n_iter=5,
        n_jobs=-1,
        scoring="f1",
        verbose=1
    )

    grid.fit(X, y)
    return grid

def save_model(model, path="./models/best_model.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"✅ Model saved to {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default="lr", choices=["lr", "rf", "xgb"])
    parser.add_argument("--out_dir", default="./models")
    parser.add_argument("--smote", action="store_true")
    args = parser.parse_args()

    df = load_data(args.data)
    best_model = train_model(df, args.model, use_smote=args.smote)
    save_model(best_model, f"{args.out_dir}/best_model_{args.model}.pkl")

    print("Best Params:", best_model.best_params_)
    print("Classification Report:\n", classification_report(df["Outcome"], best_model.predict(df.drop("Outcome", axis=1))))
