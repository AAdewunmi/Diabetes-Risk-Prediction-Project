#!/usr/bin/env python3
"""
Model Explainability & Interpretability Script

Methods:
- SHAP Feature Importance (if shap installed)
- Permutation Feature Importance (fallback)
- Partial Dependence (optional)
- Saves explainability plots under ./reports/explainability

Usage:
    python src/model_explainability_interpretability.py --model ./models/best_model_lr.pkl --data ./data/diabetes.csv
"""

import argparse
import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

from sklearn.inspection import permutation_importance

try:
    import shap
    SHAP_AVAILABLE = True
    shap.initjs()
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not installed — skipping SHAP explainability")

def explain_model(model, df, out_dir="./reports/explainability"):
    os.makedirs(out_dir, exist_ok=True)

    X = df.drop("Outcome", axis=1)

    # ✅ SHAP (if available)
    if SHAP_AVAILABLE:
        explainer = shap.Explainer(model.predict, X)
        shap_values = explainer(X)

        shap.summary_plot(shap_values, X, show=False)
        plt.savefig(f"{out_dir}/shap_summary.png", bbox_inches="tight")
        plt.close()
        print("✅ SHAP summary saved")

    # ✅ Permutation feature importance (fallback)
    results = permutation_importance(model, X, df["Outcome"], scoring="f1", n_repeats=10)
    indices = np.argsort(results.importances_mean)[::-1]

    plt.figure(figsize=(10, 6))
    plt.bar(np.array(X.columns)[indices], results.importances_mean[indices])
    plt.title("Permutation Feature Importance")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/permutation_importance.png")
    plt.close()

    print("✅ Permutation feature importance saved")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out_dir", default="./reports/explainability")
    args = parser.parse_args()

    model = joblib.load(args.model)
    df = pd.read_csv(args.data)
    explain_model(model, df, args.out_dir)
