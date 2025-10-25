#!/usr/bin/env python3
"""
Model Evaluation Script

Outputs:
- Classification Report
- Confusion Matrix
- ROC Curve + AUC
- Saves plots under ./reports

Usage:
    python src/model_evaluation.py --model ./models/best_model_lr.pkl --data ./data/diabetes.csv
"""

import argparse
import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

def load_model(path):
    return joblib.load(path)

def evaluate(model, df, out_dir="./reports"):
    os.makedirs(out_dir, exist_ok=True)

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]

    # Confusion Matrix
    cm = confusion_matrix(y, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig(f"{out_dir}/confusion_matrix.png")
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y, probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.savefig(f"{out_dir}/roc_curve.png")
    plt.close()

    print("Classification Report:\n")
    print(classification_report(y, preds))

    print(f"✅ Reports saved to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out_dir", default="./reports")
    args = parser.parse_args()

    model = load_model(args.model)
    df = pd.read_csv(args.data)
    evaluate(model, df, args.out_dir)
