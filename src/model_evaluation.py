#!/usr/bin/env python3
"""
src/model_evaluation.py

Evaluate a trained model and save evaluation artifacts.

Accepts either --model or --model_path for compatibility with older/refactored callers.
"""
import argparse
import logging
import os

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    m = joblib.load(path)
    logging.info("Loaded model from %s", path)
    return m


def evaluate(model, df: pd.DataFrame, out_dir: str = "./reports"):
    os.makedirs(out_dir, exist_ok=True)

    if "Outcome" not in df.columns:
        raise KeyError("Expected target column 'Outcome' in the dataset for evaluation.")

    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    preds = model.predict(X)
    try:
        probs = model.predict_proba(X)[:, 1]
    except Exception:
        probs = None

    # Confusion matrix
    cm = confusion_matrix(y, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    cm_path = os.path.join(out_dir, "confusion_matrix.png")
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()
    logging.info("Saved confusion matrix to %s", cm_path)

    # ROC curve if probabilities available
    if probs is not None:
        fpr, tpr, _ = roc_curve(y, probs)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        roc_path = os.path.join(out_dir, "roc_curve.png")
        plt.savefig(roc_path, bbox_inches="tight")
        plt.close()
        logging.info("Saved ROC curve to %s", roc_path)
    else:
        logging.warning("Model does not provide predict_proba; skipping ROC curve.")

    # classification report to text
    report = classification_report(y, preds)
    report_path = os.path.join(out_dir, "classification_report.txt")
    with open(report_path, "w") as fh:
        fh.write(report)
    logging.info("Saved classification report to %s", report_path)

    print("Classification Report:\n")
    print(report)
    logging.info("Evaluation artifacts saved to %s", out_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument("--model", type=str, help="Path to trained model (joblib/pkl)")
    parser.add_argument("--model_path", type=str, help="Alternative path to trained model (joblib/pkl)")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV dataset")
    parser.add_argument("--out_dir", type=str, default="reports/eval", help="Directory to save evaluation artifacts")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_path = args.model_path or args.model
    if model_path is None:
        logging.error("No model path provided. Use --model or --model_path.")
        raise SystemExit(2)

    model = load_model(model_path)
    df = pd.read_csv(args.data)
    evaluate(model, df, out_dir=args.out_dir)
