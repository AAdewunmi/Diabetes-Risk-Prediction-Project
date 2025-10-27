import os
import json
import joblib
import pandas as pd
import numpy as np
from src.model_training import train_and_tune, load_data
from src.model_evaluation import evaluate
from src.model_explainability_interpretability import permutation_importance_explain
import tempfile

def create_tiny_dataset(tmp_path):
    df = pd.DataFrame({
        "feat1": np.random.RandomState(0).rand(40),
        "feat2": np.random.RandomState(1).rand(40),
        "Outcome": np.random.RandomState(2).randint(0,2,40)
    })
    path = tmp_path / "tiny.csv"
    df.to_csv(path, index=False)
    return str(path)

def test_end_to_end(tmp_path):
    data_path = create_tiny_dataset(tmp_path)
    df = pd.read_csv(data_path)
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    reports_dir = str(tmp_path / "reports")
    # train logreg quickly
    res = train_and_tune(X, y,
                         model_name="logreg",
                         test_size=0.25,
                         random_state=1,
                         use_smote=False,
                         cv_folds=2,
                         n_iter_search=1,
                         out_dir=reports_dir)
    model_path = res["model_path"]
    assert os.path.exists(model_path)

    # evaluate
    model = joblib.load(model_path)
    eval_dir = os.path.join(reports_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    evaluate(model, df, out_dir=eval_dir)
    assert os.path.exists(os.path.join(eval_dir, "classification_report.txt"))

    # explainability: permutation importance should run
    explain_dir = os.path.join(reports_dir, "explain")
    os.makedirs(explain_dir, exist_ok=True)
    perm_csv = permutation_importance_explain(model, X, y, out_dir=explain_dir, n_repeats=3)
    assert perm_csv is None or os.path.exists(perm_csv)
