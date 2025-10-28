import os

import joblib
import numpy as np
import pandas as pd
import pytest

from src.model_evaluation import evaluate
from src.model_explainability_interpretability import permutation_importance_explain
from src.model_training import train_and_tune


def create_tiny_dataset(tmp_path):
    df = pd.DataFrame(
        {
            "feat1": np.random.RandomState(0).rand(40),
            "feat2": np.random.RandomState(1).rand(40),
            "Outcome": np.random.RandomState(2).randint(0, 2, 40),
        }
    )
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
    res = train_and_tune(
        X,
        y,
        model_name="logreg",
        test_size=0.25,
        random_state=1,
        use_smote=False,
        cv_folds=2,
        n_iter_search=1,
        out_dir=reports_dir,
    )
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
    perm_csv = permutation_importance_explain(
        model, X, y, out_dir=explain_dir, n_repeats=3
    )
    assert perm_csv is None or os.path.exists(perm_csv)


@pytest.mark.slow
def test_e2e_train_eval_explain():
    """
    Optionally run the main pipeline (quick mode) to ensure artifacts created.
    Marked slow - enable explicitly.
    """
    # This test assumes your training script supports a --quick flag (not required).
    # Adjust to suit your repository's entrypoints. Here we just check that scripts are present.
    assert os.path.exists("src/model_training.py") or os.path.exists(
        "src/model_training_refactored.py"
    )
    assert os.path.exists("src/model_evaluation.py")
    assert os.path.exists("src/model_explainability_interpretability.py")
