import os
import json
import shutil
import pytest
import numpy as np
import pandas as pd
from src.model_training import build_pipeline, train_and_tune, load_data

def small_dataset(tmp_path):
    df = pd.DataFrame({
        "feat1": np.random.RandomState(0).rand(30),
        "feat2": np.random.RandomState(1).rand(30),
        "Outcome": np.random.RandomState(2).randint(0,2,30)
    })
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return str(path)

def test_build_pipeline_variants():
    for name in ("logreg", "rf", "gb"):
        pl, pdist = build_pipeline(model_name=name)
        assert pl is not None
        assert "estimator" in dict(pl.steps)
    # xgb may be unavailable; we only assert it raises ValueError if not available
    try:
        pl, pdist = build_pipeline(model_name="xgb")
        assert pl is not None
    except ValueError:
        # acceptable if xgboost not installed
        pass

@pytest.mark.slowish
def test_train_and_tune_quick(tmp_path):
    data_path = small_dataset(tmp_path)
    df = pd.read_csv(data_path)
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]
    out_dir = str(tmp_path / "reports")
    # tiny search: n_iter_search=1, cv_folds=2 to be speedy
    res = train_and_tune(X, y,
                         model_name="logreg",
                         test_size=0.2,
                         random_state=42,
                         use_smote=False,
                         cv_folds=2,
                         n_iter_search=1,
                         out_dir=out_dir)
    assert "model_path" in res
    assert os.path.exists(res["model_path"])
    metrics_path = os.path.join(out_dir, "logreg_metrics.json")
    assert os.path.exists(metrics_path)
    with open(metrics_path) as fh:
        metrics = json.load(fh)
    assert "f1" in metrics
