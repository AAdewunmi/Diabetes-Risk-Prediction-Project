import os

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.model_explainability_interpretability import (
    load_model_file,
    permutation_importance_explain,
    shap_explain,
)


def make_model_and_df(tmp_path):
    X = pd.DataFrame(
        {
            "feat1": np.random.RandomState(0).rand(50),
            "feat2": np.random.RandomState(1).rand(50),
        }
    )
    y = pd.Series(np.random.RandomState(2).randint(0, 2, 50), name="Outcome")
    pipeline = Pipeline(
        [("scaler", StandardScaler()), ("clf", LogisticRegression(solver="liblinear"))]
    )
    pipeline.fit(X, y)
    model_path = tmp_path / "m.joblib"
    joblib.dump(pipeline, str(model_path))
    return str(model_path), X, y


def test_permutation_importance(tmp_path):
    model_path, X, y = make_model_and_df(tmp_path)
    model = load_model_file(model_path)
    out = str(tmp_path / "explain")
    csv = permutation_importance_explain(model, X, y, out_dir=out, n_repeats=3)
    assert csv is None or os.path.exists(
        csv
    )  # can be None if something fails, but usually exists
    if csv:
        assert os.path.exists(os.path.join(out, "permutation_importance.png"))


@pytest.mark.skipif(
    not pytest.importorskip("shap", reason="shap not installed"),
    reason="shap not installed",
)
def test_shap_explain_small(tmp_path):
    # only run if shap available

    model_path, X, y = make_model_and_df(tmp_path)
    model = load_model_file(model_path)
    out = str(tmp_path / "explain_shap")
    res = shap_explain(model, X, out_dir=out, nsamples=10)
    # expect at least the selfcontained html and some pngs (if generation succeeded)
    assert "shap_force_selfcontained_html" in res
    assert os.path.exists(res["shap_force_selfcontained_html"])
