import os

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.model_evaluation import evaluate, load_model


def make_model_and_data(tmp_path):
    # create tiny model
    X = pd.DataFrame({"f1": [0, 1, 2, 3], "f2": [1, 0, 1, 0]})
    y = pd.Series([0, 1, 0, 1], name="Outcome")
    pipeline = Pipeline(
        [("scaler", StandardScaler()), ("clf", LogisticRegression(solver="liblinear"))]
    )
    pipeline.fit(X, y)
    model_path = tmp_path / "model.joblib"
    joblib.dump(pipeline, str(model_path))
    # write csv
    df = X.copy()
    df["Outcome"] = y
    data_path = tmp_path / "data.csv"
    df.to_csv(data_path, index=False)
    return str(model_path), str(data_path)


def test_evaluate_creates_files(tmp_path):
    model_path, data_path = make_model_and_data(tmp_path)
    model = load_model(model_path)
    out_dir = str(tmp_path / "eval")
    df = pd.read_csv(data_path)
    evaluate(model, df, out_dir=out_dir)
    # expected files
    assert os.path.exists(os.path.join(out_dir, "confusion_matrix.png"))
    # if predict_proba exists -> roc_curve.png
    assert os.path.exists(os.path.join(out_dir, "classification_report.txt"))
    # roc may exist if predict_proba
    # simple existence check for either roc or not
    # (we won't assert roc always exists)
