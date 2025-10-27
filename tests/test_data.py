import pandas as pd

from src.model_explainability_interpretability import find_target_column
from src.model_training import load_data


def make_small_csv(path):
    df = pd.DataFrame({"A": [1, 2, 3, 4], "B": [0, 1, 0, 1], "Outcome": [0, 1, 0, 1]})
    df.to_csv(path, index=False)


def test_load_data_and_target_detection(tmp_path):
    p = tmp_path / "small.csv"
    make_small_csv(p)
    X, y = load_data(str(p), target_col="Outcome")
    assert "Outcome" not in X.columns
    assert len(X) == 4
    assert y.name == "Outcome"

    df = pd.read_csv(p)
    assert find_target_column(df) == "Outcome"
