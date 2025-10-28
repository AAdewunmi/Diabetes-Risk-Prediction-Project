import os

import pytest

from dashboard.predict import ModelWrapper, find_model


def test_find_model_exists():
    m = find_model()
    # test only that function runs; if you don't have trained models this can return None
    assert m is None or os.path.exists(m)


@pytest.mark.skipif(not os.path.isdir("reports/models"), reason="No models available")
def test_wrapper_predict_single():
    w = ModelWrapper()
    # construct a row with numeric defaults (you may need to adjust keys to match your pipeline)
    row = {
        "Pregnancies": 1,
        "Glucose": 85,
        "BloodPressure": 66,
        "SkinThickness": 29,
        "Insulin": 0,
        "BMI": 26.6,
        "DiabetesPedigreeFunction": 0.351,
        "Age": 22,
    }
    result = w.predict_single(row)
    assert "prediction" in result
    assert isinstance(result["prediction"], int)
