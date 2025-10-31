"""
Unit tests for ModelWrapper.predict_single()
"""

import pandas as pd

from dashboard.predict import ModelWrapper


def test_predict_single_basic(saved_model_path):
    wrapper = ModelWrapper(saved_model_path)
    # create a single-row DataFrame with expected features
    data = {
        "Pregnancies": 1,
        "Glucose": 120,
        "BloodPressure": 70,
        "SkinThickness": 20,
        "Insulin": 85,
        "BMI": 30.5,
        "DiabetesPedigreeFunction": 0.5,
        "Age": 35,
    }
    df = pd.DataFrame([data])
    res = wrapper.predict_single(df)
    assert "prediction" in res
    assert "probability" in res
    assert "user_message" in res
    assert isinstance(res["probability"], float)
    assert 0.0 <= res["probability"] <= 1.0
    # message contains percent
    assert "%" in res["user_message"]
