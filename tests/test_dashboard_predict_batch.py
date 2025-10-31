"""
Unit tests for ModelWrapper.predict_batch()
"""

import pandas as pd

from dashboard.predict import ModelWrapper


def test_predict_batch_drops_outcome(saved_model_path):
    wrapper = ModelWrapper(saved_model_path)
    df = pd.DataFrame(
        {
            "Pregnancies": [1, 2],
            "Glucose": [85, 130],
            "BloodPressure": [66, 70],
            "SkinThickness": [29, 30],
            "Insulin": [0, 120],
            "BMI": [26.6, 32.5],
            "DiabetesPedigreeFunction": [0.351, 0.5],
            "Age": [22, 45],
            "Outcome": [0, 1],  # must be dropped by wrapper
        }
    )
    res = wrapper.predict_batch(df)
    assert res["n_rows"] == 2
    assert "mean_probability" in res
    assert 0.0 <= res["mean_probability"] <= 1.0
    # The ModelWrapper now includes explanation_files in the batch prediction result
    assert isinstance(res["explanation_files"], list)
