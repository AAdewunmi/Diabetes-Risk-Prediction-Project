"""
Integration tests for Flask endpoints (predict, predict_batch, api_explain_files).
"""


def test_predict_endpoint_single(client):
    payload = {
        "Pregnancies": 0,
        "Glucose": 110,
        "BloodPressure": 72,
        "SkinThickness": 20,
        "Insulin": 85,
        "BMI": 28.5,
        "DiabetesPedigreeFunction": 0.4,
        "Age": 30,
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    data = res.get_json()
    assert data["ok"] is True
    assert "result" in data
    assert "user_message" in data["result"]


def test_predict_batch_endpoint(client, sample_csv_path):
    # upload sample csv
    with open(sample_csv_path, "rb") as fh:
        data = {"file": (fh, "sample.csv")}
        res = client.post(
            "/predict_batch", data=data, content_type="multipart/form-data"
        )
    assert res.status_code == 200
    payload = res.get_json()
    assert payload["ok"] is True
    assert "result" in payload
    r = payload["result"]
    assert "n_rows" in r and r["n_rows"] > 0
    assert "mean_probability" in r


def test_api_explain_files_endpoint(client, sample_csv_path):
    # upload sample csv
    with open(sample_csv_path, "rb") as fh:
        data = {"file": (fh, "sample.csv")}
        res = client.post(
            "/api_explain_files", data=data, content_type="multipart/form-data"
        )
    assert res.status_code == 200
    payload = res.get_json()
    assert payload["ok"] is True
    assert "result" in payload
    r = payload["result"]
    assert "explanations" in r
    explanations = r["explanations"]
    assert isinstance(explanations, list)
    assert len(explanations) > 0
    # Check that each explanation has expected keys
    for exp in explanations:
        assert "row_index" in exp
        assert "feature_importances" in exp
        fi = exp["feature_importances"]
        assert isinstance(fi, dict)
        assert len(fi) > 0
        for feature, importance in fi.items():
            assert isinstance(feature, str)
            assert isinstance(importance, float)
            assert importance >= 0.0
        assert abs(sum(fi.values()) - 1.0) < 1e-6  # importances sum to 1
        assert "explanation" in exp
        assert isinstance(exp["explanation"], str)
        assert len(exp["explanation"]) > 0
        assert "feature" in exp
        assert isinstance(exp["feature"], str)
        assert len(exp["feature"]) > 0
        assert "value" in exp
        assert isinstance(exp["value"], (int, float))
        assert exp["value"] >= 0.0
        assert "contribution" in exp
        assert isinstance(exp["contribution"], float)
