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
