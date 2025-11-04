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


def test_api_explain_files_endpoint(client):
    """
    Validate the /api_explain_files endpoint contract.

    The endpoint is a GET and returns JSON:
      {
        "ok": True,
        "files": [{"filename": "<name>", "mtime": <int>}, ...],
        "latest": {"filename": "<name>", "mtime": <int>} | None
      }

    This test is CI-safe: it does not assume any explain files exist.
    If files are present, it lightly validates structure and coherence.
    """
    res = client.get("/api_explain_files")
    assert res.status_code == 200

    payload = res.get_json()
    assert isinstance(payload, dict)
    assert payload.get("ok") is True
    assert "files" in payload

    files = payload["files"]
    assert isinstance(files, list)

    # When files exist, validate structure of each entry.
    for item in files:
        assert isinstance(item, dict)
        assert "filename" in item and isinstance(item["filename"], str)
        assert "mtime" in item and isinstance(item["mtime"], int)
        assert item["mtime"] >= 0

    # latest can be None or a dict mirroring file entries
    latest = payload.get("latest")
    if latest is not None:
        assert isinstance(latest, dict)
        assert "filename" in latest and isinstance(latest["filename"], str)
        assert "mtime" in latest and isinstance(latest["mtime"], int)
        assert latest["mtime"] >= 0

        # If files list is non-empty, latest should correspond to an item in files
        if files:
            names = {f["filename"] for f in files}
            assert latest["filename"] in names
