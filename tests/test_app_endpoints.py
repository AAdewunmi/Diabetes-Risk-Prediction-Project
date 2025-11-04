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


# tests/test_app_endpoints.py
def test_api_explain_files_endpoint(client):
    """
    Contract test for /api_explain_files.

    If the endpoint isn't enabled in this build (returns 404), the test skips.
    If present (200), it validates the schema:
      {
        "ok": True,
        "files": [{"filename": str, "mtime": int}, ...],
        "latest": {"filename": str, "mtime": int} | None
      }
    """
    res = client.get("/api_explain_files")

    # Allow builds that don't expose the endpoint
    if res.status_code == 404:
        import pytest

        pytest.skip("api_explain_files not enabled in this build")

    # Follow simple redirects if any (rare in local/CI)
    if res.status_code in (301, 302, 307, 308):
        location = res.headers.get("Location")
        assert location, "Redirect without Location header"
        res = client.get(location)

    assert res.status_code == 200
    payload = res.get_json()
    assert isinstance(payload, dict)
    assert payload.get("ok") is True
    assert "files" in payload

    files = payload["files"]
    assert isinstance(files, list)

    # Validate file entries if any exist
    for item in files:
        assert isinstance(item, dict)
        assert "filename" in item and isinstance(item["filename"], str)
        assert "mtime" in item and isinstance(item["mtime"], int)
        assert item["mtime"] >= 0

    latest = payload.get("latest")
    if latest is not None:
        assert isinstance(latest, dict)
        assert "filename" in latest and isinstance(latest["filename"], str)
        assert "mtime" in latest and isinstance(latest["mtime"], int)
        assert latest["mtime"] >= 0
        if files:
            names = {f["filename"] for f in files}
            assert latest["filename"] in names
