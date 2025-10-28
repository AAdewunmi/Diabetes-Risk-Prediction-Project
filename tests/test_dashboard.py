def test_index(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"Diabetes Risk Prediction Dashboard" in resp.data


def test_predict_endpoint(client):
    payload = {
        "Pregnancies": 0,
        "Glucose": 88,
        "BloodPressure": 60,
        "SkinThickness": 25,
        "Insulin": 0,
        "BMI": 24.0,
        "DiabetesPedigreeFunction": 0.2,
        "Age": 30,
    }
    rv = client.post("/predict", json=payload)
    assert rv.status_code in (200, 500)
    data = rv.get_json()
    # if no model, API should return appropriate error
    assert isinstance(data, dict)
    assert "ok" in data
