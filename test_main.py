from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_predict():
    sample = {
        "Pclass": 3,
        "Sex": 0,
        "Age": 22.0,
        "Fare": 7.25
    }
    response = client.post("/predict", json=sample)
    assert response.status_code == 200
    assert "prediction" in response.json()
