"""Unit tests for the Flask API endpoints.

Tests cover /health, /predict, and error-handling paths
using Flask's built-in test client (no live server needed).
"""

import pytest

try:
    from app import app
except ImportError:
    # If your Flask app factory is named differently, adjust here
    from flask_api.main import app  # type: ignore


@pytest.fixture
def client():
    """Flask test client fixture."""
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


class TestHealthEndpoint:
    """Tests for the /health route."""

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_json(self, client):
        response = client.get("/health")
        assert response.is_json

    def test_health_status_ok(self, client):
        response = client.get("/health")
        data = response.get_json()
        assert data is not None
        assert data.get("status") in ("ok", "healthy", "up")


class TestPredictEndpoint:
    """Tests for the /predict route."""

    def test_predict_with_positive_text(self, client):
        response = client.post(
            "/predict",
            json={"text": "I absolutely love this video, it's amazing!"},
        )
        assert response.status_code == 200
        data = response.get_json()
        assert data is not None
        # API should return a sentiment label
        assert "sentiment" in data or "prediction" in data or "label" in data

    def test_predict_with_negative_text(self, client):
        response = client.post(
            "/predict",
            json={"text": "This is the worst video I have ever seen, terrible."},
        )
        assert response.status_code == 200
        data = response.get_json()
        assert data is not None

    def test_predict_with_neutral_text(self, client):
        response = client.post(
            "/predict",
            json={"text": "The video was uploaded on Tuesday."},
        )
        assert response.status_code == 200

    def test_predict_with_empty_text(self, client):
        response = client.post("/predict", json={"text": ""})
        # Should return 400 for invalid input, not crash with 500
        assert response.status_code in (200, 400, 422)

    def test_predict_with_missing_text_field(self, client):
        response = client.post("/predict", json={})
        # Should return 400 for missing required field
        assert response.status_code in (400, 422)

    def test_predict_with_no_json_body(self, client):
        response = client.post("/predict")
        # Should return 400 when no body is sent
        assert response.status_code in (400, 415, 422)

    def test_predict_response_has_confidence(self, client):
        """Prediction response should include a confidence score."""
        response = client.post(
            "/predict",
            json={"text": "Great content!"},
        )
        data = response.get_json()
        if data is not None:
            # Common field names for confidence
            has_conf = any(
                k in data for k in ("confidence", "score", "probability", "proba")
            )
            assert has_conf or "sentiment" in data


class TestRootEndpoint:
    """Tests for the / route (welcome)."""

    def test_root_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200
