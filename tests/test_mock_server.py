"""
Integration Tests for the Stand-In Mock Service (OpenAI-compatible inference).
"""

import pytest
from fastapi.testclient import TestClient
from mock_service.mock_server import app


@pytest.fixture
def client():
    return TestClient(app)


def test_mock_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_mock_chat_completion_sync(client):
    payload = {
        "model": "mock-stand-in-model",
        "messages": [{"role": "user", "content": "What is self-attention?"}],
        "stream": False,
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "choices" in data
    assert len(data["choices"]) > 0
    assert len(data["choices"][0]["message"]["content"]) > 10


def test_mock_admin_configure_daily_budget_exhaustion(client):
    # Configure daily budget exhaustion
    config_payload = {"daily_budget_exhausted": True}
    cfg_resp = client.post("/admin/configure", json=config_payload)
    assert cfg_resp.status_code == 200

    # Next completion request should immediately return HTTP 429
    req_payload = {
        "model": "mock-stand-in-model",
        "messages": [{"role": "user", "content": "Explain attention"}],
    }
    lim_resp = client.post("/v1/chat/completions", json=req_payload)
    assert lim_resp.status_code == 429
    error_data = lim_resp.json()
    assert error_data["error"]["type"] == "daily_budget_exceeded"

    # Reset back to unexhausted
    client.post("/admin/configure", json={"daily_budget_exhausted": False, "reset_window": True})

