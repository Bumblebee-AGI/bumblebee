"""Gateway HTTP surface: allowlist and 404 behavior (requires ``pip install 'bumblebee[gateway]'``)."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from bumblebee.inference_gateway.app import create_app  # noqa: E402


@pytest.fixture
def client(monkeypatch) -> TestClient:
    monkeypatch.setenv("INFERENCE_GATEWAY_TOKEN", "test-secret-token")
    with TestClient(create_app()) as c:
        yield c


def test_unknown_path_json_404(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 404
    body = r.json()
    assert body["error"] == "not_found"
    assert "inference" in body["message"].lower()


def test_openapi_ui_not_exposed(client: TestClient) -> None:
    for path in ("/docs", "/redoc", "/openapi.json"):
        r = client.get(path)
        assert r.status_code == 404


def test_health_requires_bearer(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 401


def test_health_returns_json_with_token(monkeypatch) -> None:
    monkeypatch.setenv("INFERENCE_GATEWAY_TOKEN", "test-secret-token")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:9")
    with TestClient(create_app()) as tc:
        r = tc.get("/health", headers={"Authorization": "Bearer test-secret-token"})
    assert r.status_code == 200
    data = r.json()
    assert "ok" in data
