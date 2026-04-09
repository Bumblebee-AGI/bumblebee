"""Tests for gateway /health URL helpers."""

from bumblebee.utils.gateway_health_probe import health_url_from_base


def test_health_url_from_base_appends_path() -> None:
    assert health_url_from_base("https://bee.example.com") == "https://bee.example.com/health"


def test_health_url_from_base_preserves_existing_health() -> None:
    assert health_url_from_base("https://bee.example.com/health") == "https://bee.example.com/health"
