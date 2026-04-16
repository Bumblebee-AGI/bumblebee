"""Inference factory: hosted OpenRouter / Venice presets."""

from __future__ import annotations

import pytest

from bumblebee.config import DeploymentSettings, HarnessConfig, InferenceSettings
from bumblebee.inference.factory import (
    build_inference_provider,
    effective_inference_provider_name,
    inference_bearer_key_env,
)
from bumblebee.inference.providers import LocalRuntimeProvider, RemoteGatewayProvider


def _harness(**inf_kwargs: object) -> HarnessConfig:
    return HarnessConfig(inference=InferenceSettings(**inf_kwargs))


def test_effective_provider_openrouter_explicit() -> None:
    h = _harness(provider="openrouter")
    assert effective_inference_provider_name(h) == "openrouter"


def test_effective_provider_hybrid_defaults_to_gateway() -> None:
    h = HarnessConfig(
        deployment=DeploymentSettings(mode="hybrid_railway"),
        inference=InferenceSettings(provider=""),
    )
    assert effective_inference_provider_name(h) == "remote_gateway"


def test_inference_bearer_key_env_presets() -> None:
    assert inference_bearer_key_env(_harness(provider="openrouter")) == "OPENROUTER_API_KEY"
    assert inference_bearer_key_env(_harness(provider="venice")) == "VENICE_API_KEY"


def test_inference_bearer_custom_api_key_env() -> None:
    h = _harness(provider="openrouter", api_key_env="MY_OPENROUTER")
    assert inference_bearer_key_env(h) == "MY_OPENROUTER"


def test_build_openrouter_default_base(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    p = build_inference_provider(_harness(provider="openrouter", base_url=""))
    assert isinstance(p, LocalRuntimeProvider)
    assert p._t.base_url == "https://openrouter.ai/api"  # noqa: SLF001


def test_build_openrouter_custom_base(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    p = build_inference_provider(
        _harness(provider="openrouter", base_url="https://example.com/custom")
    )
    assert p._t.base_url == "https://example.com/custom"  # noqa: SLF001


def test_build_venice_default_base(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VENICE_API_KEY", "x")
    p = build_inference_provider(_harness(provider="venice"))
    assert isinstance(p, LocalRuntimeProvider)
    assert p._t.base_url == "https://api.venice.ai/api"  # noqa: SLF001


def test_build_remote_gateway_class(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BUMBLEBEE_INFERENCE_GATEWAY_TOKEN", "tok")
    p = build_inference_provider(
        _harness(provider="remote_gateway", base_url="https://gw.example.com")
    )
    assert isinstance(p, RemoteGatewayProvider)
