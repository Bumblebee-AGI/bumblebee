"""Build the active InferenceProvider from harness + environment."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from bumblebee.inference.openai_transport import OpenAICompatibleTransport
from bumblebee.inference.providers import LocalRuntimeProvider, RemoteGatewayProvider
from bumblebee.inference.protocol import InferenceProvider

if TYPE_CHECKING:
    from bumblebee.config import HarnessConfig

DEFAULT_GATEWAY_KEY_ENV = "BUMBLEBEE_INFERENCE_GATEWAY_TOKEN"

# OpenAI-compatible roots: transport appends /v1 (same convention as Ollama http://host:11434).
_MANAGED_REMOTE_PRESETS: dict[str, dict[str, str]] = {
    "openrouter": {
        "default_base": "https://openrouter.ai/api",
    },
    "venice": {
        "default_base": "https://api.venice.ai/api",
    },
}


def effective_inference_provider_name(harness: HarnessConfig) -> str:
    """Resolved provider from YAML + env: local | remote_gateway | openrouter | venice."""
    p = (harness.inference.provider or "").strip().lower()
    if p in ("local", "remote_gateway", "openrouter", "venice"):
        return p
    mode = (harness.deployment.mode or "local").strip().lower()
    if mode == "hybrid_railway":
        return "remote_gateway"
    return "local"


def _effective_bearer_env(harness: HarnessConfig, preset: str) -> str:
    v = (harness.inference.api_key_env or "").strip()
    if not v or v == DEFAULT_GATEWAY_KEY_ENV:
        return preset
    return v


def inference_bearer_key_env(harness: HarnessConfig) -> str:
    """Environment variable name that should hold the Bearer token for the active remote brain."""
    name = effective_inference_provider_name(harness)
    if name == "openrouter":
        return _effective_bearer_env(harness, "OPENROUTER_API_KEY")
    if name == "venice":
        return _effective_bearer_env(harness, "VENICE_API_KEY")
    if name == "remote_gateway":
        return (harness.inference.api_key_env or "").strip() or DEFAULT_GATEWAY_KEY_ENV
    return DEFAULT_GATEWAY_KEY_ENV


def _bearer_token_for_env(harness: HarnessConfig, env_name: str) -> str:
    return (os.environ.get(env_name) or "").strip()


def _inference_base_url(harness: HarnessConfig, *, provider_name: str) -> str:
    u = (harness.inference.base_url or "").strip()
    if u:
        return u.rstrip("/")
    if provider_name in _MANAGED_REMOTE_PRESETS:
        return _MANAGED_REMOTE_PRESETS[provider_name]["default_base"]
    return harness.ollama.base_url.rstrip("/")


def build_inference_provider(harness: HarnessConfig) -> InferenceProvider:
    name = effective_inference_provider_name(harness)
    timeout = float(harness.inference.timeout or harness.ollama.timeout)
    retry_attempts = int(harness.ollama.retry_attempts)
    retry_delay = float(harness.ollama.retry_delay)
    base = _inference_base_url(harness, provider_name=name)

    if name == "remote_gateway":
        env_name = inference_bearer_key_env(harness)
        token = _bearer_token_for_env(harness, env_name)
        extra = (
            {"Authorization": f"Bearer {token}"}
            if token
            else None
        )
        transport = OpenAICompatibleTransport(
            base,
            timeout=timeout,
            retry_attempts=retry_attempts,
            retry_delay=retry_delay,
            extra_headers=extra,
        )
        return RemoteGatewayProvider(
            transport,
            gateway_public_base=base,
            gateway_token=token,
            health_timeout=min(30.0, timeout),
        )

    if name in ("openrouter", "venice"):
        env_name = inference_bearer_key_env(harness)
        token = _bearer_token_for_env(harness, env_name)
        extra = (
            {"Authorization": f"Bearer {token}"}
            if token
            else None
        )
        transport = OpenAICompatibleTransport(
            base,
            timeout=timeout,
            retry_attempts=retry_attempts,
            retry_delay=retry_delay,
            extra_headers=extra,
        )
        return LocalRuntimeProvider(transport)

    transport = OpenAICompatibleTransport(
        base,
        timeout=timeout,
        retry_attempts=retry_attempts,
        retry_delay=retry_delay,
    )
    return LocalRuntimeProvider(transport)


async def validate_inference_models(
    provider: Any,
    reflex_model: str,
    deliberate_model: str,
    harness_default_deliberate: str,
) -> tuple[bool, list[str]]:
    fn = getattr(provider, "ensure_models", None)
    if not callable(fn):
        return True, []
    return await fn(reflex_model, deliberate_model, harness_default_deliberate)
