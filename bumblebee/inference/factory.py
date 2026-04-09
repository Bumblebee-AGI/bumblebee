"""Build the active InferenceProvider from harness + environment."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from bumblebee.inference.openai_transport import OpenAICompatibleTransport
from bumblebee.inference.providers import LocalRuntimeProvider, RemoteGatewayProvider
from bumblebee.inference.protocol import InferenceProvider

if TYPE_CHECKING:
    from bumblebee.config import HarnessConfig


def _resolved_inference_provider_name(harness: HarnessConfig) -> str:
    p = (harness.inference.provider or "").strip().lower()
    if p in ("local", "remote_gateway"):
        return p
    mode = (harness.deployment.mode or "local").strip().lower()
    if mode == "hybrid_railway":
        return "remote_gateway"
    return "local"


def _inference_base_url(harness: HarnessConfig) -> str:
    u = (harness.inference.base_url or "").strip()
    if u:
        return u.rstrip("/")
    return harness.ollama.base_url.rstrip("/")


def _bearer_token(harness: HarnessConfig) -> str:
    env_name = harness.inference.api_key_env or "BUMBLEBEE_INFERENCE_GATEWAY_TOKEN"
    return (os.environ.get(env_name) or "").strip()


def build_inference_provider(harness: HarnessConfig) -> InferenceProvider:
    name = _resolved_inference_provider_name(harness)
    timeout = float(harness.inference.timeout or harness.ollama.timeout)
    retry_attempts = int(harness.ollama.retry_attempts)
    retry_delay = float(harness.ollama.retry_delay)
    base = _inference_base_url(harness)

    if name == "remote_gateway":
        token = _bearer_token(harness)
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
