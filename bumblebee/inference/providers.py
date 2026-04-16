"""Concrete inference providers."""

from __future__ import annotations

from typing import Any, AsyncIterator, Optional

import aiohttp

from bumblebee.inference.openai_transport import OpenAICompatibleTransport
from bumblebee.inference.types import ChatCompletionResult


class LocalRuntimeProvider:
    """Brain co-located with the body: OpenAI-compatible endpoint (e.g. local Ollama)."""

    def __init__(self, transport: OpenAICompatibleTransport) -> None:
        self._t = transport

    async def chat_completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        tools: Optional[list[dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        think: bool = False,
        stream: bool = False,
        num_ctx: int | None = None,
    ) -> ChatCompletionResult | AsyncIterator[str]:
        return await self._t.chat_completion(
            model,
            messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
            think=think,
            stream=stream,
            num_ctx=num_ctx,
        )

    async def embed(self, model: str, text: str) -> list[float]:
        return await self._t.embed(model, text)

    async def health(self) -> dict[str, Any]:
        try:
            models = await self._t.list_models()
            return {"ok": True, "backend": "openai_compatible", "models_count": len(models)}
        except Exception as e:
            return {"ok": False, "backend": "openai_compatible", "error": str(e)[:500]}

    async def list_models(self) -> list[str]:
        return await self._t.list_models()

    async def close(self) -> None:
        await self._t.close()

    async def ensure_models(self, *names: str) -> tuple[bool, list[str]]:
        return await self._t.ensure_models(*names)


class RemoteGatewayProvider:
    """Brain behind a narrow HTTP gateway (tunnel to home). Uses /v1 for inference; /health on gateway root."""

    def __init__(
        self,
        transport: OpenAICompatibleTransport,
        *,
        gateway_public_base: str,
        gateway_token: str,
        health_timeout: float = 10.0,
    ) -> None:
        self._t = transport
        self._gw_base = gateway_public_base.rstrip("/")
        self._token = gateway_token
        self._health_timeout = aiohttp.ClientTimeout(total=health_timeout)

    async def chat_completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        tools: Optional[list[dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        think: bool = False,
        stream: bool = False,
        num_ctx: int | None = None,
    ) -> ChatCompletionResult | AsyncIterator[str]:
        return await self._t.chat_completion(
            model,
            messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
            think=think,
            stream=stream,
            num_ctx=num_ctx,
        )

    async def embed(self, model: str, text: str) -> list[float]:
        return await self._t.embed(model, text)

    async def health(self) -> dict[str, Any]:
        headers: dict[str, str] = {}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        url = f"{self._gw_base}/health"
        try:
            async with aiohttp.ClientSession(timeout=self._health_timeout) as s:
                async with s.get(url, headers=headers) as resp:
                    text = await resp.text()
                    if resp.status >= 400:
                        return {
                            "ok": False,
                            "backend": "inference_gateway",
                            "status": resp.status,
                            "error": text[:300],
                        }
                    return {"ok": True, "backend": "inference_gateway", "status": resp.status}
        except Exception as e:
            return {"ok": False, "backend": "inference_gateway", "error": str(e)[:500]}

    async def list_models(self) -> list[str]:
        return await self._t.list_models()

    async def close(self) -> None:
        await self._t.close()

    async def ensure_models(self, *names: str) -> tuple[bool, list[str]]:
        return await self._t.ensure_models(*names)
