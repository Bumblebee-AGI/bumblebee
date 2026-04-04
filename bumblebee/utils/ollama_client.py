"""Direct aiohttp client for Ollama OpenAI-compatible API (no openai SDK)."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional

import aiohttp

from bumblebee.cognition import gemma


@dataclass
class ToolCallSpec:
    name: str
    arguments: dict[str, Any]
    id: str = ""


@dataclass
class ChatCompletionResult:
    content: str
    thinking: Optional[str] = None
    tool_calls: list[ToolCallSpec] = field(default_factory=list)
    finish_reason: Optional[str] = None
    raw_message: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, int] = field(default_factory=dict)
    raw_assistant_text: str = ""


class OllamaClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
        retry_attempts: int = 3,
        retry_delay: float = 2.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api = f"{self.base_url}/v1"
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self._session: aiohttp.ClientSession | None = None

    async def _session_get(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        session = await self._session_get()
        url = f"{self.api}{path}"
        last_err: Exception | None = None
        for attempt in range(self.retry_attempts):
            try:
                async with session.post(url, json=payload) as resp:
                    text = await resp.text()
                    if resp.status >= 400:
                        raise aiohttp.ClientResponseError(
                            resp.request_info,
                            resp.history,
                            status=resp.status,
                            message=text[:500],
                        )
                    return json.loads(text) if text else {}
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_err = e
                if attempt + 1 < self.retry_attempts:
                    await asyncio.sleep(self.retry_delay * (2**attempt))
        raise last_err or RuntimeError("request failed")

    async def list_models(self) -> list[str]:
        session = await self._session_get()
        url = f"{self.api}/models"
        last_err: Exception | None = None
        for attempt in range(self.retry_attempts):
            try:
                async with session.get(url) as resp:
                    text = await resp.text()
                    if resp.status >= 400:
                        raise aiohttp.ClientResponseError(
                            resp.request_info,
                            resp.history,
                            status=resp.status,
                            message=text[:500],
                        )
                    data = json.loads(text) if text else {}
                    models = data.get("data") or []
                    return [m.get("id", "") for m in models if m.get("id")]
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_err = e
                if attempt + 1 < self.retry_attempts:
                    await asyncio.sleep(self.retry_delay * (2**attempt))
        raise last_err or RuntimeError("list_models failed")

    async def ensure_models(self, *names: str) -> tuple[bool, list[str]]:
        available = set(await self.list_models())
        missing = [n for n in names if n and n not in available]
        return (len(missing) == 0, missing)

    async def chat_completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        tools: Optional[list[dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        think: bool = False,
        stream: bool = False,
    ) -> ChatCompletionResult | AsyncIterator[str]:
        sys_parts: list[str] = []
        rest: list[dict[str, Any]] = []
        for m in messages:
            if m.get("role") == "system":
                c = m.get("content") or ""
                if isinstance(c, list):
                    c = gemma.stringify_content_blocks(c)
                sys_parts.append(str(c))
            else:
                rest.append(m)
        system_merged = "\n\n".join(sys_parts) if sys_parts else None
        ollama_messages: list[dict[str, Any]] = []
        if system_merged:
            inner = gemma.inject_thinking_instruction(system_merged, think)
            if not think:
                inner = inner + "\n" + gemma.empty_thought_channel_fragment()
            ollama_messages.append({"role": "system", "content": inner})
        ollama_messages.extend(rest)

        payload: dict[str, Any] = {
            "model": model,
            "messages": ollama_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if tools:
            payload["tools"] = tools

        if stream:
            return self._chat_stream(payload)
        data = await self._post_json("/chat/completions", payload)
        return self._parse_chat_response(data)

    async def _chat_stream(self, payload: dict[str, Any]) -> AsyncIterator[str]:
        session = await self._session_get()
        url = f"{self.api}/chat/completions"
        payload = {**payload, "stream": True}
        async with session.post(url, json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.content:
                if not line:
                    continue
                s = line.decode("utf-8", errors="replace").strip()
                if not s.startswith("data:"):
                    continue
                chunk = s[5:].strip()
                if chunk == "[DONE]":
                    break
                try:
                    obj = json.loads(chunk)
                    delta = (obj.get("choices") or [{}])[0].get("delta") or {}
                    if delta.get("content"):
                        yield delta["content"]
                except json.JSONDecodeError:
                    continue

    def _parse_chat_response(self, data: dict[str, Any]) -> ChatCompletionResult:
        choices = data.get("choices") or []
        if not choices:
            return ChatCompletionResult(content="")
        msg = choices[0].get("message") or {}
        raw_content = msg.get("content") or ""
        if isinstance(raw_content, list):
            text = gemma.stringify_content_blocks(raw_content)
        else:
            text = str(raw_content)
        parsed = gemma.parse_assistant_output(text)
        thinking = parsed.thinking
        visible = parsed.visible_user_text
        tool_calls: list[ToolCallSpec] = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            name = fn.get("name") or ""
            tid = str(tc.get("id") or gemma.new_tool_call_id())
            args_raw = fn.get("arguments") or "{}"
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            except json.JSONDecodeError:
                args = {}
            if name:
                tool_calls.append(
                    ToolCallSpec(
                        name=name,
                        arguments=args if isinstance(args, dict) else {},
                        id=tid,
                    )
                )
        if not tool_calls:
            for raw in gemma.tool_calls_from_parsed(parsed):
                tool_calls.append(
                    ToolCallSpec(
                        name=raw["name"],
                        arguments=raw.get("arguments") or {},
                        id=gemma.new_tool_call_id(),
                    )
                )
        usage = data.get("usage") or {}
        u = {}
        for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
            if k in usage:
                u[k] = int(usage[k])
        return ChatCompletionResult(
            content=visible.strip(),
            thinking=thinking,
            tool_calls=tool_calls,
            finish_reason=(choices[0].get("finish_reason")),
            raw_message=msg,
            usage=u,
            raw_assistant_text=text,
        )

    async def embed(self, model: str, text: str) -> list[float]:
        data = await self._post_json(
            "/embeddings",
            {"model": model, "input": text},
        )
        emb = (data.get("data") or [{}])[0].get("embedding")
        if not emb:
            return []
        return [float(x) for x in emb]
