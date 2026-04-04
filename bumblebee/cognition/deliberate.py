"""26B deliberate path with thinking + optional tools."""

from __future__ import annotations

import json
from typing import Any, Awaitable, Callable

from bumblebee.config import EntityConfig
from bumblebee.cognition import gemma
from bumblebee.models import Input
from bumblebee.utils.ollama_client import ChatCompletionResult, OllamaClient, ToolCallSpec


class DeliberateCognition:
    def __init__(self, entity: EntityConfig, client: OllamaClient) -> None:
        self.entity = entity
        self.client = client

    def _build_base_messages(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        user_payload: dict[str, Any],
    ) -> list[dict[str, Any]]:
        h = self.entity.harness.cognition
        msgs: list[dict[str, Any]] = [{"role": "system", "content": system_prompt[:12000]}]
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if isinstance(content, list):
                content = gemma.stringify_content_blocks(content)
            elif role == "assistant" and isinstance(content, str):
                content = gemma.strip_thinking_for_history(content)
            entry: dict[str, Any] = {"role": role, "content": content}
            if m.get("tool_calls"):
                entry["tool_calls"] = m["tool_calls"]
            if role == "tool":
                entry["tool_call_id"] = m.get("tool_call_id", "")
                entry["name"] = m.get("name", "")
            msgs.append(entry)
        msgs.append(user_payload)
        return msgs

    async def respond(
        self,
        inp: Input,
        system_prompt: str,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_executor: Callable[[ToolCallSpec], Awaitable[str]] | None = None,
    ) -> ChatCompletionResult:
        h = self.entity.harness.cognition
        user_payload: dict[str, Any] = {"role": "user", "content": inp.text[:8000]}
        msgs = self._build_base_messages(system_prompt, messages, user_payload)

        res = await self.client.chat_completion(
            self.entity.cognition.deliberate_model,
            msgs,
            tools=tools,
            temperature=h.temperature,
            max_tokens=h.deliberate_max_tokens,
            think=h.thinking_mode,
        )

        if tool_executor and res.tool_calls:
            for tc in res.tool_calls:
                if not tc.id:
                    tc.id = gemma.new_tool_call_id()
            assistant_content = (
                res.raw_assistant_text
                if res.raw_assistant_text.strip()
                else (res.content or "")
            )
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in res.tool_calls
                ],
            }
            tool_msgs: list[dict[str, Any]] = []
            for tc in res.tool_calls:
                out = await tool_executor(tc)
                tool_msgs.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tc.name,
                        "content": out,
                    }
                )
            follow_user = {
                "role": "user",
                "content": "Continue: integrate tool results into your reply for the user.",
            }
            msgs2 = msgs + [assistant_msg] + tool_msgs + [follow_user]
            res2 = await self.client.chat_completion(
                self.entity.cognition.deliberate_model,
                msgs2,
                tools=tools,
                temperature=h.temperature,
                max_tokens=min(h.deliberate_max_tokens, 2048),
                think=h.thinking_mode,
            )
            if isinstance(res2, ChatCompletionResult):
                return res2
        return res
