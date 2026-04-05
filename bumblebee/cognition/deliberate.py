"""26B deliberate path with thinking + multi-step tools."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Any, AsyncIterator, Awaitable, Callable, Literal

from bumblebee.config import EntityConfig
from bumblebee.cognition import gemma
from bumblebee.models import Input
from bumblebee.inference.protocol import InferenceProvider
from bumblebee.inference.types import ChatCompletionResult, ToolCallSpec


@dataclass
class DeliberateStreamEvent:
    """One slice of a multi-round deliberate run (tool loop)."""

    kind: Literal["intermediate", "final"]
    display_text: str = ""
    """Visible user-directed text from the model for this slice (intermediate or final)."""
    history_entries: list[dict[str, Any]] = field(default_factory=list)
    """Assistant + tool + follow-up messages to merge into rolling history (intermediate only)."""
    merged_thinking: str | None = None
    """CoT / reasoning accumulated across all model calls in this perceive cycle (final only)."""
    last_result: ChatCompletionResult | None = None
    """Raw completion for the final model call (truncation extension, metadata)."""


class DeliberateCognition:
    def __init__(self, entity: EntityConfig, client: InferenceProvider) -> None:
        self.entity = entity
        self.client = client

    def _build_messages(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        cap = max(2000, int(self.entity.cognition.system_prompt_char_limit or 12000))
        msgs: list[dict[str, Any]] = [{"role": "system", "content": system_prompt[:cap]}]
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
        return msgs

    @staticmethod
    def _assistant_content_for_message(res: ChatCompletionResult) -> str:
        return (
            res.raw_assistant_text
            if (res.raw_assistant_text or "").strip()
            else (res.content or "")
        )

    async def iter_responses(
        self,
        _inp: Input,
        system_prompt: str,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_executor: Callable[[ToolCallSpec], Awaitable[str]] | None = None,
    ) -> AsyncIterator[DeliberateStreamEvent]:
        """Multi-round tool loop; yields intermediate user-visible text before tools run."""
        h = self.entity.harness.cognition
        max_toks = int(self.entity.cognition.deliberate_max_tokens or h.deliberate_max_tokens)
        max_toks = max(128, max_toks)
        msgs = self._build_messages(system_prompt, messages)
        thinking_acc: list[str] = []
        last: ChatCompletionResult | None = None

        for _ in range(8):
            res = await self.client.chat_completion(
                self.entity.cognition.deliberate_model,
                msgs,
                tools=tools,
                temperature=h.temperature,
                max_tokens=max_toks,
                think=h.thinking_mode,
            )
            last = res
            if res.thinking:
                thinking_acc.append(res.thinking)

            if not tool_executor or not res.tool_calls:
                merged = "\n\n".join(thinking_acc) if thinking_acc else None
                yield DeliberateStreamEvent(
                    kind="final",
                    display_text=(res.content or "").strip(),
                    history_entries=[],
                    merged_thinking=merged,
                    last_result=res,
                )
                return

            for tc in res.tool_calls:
                if not tc.id:
                    tc.id = gemma.new_tool_call_id()

            assistant_content = self._assistant_content_for_message(res)
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
                "content": (
                    "Continue: integrate tool results into your reply for the person you're talking to."
                ),
            }
            msgs = msgs + [assistant_msg] + tool_msgs + [follow_user]

            yield DeliberateStreamEvent(
                kind="intermediate",
                display_text=(res.content or "").strip(),
                history_entries=[assistant_msg] + tool_msgs + [follow_user],
            )

        merged = "\n\n".join(thinking_acc) if thinking_acc else None
        yield DeliberateStreamEvent(
            kind="final",
            display_text=(last.content or "").strip() if last else "",
            history_entries=[],
            merged_thinking=merged,
            last_result=last or ChatCompletionResult(content=""),
        )

    async def respond(
        self,
        _inp: Input,
        system_prompt: str,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_executor: Callable[[ToolCallSpec], Awaitable[str]] | None = None,
    ) -> ChatCompletionResult:
        """Aggregate a full deliberate run (single return value)."""
        final_ev: DeliberateStreamEvent | None = None
        async for ev in self.iter_responses(
            _inp,
            system_prompt,
            messages,
            tools=tools,
            tool_executor=tool_executor,
        ):
            if ev.kind == "final":
                final_ev = ev
        if not final_ev or not final_ev.last_result:
            return ChatCompletionResult(content="")
        r = final_ev.last_result
        return replace(r, thinking=final_ev.merged_thinking)
