"""26B deliberate path with thinking + multi-step tools."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from typing import Any, AsyncIterator, Awaitable, Callable, Literal

import structlog

from bumblebee.config import EntityConfig
from bumblebee.cognition import gemma
from bumblebee.models import Input
from bumblebee.inference.protocol import InferenceProvider
from bumblebee.inference.types import ChatCompletionResult, ToolCallSpec
from bumblebee.presence.tools.registry import (
    TOOL_SYSTEM_PROMPT_PREFIX,
    TOOL_SYSTEM_PROMPT_PREFIX_COMPACT,
)

log = structlog.get_logger("bumblebee.cognition.deliberate")


def fit_system_prompt_to_budget(system_prompt: str, cap: int) -> str:
    """
    Apply ``cap`` without dropping the trailing tools appendix: the naive ``[:cap]`` cut
    removes everything after the identity/knowledge prefix, so the model never sees tool cues.
    """
    s = system_prompt or ""
    if len(s) <= cap:
        return s
    suffix_start = -1
    for marker in (TOOL_SYSTEM_PROMPT_PREFIX, TOOL_SYSTEM_PROMPT_PREFIX_COMPACT):
        idx = s.find(marker)
        if idx != -1 and (suffix_start == -1 or idx < suffix_start):
            suffix_start = idx
    if suffix_start == -1:
        return s[:cap]
    suffix = s[suffix_start:]
    if len(suffix) >= cap:
        return s[:cap]
    room = cap - len(suffix)
    return s[:suffix_start][:room] + suffix


# After tool results, if the model answers with text only but clearly still intends another tool
# (or hit max_tokens before emitting tool calls), inject a user nudge and continue the loop.
MAX_TOOL_CONTINUATION_ROUNDS = 2

_TOOL_CONTINUATION_USER = (
    "Continue in this same turn: if you still need to call a tool to finish what you started "
    "(retry after an error, install a missing dependency, run a shell command, etc.), "
    "invoke the tool now. Do not only say you will do it later. "
    "Otherwise answer in plain text for the user—stay in character. "
    "Do not add meta lines about being finished or the task completing."
)


def _finish_reason_hits_limit(finish_reason: str | None) -> bool:
    if not finish_reason:
        return False
    fr = finish_reason.lower().replace("_", "").replace("-", "")
    return fr in ("length", "maxtokens", "maxlength")


# Tight phrases only: bare "try to" / "going to" / "i'll" match normal prose (e.g. "try to land")
# and spuriously trigger continuation → models echo the nudge as "I'm done."
_INTENTS_FURTHER_TOOL = re.compile(
    r"\b("
    r"lemme\b|hang on|hold on|one sec|give me a sec|real quick|"
    r"(i'?ll|i will)\s+(install|run|check|try|exec|search|look\s+up|pull\s+up|fetch)\b|"
    r"(gonna|going\s+to)\s+(install|run|check|try|exec|search|look\s+up|pull\s+up|fetch)\b|"
    r"need\s+to\s+(install|run|check|try|exec|search|look\s+up|pull\s+up|fetch)\b|"
    r"pip3?\s+install|run_command|run command|"
    r"fix that|sort that|sort this|be right back|brb|"
    r"let me (try|install|run|check|see|grab|get|do)\b"
    r")",
    re.IGNORECASE,
)


def _tool_content_suggests_retry(content: str) -> bool:
    raw = (content or "").strip()
    if not raw:
        return False
    lt = raw.lower()
    if "not installed" in lt or ("missing" in lt and "package" in lt):
        return True
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return '"error"' in raw
    if isinstance(obj, dict):
        if obj.get("error"):
            return True
        if obj.get("ok") is False:
            return True
    return False


def should_inject_tool_continuation(msgs: list[dict[str, Any]], res: ChatCompletionResult) -> bool:
    """
    True when the model returned no tool calls but context suggests another tool round is needed
    (failed tool, truncation, or explicit intent to run something).
    """
    if res.tool_calls:
        return False
    tool_msgs = [m for m in msgs if m.get("role") == "tool"]
    if not tool_msgs:
        return False
    last_tool = str(tool_msgs[-1].get("content") or "")
    tool_suggests_retry = _tool_content_suggests_retry(last_tool)
    visible = (res.content or "").strip()
    if not visible and _finish_reason_hits_limit(res.finish_reason):
        return True
    if not visible:
        return False
    assistant_intends = bool(_INTENTS_FURTHER_TOOL.search(visible))
    if _finish_reason_hits_limit(res.finish_reason):
        return True
    return tool_suggests_retry or assistant_intends


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
        msgs: list[dict[str, Any]] = [
            {"role": "system", "content": fit_system_prompt_to_budget(system_prompt, cap)},
        ]
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if isinstance(content, list):
                # Preserve multimodal user/tool payloads (e.g. image_url) for OpenAI-compatible backends.
                # Only assistant content should be flattened to text for history carryover.
                if role == "assistant":
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
        """Multi-round tool loop; appends tool rounds to history; user-visible text only on final (or continuation nudges)."""
        h = self.entity.harness.cognition
        max_toks = int(self.entity.cognition.deliberate_max_tokens or h.deliberate_max_tokens)
        max_toks = max(128, max_toks)
        msgs = self._build_messages(system_prompt, messages)
        thinking_acc: list[str] = []
        last: ChatCompletionResult | None = None
        continuation_remaining = MAX_TOOL_CONTINUATION_ROUNDS

        num_ctx = self.entity.effective_ollama_num_ctx()
        for _ in range(8):
            res = await self.client.chat_completion(
                self.entity.cognition.deliberate_model,
                msgs,
                tools=tools,
                temperature=h.temperature,
                max_tokens=max_toks,
                think=h.thinking_mode,
                num_ctx=num_ctx,
            )
            last = res
            if res.thinking:
                thinking_acc.append(res.thinking)

            if not tool_executor or not res.tool_calls:
                if (
                    tool_executor
                    and tools
                    and continuation_remaining > 0
                    and should_inject_tool_continuation(msgs, res)
                ):
                    continuation_remaining -= 1
                    asst = self._assistant_content_for_message(res)
                    asst_msg: dict[str, Any] = {"role": "assistant", "content": asst}
                    cont_user = {"role": "user", "content": _TOOL_CONTINUATION_USER}
                    msgs = msgs + [asst_msg, cont_user]
                    log.info(
                        "deliberate_tool_continuation",
                        remaining_after=continuation_remaining,
                        finish_reason=res.finish_reason,
                    )
                    yield DeliberateStreamEvent(
                        kind="intermediate",
                        display_text=(res.content or "").strip(),
                        history_entries=[asst_msg, cont_user],
                    )
                    continue

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
                    "Continue: integrate tool results for the person you're talking to. "
                    "If a tool failed or returned an error you can fix with another tool "
                    "(install a dependency, retry, etc.), call that tool now — do not only promise to do it later."
                ),
            }
            msgs = msgs + [assistant_msg] + tool_msgs + [follow_user]

            yield DeliberateStreamEvent(
                kind="intermediate",
                display_text="",
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
