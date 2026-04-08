"""Primary agent loop with tool use + bounded continuation."""

from __future__ import annotations

import asyncio
import json
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

InferenceProfile = Literal["deliberate", "reflex"]


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


_TOOL_CONTINUATION_USER = (
    "Same turn. Keep going until you can either answer the person clearly from the current tool "
    "results, call another tool, or state the exact failure. Do not stop at progress chatter."
)

def _build_post_tool_nudge(
    tool_calls: list[ToolCallSpec],
    tool_msgs: list[dict[str, Any]],
    user_text: str,
) -> str:
    """Context-aware nudge referencing what tools ran and what the user asked."""
    if not tool_calls:
        return (
            "Same turn. You have tool results now. Either answer the user from those results, "
            "or if a tool failed and another tool can fix it, call that tool now. "
            "Do not only say you're checking."
        )
    summaries: list[str] = []
    for tc_spec, tm in zip(tool_calls, tool_msgs):
        failed = _tool_content_failed(tm.get("content") or "")
        status = "failed" if failed else "returned results"
        summaries.append(f"  - {tc_spec.name}: {status}")
    tool_block = "\n".join(summaries)
    user_snippet = (user_text or "").strip()[:300]
    return (
        f"Same turn. Tool results are ready:\n{tool_block}\n"
        f"The user asked: {user_snippet}\n"
        "Answer from these results, call another tool if needed, "
        "or state exactly what failed. Do not just acknowledge."
    )

_FINAL_RECOVERY_USER = (
    "Same turn. Your last message did not finish the task. Continue until you either answer the user "
    "clearly from tool results or state the exact failure."
)


def _finish_reason_hits_limit(finish_reason: str | None) -> bool:
    if not finish_reason:
        return False
    fr = finish_reason.lower().replace("_", "").replace("-", "")
    return fr in ("length", "maxtokens", "maxlength")


def _tool_content_failed(content: str) -> bool:
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


def _visible_assistant_text(res: ChatCompletionResult) -> str:
    return (res.content or "").strip()


@dataclass
class AgentLoopState:
    step_index: int = 0
    step_budget: int = 0
    tool_rounds: int = 0
    tool_calls_seen: int = 0
    last_step_had_tools: bool = False
    last_tool_failed: bool = False
    completion_failures: int = 0
    user_requested_tools: bool = False


FinalCheck = Callable[[str, ChatCompletionResult, AgentLoopState], Awaitable[tuple[bool, str | None]]]


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

    def _inference_params(self, profile: InferenceProfile) -> tuple[str, int, bool, float]:
        """Model id, max_tokens, think flag, temperature."""
        h = self.entity.harness.cognition
        if profile == "reflex":
            model = (self.entity.cognition.reflex_model or "").strip()
            if not model:
                model = (self.entity.cognition.deliberate_model or "").strip()
            mt = max(32, int(h.reflex_max_tokens))
            return model, mt, False, float(min(0.9, h.temperature))
        model = (self.entity.cognition.deliberate_model or "").strip()
        mt = int(self.entity.cognition.deliberate_max_tokens or h.deliberate_max_tokens)
        mt = max(128, mt)
        return model, mt, bool(h.thinking_mode), float(h.temperature)

    def _tool_continuation_rounds_cap(self) -> int:
        """Legacy continuation knob; still shapes the bounded agent loop."""
        h = int(self.entity.harness.cognition.tool_continuation_rounds)
        ec = int(self.entity.cognition.tool_continuation_rounds)
        n = ec if ec > 0 else h
        return max(0, min(16, n))

    def _agent_step_cap(self) -> int:
        """Primary bounded loop budget for a single user turn."""
        return max(3, min(16, 4 + self._tool_continuation_rounds_cap()))

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
        inference_profile: InferenceProfile = "deliberate",
        final_checker: FinalCheck | None = None,
    ) -> AsyncIterator[DeliberateStreamEvent]:
        """Primary bounded agent loop for one user turn."""
        model, max_toks, think_flag, temperature = self._inference_params(inference_profile)
        msgs = self._build_messages(system_prompt, messages)
        thinking_acc: list[str] = []
        last: ChatCompletionResult | None = None
        num_ctx = self.entity.effective_ollama_num_ctx()
        loop_state = AgentLoopState(
            step_budget=self._agent_step_cap(),
            user_requested_tools="use tools" in (_inp.text or "").lower(),
        )
        for step_idx in range(loop_state.step_budget):
            loop_state.step_index = step_idx + 1
            res = await self.client.chat_completion(
                model,
                msgs,
                tools=tools,
                temperature=temperature,
                max_tokens=max_toks,
                think=think_flag,
                num_ctx=num_ctx,
            )
            last = res
            if res.thinking:
                thinking_acc.append(res.thinking)

            if tool_executor and res.tool_calls:
                for tc in res.tool_calls:
                    if not tc.id:
                        tc.id = gemma.new_tool_call_id()

                loop_state.last_step_had_tools = True
                loop_state.tool_rounds += 1
                loop_state.tool_calls_seen += len(res.tool_calls)
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
                tool_failed = False

                async def _safe_exec(spec: ToolCallSpec) -> str:
                    try:
                        return await tool_executor(spec)
                    except Exception as e:
                        return json.dumps({"error": str(e)})

                results = await asyncio.gather(
                    *[_safe_exec(tc) for tc in res.tool_calls]
                )
                for tc, out in zip(res.tool_calls, results):
                    tool_failed = tool_failed or _tool_content_failed(out)
                    tool_msgs.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": tc.name,
                            "content": out,
                        }
                    )
                loop_state.last_tool_failed = tool_failed
                follow_user = {
                    "role": "user",
                    "content": _build_post_tool_nudge(
                        res.tool_calls, tool_msgs, _inp.text,
                    ),
                }
                msgs = msgs + [assistant_msg] + tool_msgs + [follow_user]

                yield DeliberateStreamEvent(
                    kind="intermediate",
                    display_text="",
                    history_entries=[assistant_msg] + tool_msgs + [follow_user],
                )
                continue

            loop_state.last_step_had_tools = False
            visible = _visible_assistant_text(res)
            done = True
            recovery = None
            if final_checker is not None:
                done, recovery = await final_checker(visible, res, loop_state)
            elif _finish_reason_hits_limit(res.finish_reason):
                done = False
                recovery = _TOOL_CONTINUATION_USER
            if not done and step_idx + 1 < loop_state.step_budget:
                loop_state.completion_failures += 1
                asst = self._assistant_content_for_message(res)
                asst_msg = {"role": "assistant", "content": asst}
                cont_user = {"role": "user", "content": recovery or _FINAL_RECOVERY_USER}
                msgs = msgs + [asst_msg, cont_user]
                log.info(
                    "deliberate_loop_continue",
                    step=loop_state.step_index,
                    step_budget=loop_state.step_budget,
                    completion_failures=loop_state.completion_failures,
                    finish_reason=res.finish_reason,
                )
                yield DeliberateStreamEvent(
                    kind="intermediate",
                    display_text="",
                    history_entries=[asst_msg, cont_user],
                )
                continue

            merged = "\n\n".join(thinking_acc) if thinking_acc else None
            yield DeliberateStreamEvent(
                kind="final",
                display_text=visible,
                history_entries=[],
                merged_thinking=merged,
                last_result=res,
            )
            return

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
        inference_profile: InferenceProfile = "deliberate",
        final_checker: FinalCheck | None = None,
    ) -> ChatCompletionResult:
        """Aggregate a full deliberate run (single return value)."""
        final_ev: DeliberateStreamEvent | None = None
        async for ev in self.iter_responses(
            _inp,
            system_prompt,
            messages,
            tools=tools,
            tool_executor=tool_executor,
            inference_profile=inference_profile,
            final_checker=final_checker,
        ):
            if ev.kind == "final":
                final_ev = ev
        if not final_ev or not final_ev.last_result:
            return ChatCompletionResult(content="")
        r = final_ev.last_result
        return replace(r, thinking=final_ev.merged_thinking)
