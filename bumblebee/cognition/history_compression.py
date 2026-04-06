"""Merge dropped chat turns into a rolling summary so context trims do not erase the thread."""

from __future__ import annotations

import structlog
from typing import Any

from bumblebee.cognition import gemma
from bumblebee.inference import InferenceProvider

log = structlog.get_logger("bumblebee.cognition.history_compression")

_SYS = (
    "You maintain a compact rolling memory of a conversation. "
    "You receive (1) a prior summary, possibly empty, and (2) older transcript lines that no longer "
    "fit in the live chat window. Merge them: keep concrete facts, names, decisions, open questions, "
    "emotional beats, and what the user and assistant were working on. Drop filler and repetition. "
    "Write in first person as the assistant (I / we / they / the user). "
    "Output ONLY the updated summary — no title, no preamble, no quotes around it."
)


def format_history_messages_for_summary(messages: list[dict[str, Any]], per_msg_cap: int = 2200) -> str:
    """Flatten history dicts into lines for the summarizer."""
    lines: list[str] = []
    for m in messages:
        role = str(m.get("role") or "")
        content = m.get("content", "")
        if isinstance(content, list):
            content = gemma.stringify_content_blocks(content)
        text = str(content).strip()
        if len(text) > per_msg_cap:
            text = text[: per_msg_cap - 1] + "…"
        name = str(m.get("name") or "").strip()
        tool_calls = m.get("tool_calls")
        if role == "tool" and name:
            lines.append(f"tool_result({name}): {text}")
        elif tool_calls:
            lines.append(f"{role} [tool_calls]: {text[:800]}")
        else:
            lines.append(f"{role}: {text}")
    return "\n".join(lines).strip()


def _clip_block(s: str, max_chars: int) -> str:
    s = (s or "").strip()
    if len(s) <= max_chars:
        return s
    head = max_chars // 2
    tail = max_chars - head - 40
    return s[:head] + "\n… [middle omitted] …\n" + s[-tail:]


async def merge_rolling_summary(
    client: InferenceProvider,
    model: str,
    *,
    entity_name: str,
    prior_summary: str,
    dropped_messages: list[dict[str, Any]],
    per_msg_cap: int,
    max_merge_input_chars: int,
    merge_max_tokens: int,
    summary_max_chars: int,
    num_ctx: int | None = None,
) -> str:
    """Call the model once to fold ``dropped_messages`` into ``prior_summary``."""
    dropped_fmt = format_history_messages_for_summary(dropped_messages, per_msg_cap=per_msg_cap)
    dropped_fmt = _clip_block(dropped_fmt, max_merge_input_chars)
    prior = (prior_summary or "").strip()
    prior_clipped = prior[:8000] if len(prior) > 8000 else prior
    user_block = (
        f"Entity name (for tone only): {entity_name}\n\n"
        f"Prior summary:\n{(prior_clipped or '[none]')}\n\n"
        f"Older transcript to fold in:\n{dropped_fmt or '[empty]'}"
    )
    try:
        res = await client.chat_completion(
            model,
            [
                {"role": "system", "content": _SYS},
                {"role": "user", "content": user_block},
            ],
            temperature=0.25,
            max_tokens=max(200, min(int(merge_max_tokens), 2048)),
            think=False,
            num_ctx=num_ctx,
        )
        out = (getattr(res, "content", None) or "").strip()
    except Exception as e:
        log.warning("history_merge_failed", error=str(e))
        return (prior + "\n\n[Earlier thread dropped; summary merge failed.]").strip()[:summary_max_chars]

    if len(out) < 20:
        log.info("history_merge_too_short", out_len=len(out))
        return (prior + "\n\n[Earlier thread dropped; summary merge returned empty.]").strip()[
            :summary_max_chars
        ]

    if len(out) > summary_max_chars:
        out = out[: summary_max_chars - 20].rstrip() + "\n… [trimmed]"
    log.info(
        "history_merged",
        dropped_messages=len(dropped_messages),
        summary_chars=len(out),
    )
    return out
