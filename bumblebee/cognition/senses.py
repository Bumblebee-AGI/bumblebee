"""Multimodal input normalization for Gemma / Ollama."""

from __future__ import annotations

import base64
from typing import Any

from bumblebee.config import EntityConfig
from bumblebee.models import Input


def input_to_message_content(inp: Input, _image_budget: int) -> str | list[dict[str, Any]]:
    """Build OpenAI-style content; text + optional image_url parts."""
    parts: list[dict[str, Any]] = [{"type": "text", "text": inp.text}]
    for img in inp.images[:3]:
        b64 = img.get("base64")
        if not b64 and img.get("path"):
            try:
                with open(img["path"], "rb") as f:
                    b64 = base64.standard_b64encode(f.read()).decode("ascii")
            except OSError:
                continue
        if b64:
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                }
            )
    if len(parts) == 1:
        return inp.text
    return parts


def clip_text_for_budget(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"
