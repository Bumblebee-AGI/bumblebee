"""Backward-compatible imports; prefer ``bumblebee.inference`` for new code."""

from __future__ import annotations

from bumblebee.inference.openai_transport import OpenAICompatibleTransport
from bumblebee.inference.types import ChatCompletionResult, ToolCallSpec

# Historical name — same implementation as the OpenAI-compatible transport.
OllamaClient = OpenAICompatibleTransport

__all__ = ["ChatCompletionResult", "OllamaClient", "ToolCallSpec", "OpenAICompatibleTransport"]
