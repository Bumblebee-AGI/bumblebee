"""Embedding helper using Ollama."""

from __future__ import annotations

from bumblebee.utils.ollama_client import OllamaClient


async def embed_text(client: OllamaClient, model: str, text: str) -> list[float]:
    return await client.embed(model, text)
