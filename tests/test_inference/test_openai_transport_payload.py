"""OpenAI-compatible transport payload shaping (Ollama extensions)."""

from __future__ import annotations

import pytest

from bumblebee.inference.openai_transport import OpenAICompatibleTransport


@pytest.mark.asyncio
async def test_chat_completion_adds_options_num_ctx() -> None:
    t = OpenAICompatibleTransport(base_url="http://127.0.0.1:9")
    captured: dict = {}

    async def fake_post(_path: str, payload: dict) -> dict:
        captured.clear()
        captured.update(payload)
        return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}

    t._post_json = fake_post  # type: ignore[method-assign]

    await t.chat_completion(
        "m",
        [{"role": "user", "content": "hi"}],
        num_ctx=16384,
    )
    assert captured.get("options") == {"num_ctx": 16384}


def test_build_payload_omits_options_when_num_ctx_none() -> None:
    t = OpenAICompatibleTransport(base_url="http://127.0.0.1:9")
    p = t._build_chat_completions_payload(
        "m",
        [{"role": "user", "content": "x"}],
        temperature=0.5,
        max_tokens=10,
        think=False,
        stream=False,
        num_ctx=None,
    )
    assert "options" not in p


def test_build_payload_omits_options_when_num_ctx_zero() -> None:
    t = OpenAICompatibleTransport(base_url="http://127.0.0.1:9")
    p = t._build_chat_completions_payload(
        "m",
        [{"role": "user", "content": "x"}],
        temperature=0.5,
        max_tokens=10,
        think=False,
        stream=False,
        num_ctx=0,
    )
    assert "options" not in p
