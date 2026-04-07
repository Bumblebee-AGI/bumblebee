"""Telegram (chat_id, message_id) dedup + per-chat serialization for entity callback."""

from __future__ import annotations

import pytest

from bumblebee.models import Input
from bumblebee.presence.platforms.telegram_platform import TelegramPlatform


class _StubEntity:
    """TelegramPlatform only needs an object for typing; callback tests skip connect()."""


@pytest.mark.asyncio
async def test_telegram_skips_second_callback_same_message_id() -> None:
    calls: list[int] = []

    async def cb(_inp: Input) -> None:
        calls.append(1)

    p = TelegramPlatform("000000:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", entity=_StubEntity())
    p._cb = cb
    inp = Input(
        text="hi",
        person_id="1",
        person_name="u",
        channel="99",
        platform="telegram",
        metadata={"telegram_message_id": 42, "chat_type": "private"},
    )
    await p._run_callback(inp)
    await p._run_callback(inp)
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_telegram_reruns_after_failed_callback_releases_claim() -> None:
    calls: list[int] = []
    fail_once = {"v": True}

    async def cb(_inp: Input) -> None:
        calls.append(1)
        if fail_once["v"]:
            fail_once["v"] = False
            raise RuntimeError("boom")

    p = TelegramPlatform("000000:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", entity=_StubEntity())
    p._cb = cb
    inp = Input(
        text="hi",
        person_id="1",
        person_name="u",
        channel="100",
        platform="telegram",
        metadata={"telegram_message_id": 7, "chat_type": "private"},
    )
    await p._run_callback(inp)
    await p._run_callback(inp)
    assert len(calls) == 2
