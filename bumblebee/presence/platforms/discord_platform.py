"""Discord adapter (discord.py)."""

from __future__ import annotations

import asyncio
import os
from typing import Any, Callable

import discord

from bumblebee.models import Input
from bumblebee.presence.platforms.base import Platform


class DiscordPlatform(Platform):
    def __init__(self, token: str, channel_names: list[str]) -> None:
        self.token = token
        self.channel_names = [c.lower() for c in channel_names]
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        self.client = discord.Client(intents=intents)
        self._cb: Callable[[Input], Any] | None = None
        self._ready = asyncio.Event()

        @self.client.event
        async def on_ready() -> None:
            self._ready.set()

        @self.client.event
        async def on_message(message: discord.Message) -> None:
            if message.author.bot:
                return
            if not message.guild or not message.channel:
                return
            ch_name = getattr(message.channel, "name", "") or ""
            if self.channel_names and ch_name.lower() not in self.channel_names:
                return
            if self._cb:
                inp = Input(
                    text=message.content,
                    person_id=str(message.author.id),
                    person_name=str(message.author.display_name or message.author.name),
                    channel=str(message.channel.id),
                    metadata={"channel_name": ch_name},
                    platform="discord",
                )
                await self._cb(inp)

    async def connect(self) -> None:
        asyncio.create_task(self.client.start(self.token))

    async def send_message(self, channel: str, content: str) -> None:
        await self._ready.wait()
        try:
            cid = int(channel)
            ch = self.client.get_channel(cid)
            if ch is not None and isinstance(ch, discord.TextChannel):
                await ch.send(content[:2000])
                return
        except ValueError:
            pass
        for g in self.client.guilds:
            for ch in g.text_channels:
                if ch.name.lower() == channel.lower():
                    await ch.send(content[:2000])
                    return

    async def on_message(self, callback: Callable[..., Any]) -> None:
        self._cb = callback

    async def set_presence(self, status: str) -> None:
        await self._ready.wait()
        await self.client.change_presence(activity=discord.Game(name=status[:128]))

    async def disconnect(self) -> None:
        await self.client.close()

    def get_person_id(self, message: Any) -> str:
        if isinstance(message, discord.Message):
            return str(message.author.id)
        return ""

    def get_person_name(self, message: Any) -> str:
        if isinstance(message, discord.Message):
            return str(message.author.display_name or message.author.name)
        return ""


def token_from_config(token_env: str) -> str:
    return os.environ.get(token_env, "")
