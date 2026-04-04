"""Discord adapter — plain text, relational memory ids, multimodal, presence, reconnect."""

from __future__ import annotations

import asyncio
import base64
import os
from typing import TYPE_CHECKING, Any, Callable

import discord
import structlog
from discord.utils import escape_markdown

from bumblebee.models import EmotionCategory, Input

_discord_log = structlog.get_logger("bumblebee.presence.discord")
from bumblebee.presence.platforms.base import Platform

if TYPE_CHECKING:
    from bumblebee.entity import Entity

def _emotion_discord_presence(primary: EmotionCategory) -> discord.Status:
    if primary in (
        EmotionCategory.CONTENT,
        EmotionCategory.CURIOUS,
        EmotionCategory.EXCITED,
        EmotionCategory.AMUSED,
        EmotionCategory.AFFECTIONATE,
    ):
        return discord.Status.online
    if primary in (EmotionCategory.NEUTRAL, EmotionCategory.RESTLESS):
        return discord.Status.idle
    if primary in (
        EmotionCategory.WITHDRAWN,
        EmotionCategory.FRUSTRATED,
        EmotionCategory.ANXIOUS,
        EmotionCategory.MELANCHOLY,
    ):
        return discord.Status.dnd
    return discord.Status.online


class DiscordPlatform(Platform):
    def __init__(
        self,
        token: str,
        channel_names: list[str],
        *,
        entity: Entity | None = None,
        proactive_channel_id: int | None = None,
    ) -> None:
        self.token = token
        self.channel_names = [c.lower() for c in channel_names] if channel_names else []
        self._entity = entity
        self._proactive_channel_id = proactive_channel_id
        self.last_channel_id: str | None = None
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        intents.voice_states = False
        self.client = discord.Client(intents=intents)
        self._cb: Callable[[Input], Any] | None = None
        self._ready = asyncio.Event()
        self._run = True
        self._runner_task: asyncio.Task[None] | None = None
        self._bot_id: int | None = None

        @self.client.event
        async def on_ready() -> None:
            self._bot_id = self.client.user.id if self.client.user else None
            self._ready.set()
            _discord_log.info("discord_ready", platform="discord", user=str(self.client.user))

        @self.client.event
        async def on_message(message: discord.Message) -> None:
            await self._dispatch_message(message)

    def _should_respond(self, message: discord.Message) -> bool:
        if message.author.bot:
            return False
        if self._bot_id and message.author.id == self._bot_id:
            return False
        if not message.channel:
            return False
        if isinstance(message.channel, discord.DMChannel):
            return True
        if not message.guild:
            return False
        ch_name = getattr(message.channel, "name", "") or ""
        mentioned = self._bot_id and self.client.user in message.mentions
        in_list = not self.channel_names or ch_name.lower() in self.channel_names
        return bool(mentioned or in_list)

    async def _dispatch_message(self, message: discord.Message) -> None:
        if not self._should_respond(message):
            return
        if not self._cb:
            return
        self.last_channel_id = str(message.channel.id)
        text = (message.content or "").strip()
        if self._bot_id and self.client.user:
            text = text.replace(f"<@{self._bot_id}>", "").replace(f"<@!{self._bot_id}>", "").strip()
        images: list[dict[str, Any]] = []
        audio_parts: list[dict[str, Any]] = []
        for att in message.attachments:
            ct = (att.content_type or "").lower()
            fn = (att.filename or "").lower()
            try:
                if ct.startswith("image/") or fn.endswith((".png", ".jpg", ".jpeg", ".gif", ".webp")):
                    raw = await att.read()
                    if len(raw) > 12_000_000:
                        continue
                    b64 = base64.standard_b64encode(raw).decode("ascii")
                    images.append({"base64": b64, "mime": ct or "image/jpeg"})
                elif "audio" in ct or fn.endswith((".ogg", ".oga", ".mp3", ".wav", ".m4a")):
                    raw = await att.read()
                    if len(raw) > 8_000_000:
                        continue
                    b64 = base64.standard_b64encode(raw).decode("ascii")
                    fmt = "ogg"
                    if "mpeg" in ct or fn.endswith(".mp3"):
                        fmt = "mp3"
                    elif "wav" in ct or fn.endswith(".wav"):
                        fmt = "wav"
                    audio_parts.append({"base64": b64, "format": fmt})
            except Exception as e:
                _discord_log.warning("discord_attachment_failed", error=str(e))
        if not text and images:
            text = "What do you see?"
        if not text and audio_parts:
            text = ""
        inp = Input(
            text=text,
            person_id=str(message.author.id),
            person_name=str(message.author.display_name or message.author.name),
            channel=str(message.channel.id),
            metadata={"channel_name": getattr(message.channel, "name", "") or ""},
            platform="discord",
            images=images,
            audio=audio_parts,
        )
        _discord_log.info("discord_inbound", platform="discord", channel_id=inp.channel)
        await self._cb(inp)

    async def _run_client(self) -> None:
        while self._run:
            try:
                await self.client.start(self.token)
            except asyncio.CancelledError:
                break
            except Exception as e:
                _discord_log.warning("discord_disconnect", error=str(e))
                await asyncio.sleep(8)
            finally:
                try:
                    await self.client.close()
                except Exception:
                    pass
                self._ready.clear()

    async def connect(self) -> None:
        if self._runner_task and not self._runner_task.done():
            return
        self._runner_task = asyncio.create_task(self._run_client(), name="discord-runner")

    async def send_message(self, channel: str, content: str) -> None:
        await self._ready.wait()
        plain = content[:2000]
        try:
            cid = int(channel)
            ch = self.client.get_channel(cid)
            if ch is not None and isinstance(ch, discord.abc.Messageable):
                await ch.send(plain)
                return
        except ValueError:
            pass
        for g in self.client.guilds:
            for c in g.text_channels:
                if c.name.lower() == channel.lower():
                    await c.send(plain)
                    return

    async def send_tool_activity(self, description: str) -> None:
        if not description.strip() or not self.last_channel_id:
            return
        await self._ready.wait()
        esc = escape_markdown(description.strip())
        text = f"_{esc}_"[:2000]
        try:
            cid = int(self.last_channel_id)
            ch = self.client.get_channel(cid)
            if ch is not None and isinstance(ch, discord.abc.Messageable):
                await ch.send(text)
        except Exception as e:
            _discord_log.debug("send_tool_activity_failed", error=str(e))

    async def send_plain_chunks(self, channel_id: str, text: str, chunk_pause: float = 1.2) -> None:
        """Human-paced plain text (no embeds)."""
        t = text.strip()
        if not t:
            return
        max_len = 1900
        parts: list[str] = []
        buf = ""
        for sentence in t.replace("\n\n", "\n").split("\n"):
            line = sentence.strip()
            if not line:
                continue
            if len(buf) + len(line) + 1 > max_len:
                if buf:
                    parts.append(buf)
                buf = line
            else:
                buf = (buf + " " + line).strip() if buf else line
        if buf:
            parts.append(buf)
        if len(parts) == 1 and len(parts[0]) > max_len:
            parts = [parts[0][i : i + max_len] for i in range(0, len(parts[0]), max_len)]
        for i, p in enumerate(parts):
            await self.send_message(channel_id, p)
            if i < len(parts) - 1:
                await asyncio.sleep(chunk_pause)

    async def send_proactive_default(self, message: str) -> None:
        cid = self._proactive_channel_id
        if cid is not None:
            await self.send_plain_chunks(str(cid), message)
            return
        if self.last_channel_id:
            await self.send_plain_chunks(self.last_channel_id, message)

    async def on_message(self, callback: Callable[..., Any]) -> None:
        self._cb = callback

    async def set_presence(self, status: str) -> None:
        await self._ready.wait()
        await self.client.change_presence(activity=discord.Game(name=status[:128]))

    async def sync_emotion_presence(self, primary: EmotionCategory) -> None:
        try:
            await self._ready.wait()
            st = _emotion_discord_presence(primary)
            await self.client.change_presence(status=st)
        except Exception as e:
            _discord_log.debug("discord_presence_failed", error=str(e))

    async def disconnect(self) -> None:
        self._run = False
        if self._runner_task:
            self._runner_task.cancel()
            try:
                await self._runner_task
            except asyncio.CancelledError:
                pass
            self._runner_task = None
        try:
            await self.client.close()
        except Exception:
            pass

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
