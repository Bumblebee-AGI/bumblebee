"""Telegram — plain text entity replies, photos/voice, relational memory, no command menus."""

from __future__ import annotations

import asyncio
import base64
import io
import os
from typing import TYPE_CHECKING, Any, Callable

import structlog
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, ContextTypes, MessageHandler, filters
from telegram.helpers import escape_markdown

from bumblebee.models import Input
from bumblebee.presence.platforms.base import Platform

if TYPE_CHECKING:
    from bumblebee.entity import Entity

log = structlog.get_logger("bumblebee.presence.telegram")


class TelegramPlatform(Platform):
    def __init__(
        self,
        token: str,
        *,
        entity: Entity,
        app_version: str = "0.1.0",
        allowed_user_ids: set[int] | None = None,
        allowed_chat_ids: set[int] | None = None,
    ) -> None:
        self.token = token
        self._entity = entity
        self._app_version = app_version
        self._allowed_user_ids = allowed_user_ids
        self._allowed_chat_ids = allowed_chat_ids
        self.app = Application.builder().token(token).build()
        self._cb: Callable[[Input], Any] | None = None
        self.last_chat_id: str | None = None

        self.app.add_handler(MessageHandler(filters.PHOTO, self._on_photo))
        self.app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, self._on_voice))
        self.app.add_handler(MessageHandler(filters.Document.ALL, self._on_doc_image))
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_text))

    def _allowed(self, update: Update) -> bool:
        chat = update.effective_chat
        user = update.effective_user
        if chat is None:
            return False
        if self._allowed_chat_ids is not None and chat.id not in self._allowed_chat_ids:
            return False
        if self._allowed_user_ids is not None:
            if user is None or user.id not in self._allowed_user_ids:
                return False
        return True

    async def _on_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.message.text or not self._cb:
            return
        if not self._allowed(update):
            return
        msg = update.message
        self.last_chat_id = str(msg.chat_id)
        log.info("telegram_inbound", platform="telegram", chat_id=msg.chat_id, kind="text")
        u = update.effective_user
        inp = Input(
            text=msg.text,
            person_id=str(u.id) if u else "unknown",
            person_name=(u.first_name or u.username or str(u.id)) if u else "User",
            channel=str(msg.chat_id),
            platform="telegram",
        )
        await self._cb(inp)

    async def _on_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.message.photo or not self._cb:
            return
        if not self._allowed(update):
            return
        msg = update.message
        self.last_chat_id = str(msg.chat_id)
        log.info("telegram_inbound", platform="telegram", chat_id=msg.chat_id, kind="photo")
        try:
            photo = msg.photo[-1]
            tfile = await context.bot.get_file(photo.file_id)
            data = await tfile.download_as_bytearray()
            b64 = base64.standard_b64encode(bytes(data)).decode("ascii")
            caption = (msg.caption or "").strip() or "What do you see?"
            u = update.effective_user
            inp = Input(
                text=caption,
                person_id=str(u.id) if u else "unknown",
                person_name=(u.first_name or u.username or str(u.id)) if u else "User",
                channel=str(msg.chat_id),
                platform="telegram",
                images=[{"base64": b64, "mime": "image/jpeg"}],
            )
            await self._cb(inp)
        except Exception as e:
            log.warning("telegram_photo_failed", error=str(e))

    async def _on_voice(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not self._cb:
            return
        if not self._allowed(update):
            return
        msg = update.message
        self.last_chat_id = str(msg.chat_id)
        log.info("telegram_inbound", platform="telegram", chat_id=msg.chat_id, kind="voice")
        file_id = None
        if msg.voice:
            file_id = msg.voice.file_id
        elif msg.audio:
            file_id = msg.audio.file_id
        if not file_id:
            return
        try:
            tfile = await context.bot.get_file(file_id)
            data = await tfile.download_as_bytearray()
            if len(data) > 8_000_000:
                return
            b64 = base64.standard_b64encode(bytes(data)).decode("ascii")
            cap = (msg.caption or "").strip()
            u = update.effective_user
            inp = Input(
                text=cap,
                person_id=str(u.id) if u else "unknown",
                person_name=(u.first_name or u.username or str(u.id)) if u else "User",
                channel=str(msg.chat_id),
                platform="telegram",
                audio=[{"base64": b64, "format": "ogg"}],
            )
            await self._cb(inp)
        except Exception as e:
            log.warning("telegram_voice_failed", error=str(e))

    async def _on_doc_image(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.message.document or not self._cb:
            return
        if not self._allowed(update):
            return
        doc = update.message.document
        mime = (doc.mime_type or "").lower()
        if not mime.startswith("image/"):
            return
        if not doc.file_id:
            return
        msg = update.message
        self.last_chat_id = str(msg.chat_id)
        log.info("telegram_inbound", platform="telegram", chat_id=msg.chat_id, kind="document_image")
        try:
            tfile = await context.bot.get_file(doc.file_id)
            data = await tfile.download_as_bytearray()
            if len(data) > 8_000_000:
                return
            b64 = base64.standard_b64encode(bytes(data)).decode("ascii")
            cap = (msg.caption or "").strip() or "What's in this image?"
            u = update.effective_user
            inp = Input(
                text=cap,
                person_id=str(u.id) if u else "unknown",
                person_name=(u.first_name or u.username or str(u.id)) if u else "User",
                channel=str(msg.chat_id),
                platform="telegram",
                images=[{"base64": b64, "mime": mime or "image/jpeg"}],
            )
            await self._cb(inp)
        except Exception as e:
            log.warning("telegram_document_failed", error=str(e))

    async def connect(self) -> None:
        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling(allowed_updates=Update.ALL_TYPES)

    async def send_typing(self, chat_id: int) -> None:
        try:
            await self.app.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        except Exception as e:
            log.debug("send_typing_failed", error=str(e))

    async def send_attachment_bytes(
        self,
        channel: str,
        data: bytes,
        *,
        content_type: str | None = None,
        filename: str = "attachment.bin",
    ) -> None:
        """Deliver bytes from blob storage as photo (image/*) or document."""
        if not data:
            return
        chat_id = int(channel)
        ct = (content_type or "").lower()
        try:
            if ct.startswith("image/"):
                await self.app.bot.send_photo(chat_id=chat_id, photo=io.BytesIO(data))
            else:
                await self.app.bot.send_document(
                    chat_id=chat_id,
                    document=io.BytesIO(data),
                    filename=filename[:255],
                )
        except Exception as e:
            log.warning("telegram_send_attachment_failed", error=str(e))

    async def send_message(self, channel: str, content: str) -> None:
        chat_id = int(channel)
        text = content[:4096]
        await self.app.bot.send_message(chat_id=chat_id, text=text)

    async def send_tool_activity(self, description: str) -> None:
        if not description.strip() or not self.last_chat_id:
            return
        esc = escape_markdown(description.strip(), version=2)
        text = f"_{esc}_"
        try:
            await self.app.bot.send_message(
                chat_id=int(self.last_chat_id),
                text=text[:4096],
                parse_mode="MarkdownV2",
            )
        except Exception as e:
            log.debug("send_tool_activity_failed", error=str(e))

    async def send_plain_chunks(self, channel: str, text: str, pause: float = 1.0) -> None:
        t = text.strip()
        if not t:
            return
        n = 4000
        for i in range(0, len(t), n):
            await self.send_message(channel, t[i : i + n])
            if i + n < len(t):
                await asyncio.sleep(pause)

    async def send_proactive_default(self, message: str) -> None:
        if self.last_chat_id:
            await self.send_plain_chunks(self.last_chat_id, message)

    async def on_message(self, callback: Callable[..., Any]) -> None:
        self._cb = callback

    async def set_presence(self, status: str) -> None:
        pass

    async def disconnect(self) -> None:
        await self.app.updater.stop()
        await self.app.stop()
        await self.app.shutdown()

    def get_person_id(self, message: Any) -> str:
        if isinstance(message, Update):
            u = message.effective_user
            return str(u.id) if u else ""
        return ""

    def get_person_name(self, message: Any) -> str:
        if isinstance(message, Update):
            u = message.effective_user
            if not u:
                return ""
            if u.first_name:
                return f"{u.first_name} {u.last_name or ''}".strip()
            return u.username or str(u.id)
        return ""


def token_from_config(token_env: str) -> str:
    return os.environ.get(token_env, "")
