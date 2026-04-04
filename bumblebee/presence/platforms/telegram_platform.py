"""Telegram adapter — product-grade UX: commands menu, /commands pages, photos, inline hints."""

from __future__ import annotations

import asyncio
import base64
import logging
import os
from typing import TYPE_CHECKING, Any, Callable

from telegram import BotCommand, InlineQueryResultArticle, InputTextMessageContent, Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    InlineQueryHandler,
    MessageHandler,
    filters,
)

from bumblebee.models import Input
from bumblebee.presence.platforms.base import Platform
from bumblebee.presence.platforms.telegram_format import (
    build_status_html,
    format_access_denied,
    format_commands_page,
    format_feelings_html,
    format_help_html,
    format_media_unavailable,
    format_memories_html,
    format_start_html,
    split_telegram_chunks,
)

if TYPE_CHECKING:
    from bumblebee.entity import Entity

log = logging.getLogger("bumblebee.telegram")


class TelegramPlatform(Platform):
    def __init__(
        self,
        token: str,
        *,
        entity: Entity,
        app_version: str = "0.1.0",
        allowed_user_ids: set[int] | None = None,
    ) -> None:
        self.token = token
        self._entity = entity
        self._app_version = app_version
        self._allowed_ids = allowed_user_ids
        self.app = Application.builder().token(token).build()
        self._cb: Callable[[Input], Any] | None = None

        self.app.add_handler(CommandHandler("start", self._cmd_start))
        self.app.add_handler(CommandHandler("help", self._cmd_help))
        self.app.add_handler(CommandHandler("commands", self._cmd_commands))
        self.app.add_handler(CommandHandler("status", self._cmd_status))
        self.app.add_handler(CommandHandler("memories", self._cmd_memories))
        self.app.add_handler(CommandHandler("feelings", self._cmd_feelings))
        self.app.add_handler(CommandHandler("reset", self._cmd_reset))
        self.app.add_handler(InlineQueryHandler(self._inline_query))
        self.app.add_handler(MessageHandler(filters.PHOTO, self._handle_photo))
        self.app.add_handler(
            MessageHandler(filters.VOICE | filters.AUDIO | filters.VIDEO_NOTE, self._handle_av_media)
        )
        self.app.add_handler(MessageHandler(filters.Sticker.ALL, self._handle_sticker))
        self.app.add_handler(MessageHandler(filters.Document.ALL, self._handle_document))
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_text))

    def _allowed_uid(self, uid: int | None) -> bool:
        if self._allowed_ids is None:
            return True
        if uid is None:
            return False
        return uid in self._allowed_ids

    def _allowed(self, update: Update) -> bool:
        u = update.effective_user
        return self._allowed_uid(u.id if u else None)

    async def _reply_html(self, update: Update, html: str) -> None:
        if not update.message:
            return
        chat_id = update.message.chat_id
        for part in split_telegram_chunks(html, 3900):
            await self.app.bot.send_message(
                chat_id=chat_id,
                text=part,
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True,
            )
            await asyncio.sleep(0.2)

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message:
            return
        if not self._allowed(update):
            await self._reply_html(update, format_access_denied())
            return
        await self._reply_html(
            update,
            format_start_html(self._entity.config.name, self._app_version),
        )

    async def _cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message:
            return
        if not self._allowed(update):
            await self._reply_html(update, format_access_denied())
            return
        await self._reply_html(update, format_help_html(self._entity.config.name))

    async def _cmd_commands(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message:
            return
        if not self._allowed(update):
            await self._reply_html(update, format_access_denied())
            return
        page = 0
        if context.args:
            try:
                page = max(0, int(context.args[0]) - 1)
            except ValueError:
                page = 0
        body, _, _ = format_commands_page(page)
        await self._reply_html(update, body)

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message:
            return
        if not self._allowed(update):
            await self._reply_html(update, format_access_denied())
            return
        try:
            body = await build_status_html(self._entity, self._app_version)
            await self._reply_html(update, body)
        except Exception as e:
            log.warning("telegram_status_failed: %s", e)
            await self._reply_html(update, f"Couldn’t read status right now: {e!s}")

    async def _cmd_memories(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message:
            return
        if not self._allowed(update):
            await self._reply_html(update, format_access_denied())
            return
        try:
            db = await self._entity.store.connect()
            try:
                summaries = await self._entity.episodic.recent_summaries(db, 5)
            finally:
                await db.close()
            await self._reply_html(
                update,
                format_memories_html(self._entity.config.name, summaries),
            )
        except Exception as e:
            log.warning("telegram_memories_failed: %s", e)
            await self._reply_html(update, f"Memories didn’t load: {e!s}")

    async def _cmd_feelings(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message:
            return
        if not self._allowed(update):
            await self._reply_html(update, format_access_denied())
            return
        st = self._entity.emotions.get_state()
        await self._reply_html(update, format_feelings_html(self._entity.config.name, st))

    async def _cmd_reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message:
            return
        if not self._allowed(update):
            await self._reply_html(update, format_access_denied())
            return
        self._entity.clear_conversation_history()
        await self._reply_html(
            update,
            "<i>Conversation context cleared.</i> Episodic memory in SQLite is untouched.",
        )

    async def _inline_query(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.inline_query:
            return
        iq_user = update.inline_query.from_user.id if update.inline_query.from_user else None
        if not self._allowed_uid(iq_user):
            await update.inline_query.answer([], cache_time=0)
            return
        name = self._entity.config.name
        un = context.bot.username or "bot"
        about = InlineQueryResultArticle(
            id="about",
            title=f"{name} · bumblebee",
            description="Persistent entity — not a task bot",
            input_message_content=InputTextMessageContent(
                message_text=(
                    f"◈ <b>{name}</b>\n"
                    f"Open @{un} to talk. Built with bumblebee (entitative harness).\n"
                    "Natural chat, photos, /help for commands."
                ),
                parse_mode=ParseMode.HTML,
            ),
        )
        tips = InlineQueryResultArticle(
            id="tips",
            title="Commands",
            description="/status · /memories · /feelings · /commands",
            input_message_content=InputTextMessageContent(
                message_text=(
                    f"Try <code>/help</code> or <code>/commands</code> in @{un} "
                    f"with {name}."
                ),
                parse_mode=ParseMode.HTML,
            ),
        )
        await update.inline_query.answer([about, tips], cache_time=300)

    async def _handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.message.photo or not self._cb:
            return
        if not self._allowed(update):
            await self._reply_html(update, format_access_denied())
            return
        try:
            photo = update.message.photo[-1]
            tfile = await context.bot.get_file(photo.file_id)
            data = await tfile.download_as_bytearray()
            b64 = base64.standard_b64encode(bytes(data)).decode("ascii")
            caption = (update.message.caption or "").strip() or "What do you see?"
            u = update.effective_user
            inp = Input(
                text=caption,
                person_id=str(u.id) if u else "unknown",
                person_name=u.first_name if u else "User",
                channel=str(update.message.chat_id),
                platform="telegram",
                images=[{"base64": b64}],
            )
            await self._cb(inp)
        except Exception as e:
            log.warning("telegram_photo_failed: %s", e)
            await self._reply_html(update, f"Couldn’t read that image: {e!s}")

    async def _handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.message.document:
            return
        if not self._allowed(update):
            await self._reply_html(update, format_access_denied())
            return
        doc = update.message.document
        mime = (doc.mime_type or "").lower()
        if not mime.startswith("image/"):
            await self._reply_html(
                update,
                "I only take <b>image</b> files as uploads right now — try a photo or screenshot.",
            )
            return
        if not self._cb or not doc.file_id:
            return
        try:
            tfile = await context.bot.get_file(doc.file_id)
            data = await tfile.download_as_bytearray()
            if len(data) > 8_000_000:
                await self._reply_html(update, "That file is too large for me to pull in.")
                return
            b64 = base64.standard_b64encode(bytes(data)).decode("ascii")
            cap = (update.message.caption or "").strip() or "What's in this image?"
            u = update.effective_user
            inp = Input(
                text=cap,
                person_id=str(u.id) if u else "unknown",
                person_name=u.first_name if u else "User",
                channel=str(update.message.chat_id),
                platform="telegram",
                images=[{"base64": b64}],
            )
            await self._cb(inp)
        except Exception as e:
            log.warning("telegram_document_failed: %s", e)
            await self._reply_html(update, f"Couldn't open that file: {e!s}")

    async def _handle_av_media(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message:
            return
        if not self._allowed(update):
            await self._reply_html(update, format_access_denied())
            return
        label = "voice or audio"
        if update.message.video_note:
            label = "video note"
        await self._reply_html(update, format_media_unavailable(label))

    async def _handle_sticker(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message:
            return
        if not self._allowed(update):
            await self._reply_html(update, format_access_denied())
            return
        await self._reply_html(
            update,
            "Cute sticker — I don’t unpack those yet. Send words or a photo.",
        )

    async def _handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.message.text or not self._cb:
            return
        if not self._allowed(update):
            await self._reply_html(update, format_access_denied())
            return
        u = update.effective_user
        inp = Input(
            text=update.message.text,
            person_id=str(u.id) if u else "unknown",
            person_name=u.first_name if u else "User",
            channel=str(update.message.chat_id),
            platform="telegram",
        )
        await self._cb(inp)

    async def connect(self) -> None:
        await self.app.initialize()
        await self.app.start()
        await self._install_bot_commands()
        await self.app.updater.start_polling(allowed_updates=Update.ALL_TYPES)

    async def _install_bot_commands(self) -> None:
        cmds = [
            BotCommand("start", "Welcome & who I am"),
            BotCommand("help", "Text, photos, commands"),
            BotCommand("commands", "Full list (paginated)"),
            BotCommand("status", "Mood, memory, stack"),
            BotCommand("memories", "Recent episodes"),
            BotCommand("feelings", "Inner state"),
            BotCommand("reset", "Clear chat context only"),
        ]
        try:
            await self.app.bot.set_my_commands(cmds)
        except Exception as e:
            log.warning("set_my_commands_failed: %s", e)

    async def send_typing(self, chat_id: int) -> None:
        """Telegram client 'typing…' indicator (refreshed every few seconds while model runs)."""
        try:
            await self.app.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        except Exception as e:
            log.debug("send_typing_failed: %s", e)

    async def send_message(self, channel: str, content: str) -> None:
        chat_id = int(channel)
        for part in split_telegram_chunks(content, 4096):
            await self.app.bot.send_message(chat_id=chat_id, text=part)
            await asyncio.sleep(0.2)

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
                return f"{u.first_name} {u.last_name}".strip() if u.last_name else u.first_name
            return u.username or str(u.id)
        return ""


def token_from_config(token_env: str) -> str:
    return os.environ.get(token_env, "")
