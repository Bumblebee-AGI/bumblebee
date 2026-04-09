"""Terminal UI: prompt_toolkit input, rich output, entitative session chrome."""

from __future__ import annotations

import asyncio
import shutil
import sqlite3
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style as PTStyle
from rich.console import Console
from rich.text import Text

from bumblebee.models import Input
from bumblebee.entity import Entity
from bumblebee.presence.platforms.base import Platform
from bumblebee.presence.platforms.cli_render import (
    CLIHeaderSnapshot,
    SessionShutdownSummary,
    compute_awake_summary,
    dominant_drive_line,
    dominant_drive_toolbar,
    render_error,
    render_feelings_introspection,
    render_memories_introspection,
    render_shutdown,
    render_side_panel,
    render_startup,
    STYLE_ENTITY,
    STYLE_VALUE,
    STYLE_WHISPER,
)


def _sync_episode_count(db_path: str) -> int:
    """Avoid aiosqlite here — daemon + REPL can race on Windows/asyncio."""
    p = Path(db_path).expanduser()
    if not p.is_file():
        return 0
    try:
        with sqlite3.connect(str(p)) as conn:
            row = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()
            return int(row[0]) if row else 0
    except sqlite3.Error:
        return 0


def _sync_relationship_count(db_path: str) -> int:
    p = Path(db_path).expanduser()
    if not p.is_file():
        return 0
    try:
        with sqlite3.connect(str(p)) as conn:
            row = conn.execute("SELECT COUNT(*) FROM relationships").fetchone()
            return int(row[0]) if row else 0
    except sqlite3.Error:
        return 0


def _sync_min_episode_timestamp(db_path: str) -> float | None:
    p = Path(db_path).expanduser()
    if not p.is_file():
        return None
    try:
        with sqlite3.connect(str(p)) as conn:
            row = conn.execute("SELECT MIN(timestamp) FROM episodes").fetchone()
            if row and row[0] is not None:
                return float(row[0])
    except sqlite3.Error:
        pass
    return None


def _sync_recent_summaries(db_path: str, limit: int = 5) -> list[str]:
    p = Path(db_path).expanduser()
    if not p.is_file():
        return []
    try:
        with sqlite3.connect(str(p)) as conn:
            cur = conn.execute(
                "SELECT summary FROM episodes ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            )
            return [str(r[0]) for r in cur.fetchall() if r and r[0]]
    except sqlite3.Error:
        return []


class CLIPlatform(Platform):
    def __init__(
        self,
        entity: Entity,
        app_version: str = "0.1.0",
        *,
        immersive: bool = True,
    ) -> None:
        self.entity = entity
        self.entity_name = entity.config.name
        self.app_version = app_version
        self.immersive = immersive
        self._cb: Callable[[Input], Awaitable[Any]] | None = None
        self._connected = False
        cols, _rows = shutil.get_terminal_size(fallback=(88, 24))
        # Cap width so Rich rules/panels don't stretch across huge IDE terminals.
        tw = max(52, min(int(cols), 90))
        self.console = Console(highlight=False, soft_wrap=True, width=tw)

        self._session_t0: float = 0.0
        self._exchange_count = 0
        self._episodes_start: int | None = None
        self._mood_start_label = ""
        self._session_begun = False

        self._thinking_task: asyncio.Task[None] | None = None
        self._emitted_this_turn = False
        self._chunk_acc = ""
        self._turn_visible = ""
        self._expression_meta = entity.voice_ctl.meta_for_response(
            entity.emotions.get_state(), 80, "cli"
        )

    async def connect(self) -> None:
        self._connected = True

    def _invalidate_app(self) -> None:
        try:
            from prompt_toolkit.application import get_app

            get_app().invalidate()
        except Exception:
            pass

    def _sync_toolbar_state(self) -> None:
        self._tb_mood = self.entity.emotions.get_state().primary.value
        self._tb_drive = dominant_drive_toolbar(self.entity.drives.all_drives())

    async def _build_header_snapshot(self) -> CLIHeaderSnapshot:
        cfg = self.entity.config
        st = self.entity.emotions.get_state()
        if getattr(self.entity.store, "dialect", "sqlite") == "sqlite":
            path = self.entity.store.db_path
            n_ep = _sync_episode_count(path)
            n_people = _sync_relationship_count(path)
            first_ts = _sync_min_episode_timestamp(path)
        else:
            n_ep, n_people, first_ts = await self.entity.fetch_cli_header_counts()
        awake = compute_awake_summary(
            created_raw=cfg.raw.get("created"),
            first_episode_ts=first_ts,
        )
        return CLIHeaderSnapshot(
            app_version=self.app_version,
            entity_name=cfg.name,
            mood_label=st.primary.value,
            drive_line=dominant_drive_line(self.entity.drives.all_drives()),
            episode_count=n_ep,
            reflex_model=cfg.cognition.reflex_model,
            max_context_tokens=cfg.cognition.max_context_tokens,
            tool_count=len(self.entity.tools.openai_tools()),
            awake_summary=awake,
            people_count=n_people,
        )

    async def begin_talk_session(self) -> None:
        if self._session_begun or not self.immersive:
            return
        self._session_begun = True
        self._session_t0 = time.time()
        self._mood_start_label = self.entity.emotions.get_state().primary.value
        if getattr(self.entity.store, "dialect", "sqlite") == "sqlite":
            self._episodes_start = _sync_episode_count(self.entity.store.db_path)
        else:
            self._episodes_start = (await self.entity.fetch_cli_header_counts())[0]

        self._sync_toolbar_state()
        snap = await self._build_header_snapshot()
        render_startup(self.console, snap)
        await asyncio.sleep(0.5)

        await self._before_generation(estimated_reply_len=48)
        reply = ""
        try:
            reply = await self.entity.cli_opening(
                stream=self.stream_delta,
                reply_platform=self,
            )
        except Exception as e:
            await self._cancel_thinking()
            render_error(self.console, str(e))
        finally:
            await self._after_generation(reply)

    async def send_message(self, channel: str, content: str) -> None:
        prefix = Text(self.entity_name + " ", style=STYLE_ENTITY)
        body = Text(content, style=STYLE_VALUE)
        self.console.print(prefix, body, sep="", end="")
        self.console.print()

    async def send_tool_activity(self, description: str) -> None:
        t = (description or "").strip()
        if not t:
            return
        await self._cancel_thinking()
        self.console.print(Text(f"   {t}", style=STYLE_WHISPER), highlight=False)

    async def on_message(self, callback: Callable[..., Any]) -> None:
        self._cb = callback

    def _print_entity_prefix(self) -> None:
        self.console.print(Text(f"{self.entity_name} ", style=STYLE_ENTITY), end="")

    async def _thinking_loop(self) -> None:
        # Avoid \r spinners: they smear horizontally under prompt_toolkit's patched stdout.
        try:
            while True:
                await asyncio.sleep(3600.0)
        except asyncio.CancelledError:
            raise

    async def _cancel_thinking(self) -> None:
        if self._thinking_task and not self._thinking_task.done():
            self._thinking_task.cancel()
            try:
                await self._thinking_task
            except asyncio.CancelledError:
                pass
        self._thinking_task = None

    async def _before_generation(self, estimated_reply_len: int = 120) -> None:
        self._emitted_this_turn = False
        self._chunk_acc = ""
        self._turn_visible = ""
        st = self.entity.emotions.get_state()
        self._expression_meta = self.entity.voice_ctl.meta_for_response(
            st,
            estimated_reply_len,
            "cli",
        )
        await asyncio.sleep(min(3.0, self._expression_meta.typing_delay_seconds))
        await self._cancel_thinking()
        self._thinking_task = asyncio.create_task(self._thinking_loop())

    async def stream_delta(self, s: str) -> None:
        await self._cancel_thinking()
        if not self._emitted_this_turn:
            self._print_entity_prefix()
            self._emitted_this_turn = True
        self._chunk_acc += s
        self._turn_visible += s
        self.console.print(s, end="", highlight=False, markup=False)

        m = self.entity.config.harness.presence.message_chunk_max
        if len(self._chunk_acc) >= m and self._sentence_boundary(self._turn_visible):
            self.console.print()
            await asyncio.sleep(self._expression_meta.chunk_pause)
            self._print_entity_prefix()
            self._chunk_acc = ""

    @staticmethod
    def _sentence_boundary(text: str) -> bool:
        t = text.rstrip()
        if not t:
            return False
        return t[-1] in ".!?…"

    async def _after_generation(self, reply: str) -> None:
        await self._cancel_thinking()
        if not self._emitted_this_turn and reply.strip():
            self._print_entity_prefix()
            self.console.print(reply, style=STYLE_VALUE, highlight=False, markup=False)
        self.console.print()
        self._emitted_this_turn = False
        self._chunk_acc = ""
        self._turn_visible = ""
        self._sync_toolbar_state()
        self._invalidate_app()

    def _slash_self(self) -> None:
        self.console.print()
        render_side_panel(self.console, self.entity)

    async def _slash_status(self) -> None:
        snap = await self._build_header_snapshot()
        self.console.print()
        render_startup(self.console, snap)

    async def _slash_memories(self) -> None:
        summaries = await self.entity.fetch_cli_recent_summaries(5)
        self.console.print()
        render_memories_introspection(self.console, self.entity_name, summaries)

    def _slash_feelings(self) -> None:
        self.console.print()
        render_feelings_introspection(self.console, self.entity_name, self.entity.emotions.get_state())

    def _slash_tools(self) -> None:
        list_fn = getattr(self.entity.tools, "list_tools", None)
        if callable(list_fn):
            rows = list_fn()
        else:
            rows = []
            for t in self.entity.tools.openai_tools():
                fn = t.get("function", {}) if isinstance(t, dict) else {}
                name = str(fn.get("name") or "").strip()
                desc = str(fn.get("description") or "").strip()
                if name:
                    rows.append((name, desc))
            rows = sorted(rows, key=lambda r: r[0])
        self.console.print()
        self.console.print(Text(f"Active tools ({len(rows)})", style=STYLE_LABEL))
        if not rows:
            self.console.print(Text("none", style=STYLE_WHISPER))
            return
        for name, desc in rows:
            self.console.print(Text(f"  - {name}", style=STYLE_VALUE))
            if desc:
                self.console.print(Text(f"      {desc}", style=STYLE_WHISPER))

    async def _graceful_bye(self) -> None:
        if getattr(self.entity.store, "dialect", "sqlite") == "sqlite":
            episodes_end = _sync_episode_count(self.entity.store.db_path)
        else:
            episodes_end = (await self.entity.fetch_cli_header_counts())[0]
        mood_end = self.entity.emotions.get_state().primary.value
        start_ep = (
            self._episodes_start if self._episodes_start is not None else episodes_end
        )
        summary = SessionShutdownSummary(
            entity_name=self.entity_name,
            duration_seconds=max(0.0, time.time() - self._session_t0),
            exchange_count=self._exchange_count,
            mood_start=self._mood_start_label or mood_end,
            mood_end=mood_end,
            episodes_saved=max(0, episodes_end - start_ep),
        )
        try:
            render_shutdown(self.console, summary)
        except Exception:
            self.console.print(f"\n[dim]{self.entity_name} — session ended.[/dim]\n")
        self._connected = False

    async def run_repl(self) -> None:
        self._sync_toolbar_state()
        if self.immersive:
            await self.begin_talk_session()
        else:
            self._session_t0 = time.time()
            self._mood_start_label = self.entity.emotions.get_state().primary.value
            if getattr(self.entity.store, "dialect", "sqlite") == "sqlite":
                self._episodes_start = _sync_episode_count(self.entity.store.db_path)
            else:
                self._episodes_start = (await self.entity.fetch_cli_header_counts())[0]

        # noreverse: default toolbar is reverse-video and clashes with Rich / many themes.
        _repl_style = PTStyle.from_dict(
            {
                "bottom-toolbar": "noreverse fg:#8a8a8a",
                "prompt": "#707070",
            }
        )

        def bottom_toolbar() -> str:
            return f"{self._tb_mood} · {self._tb_drive}     /self   /bye"

        session = PromptSession(
            message=[("class:prompt", "› ")],
            style=_repl_style,
            bottom_toolbar=bottom_toolbar,
        )

        async def _prompt_line() -> str:
            if hasattr(session, "prompt_async"):
                return await session.prompt_async()
            return await asyncio.to_thread(session.prompt, "")

        async def _bye_safe() -> None:
            try:
                await self._graceful_bye()
            except Exception as e:
                self._connected = False
                self.console.print(f"\n[dim]bye ({e})[/dim]\n")

        # raw=True: keep Rich's ANSI intact; default False strips VT sequences → garbled `?[2m` output.
        with patch_stdout(raw=True):
            while self._connected:
                try:
                    line = await _prompt_line()
                except (EOFError, KeyboardInterrupt):
                    await _bye_safe()
                    break
                line = (line or "").strip()
                if not line:
                    continue
                if line == "/bye":
                    await _bye_safe()
                    break
                if line == "/status":
                    await self._slash_status()
                    continue
                if line == "/self":
                    self._slash_self()
                    continue
                if line == "/memories":
                    await self._slash_memories()
                    continue
                if line == "/feelings":
                    self._slash_feelings()
                    continue
                if line == "/tools":
                    self._slash_tools()
                    continue

                self._exchange_count += 1
                if self._cb:
                    inp = Input(
                        text=line,
                        person_id="cli_user",
                        person_name="You",
                        channel="cli",
                        platform="cli",
                    )
                    try:
                        await self._before_generation(estimated_reply_len=len(line))
                        reply = await self._cb(inp)
                        reply = reply or ""
                        await self._after_generation(str(reply))
                    except Exception as e:
                        await self._cancel_thinking()
                        render_error(self.console, str(e))
                        self.console.print()
                        self._sync_toolbar_state()
                        self._invalidate_app()

    async def set_presence(self, status: str) -> None:
        pass

    async def disconnect(self) -> None:
        self._connected = False

    async def send_audio(self, channel: str, path: str) -> bool:
        # CLI has no media transport; intentionally no-op.
        return False

    async def send_image(self, channel: str, path: str) -> None:
        # CLI has no media transport; intentionally no-op.
        return None

    def get_person_id(self, message: Any) -> str:
        return "cli_user"

    def get_person_name(self, message: Any) -> str:
        return "You"
