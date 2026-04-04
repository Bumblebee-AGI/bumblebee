"""Plain-text / HTML copy for Telegram (onboarding, slash commands, introspection)."""

from __future__ import annotations

import html
import time
from bumblebee.presence.platforms.cli_render import (
    CLIHeaderSnapshot,
    _fmt_duration,
    compute_awake_summary,
    dominant_drive_line,
)


def split_telegram_chunks(text: str, limit: int = 4096) -> list[str]:
    """Split for Telegram message size; prefer paragraph then word boundaries."""
    text = text.strip() if text else ""
    if not text:
        return [""]
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    rest = text
    while rest:
        if len(rest) <= limit:
            chunks.append(rest.rstrip())
            break
        cut = rest.rfind("\n\n", 0, limit)
        if cut < limit // 3:
            cut = rest.rfind("\n", 0, limit)
        if cut < limit // 3:
            cut = rest.rfind(" ", 0, limit)
        if cut < limit // 3:
            cut = limit
        piece = rest[:cut].rstrip()
        if piece:
            chunks.append(piece)
        rest = rest[cut:].lstrip()
    return chunks


def format_start_html(entity_name: str, app_version: str) -> str:
    en = html.escape(entity_name)
    return (
        f"◈ <b>{en}</b> · bumblebee v{html.escape(app_version)}\n\n"
        "I'm not a task bot — I'm a persistent presence. "
        "Talk like you would to someone who remembers.\n\n"
        "<i>Tip:</i> tap the <b>/</b> menu for commands, or try "
        "<code>/commands</code> to browse everything."
    )


def format_help_html(entity_name: str) -> str:
    en = html.escape(entity_name)
    return (
        f"<b>{en}</b> understands natural messages, photos (vision), and slash commands.\n\n"
        "<b>Core</b>\n"
        "/start — this welcome\n"
        "/help — short guide\n"
        "/commands [page] — full command list (paginated)\n"
        "/status — mood, stack, memory snapshot\n"
        "/memories — recent episode summaries\n"
        "/feelings — how I'm doing inside\n"
        "/reset — clear <i>conversation</i> history (not your memories DB)\n\n"
        "Send a photo with an optional caption; I'll look with the model's eyes."
    )


# (command, one-line description) for paginated /commands
COMMAND_REGISTRY: list[tuple[str, str]] = [
    ("start", "Welcome, identity, and how to use the menu"),
    ("help", "What I accept: text, photos, commands"),
    ("commands", "This list — add a page number to flip"),
    ("status", "Mood, drives, models, memory counts"),
    ("memories", "Last few things I've written into episodic memory"),
    ("feelings", "Introspective read on emotional state"),
    ("reset", "Clear rolling chat context (keeps SQLite memory)"),
]


def format_commands_page(page: int, *, per_page: int = 5) -> tuple[str, int, int]:
    """
    page: 0-based index.
    Returns (html_body, page_index, total_pages).
    """
    total = max(1, (len(COMMAND_REGISTRY) + per_page - 1) // per_page)
    page = max(0, min(page, total - 1))
    start = page * per_page
    rows = COMMAND_REGISTRY[start : start + per_page]
    lines = ["<b>Commands</b> — page {}/{}".format(page + 1, total), ""]
    for cmd, desc in rows:
        lines.append(f"/{html.escape(cmd)} — {html.escape(desc)}")
    lines.append("")
    lines.append("<i>Send</i> <code>/commands {}</code> <i>for next page.</i>".format(
        page + 2 if page + 1 < total else 1
    ))
    return "\n".join(lines), page, total


async def build_status_html(entity: "Entity", app_version: str) -> str:
    db = await entity.store.connect()
    async with db:
        n_ep = await entity.store.count_episodes(db)
        n_people = await entity.store.count_relationships(db)
        first_ts = await entity.store.min_episode_timestamp(db)
    cfg = entity.config
    st = entity.emotions.get_state()
    awake = compute_awake_summary(
        created_raw=cfg.raw.get("created"),
        first_episode_ts=first_ts,
    )
    snap = CLIHeaderSnapshot(
        app_version=app_version,
        entity_name=cfg.name,
        mood_label=st.primary.value,
        drive_line=dominant_drive_line(entity.drives.all_drives()),
        episode_count=n_ep,
        reflex_model=cfg.cognition.reflex_model,
        max_context_tokens=cfg.cognition.max_context_tokens,
        tool_count=len(entity.tools.openai_tools()),
        awake_summary=awake,
        people_count=n_people,
    )
    en = html.escape(snap.entity_name)
    return (
        f"◈ <b>{en}</b>\n"
        f"mood · {html.escape(snap.mood_label)} · {html.escape(snap.drive_line)}\n"
        f"memory · {snap.episode_count} episodes · knows {snap.people_count} people\n"
        f"model · {html.escape(snap.reflex_model)} · context · {snap.max_context_tokens // 1000}k\n"
        f"tools · {snap.tool_count} active · awake · {html.escape(snap.awake_summary)}"
    )


def format_memories_html(entity_name: str, summaries: list[str]) -> str:
    en = html.escape(entity_name)
    if not summaries:
        return f"<i>{en}</i> reaches for recent memories… the shelf is still sparse."
    lines = [f"<i>{en}</i> leafs through recent memories…", ""]
    for i, s in enumerate(summaries, 1):
        lines.append(f"{i}. {html.escape(s[:500])}")
    return "\n".join(lines)


def format_feelings_html(entity_name: str, state) -> str:
    en = html.escape(entity_name)
    primary = html.escape(state.primary.value)
    pi = f"{state.intensity:.1f}"
    parts = [f"<i>{en}</i> is feeling <b>{primary}</b> ({pi})"]
    if state.secondary is not None:
        parts.append(
            f"with an undercurrent of {html.escape(state.secondary.value)} "
            f"({state.secondary_intensity:.1f})"
        )
    ago = _fmt_duration(max(0.0, time.time() - state.last_transition))
    parts.append(f"— in this shade for about {html.escape(ago)}.")
    return " ".join(parts)


def format_media_unavailable(kind: str) -> str:
    k = html.escape(kind)
    return (
        f"I can’t take in <b>{k}</b> yet — send text or a <b>photo</b> "
        "(caption optional). Voice and video notes are on the roadmap."
    )


def format_access_denied() -> str:
    return (
        "◈ This instance is locked to approved people. "
        "If you should have access, ask whoever runs the harness to add your Telegram user id."
    )
