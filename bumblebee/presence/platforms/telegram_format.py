"""HTML copy helpers for Telegram onboarding, commands, and introspection."""

from __future__ import annotations

import html
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from bumblebee.presence.platforms.cli_render import (
    CLIHeaderSnapshot,
    _fmt_duration,
    compute_awake_summary,
    dominant_drive_line,
)

if TYPE_CHECKING:
    from bumblebee.entity import Entity


@dataclass(frozen=True)
class TelegramCommandSpec:
    name: str
    summary: str
    usage: str
    category: str


def split_telegram_chunks(text: str, limit: int = 4096) -> list[str]:
    """Split under Telegram's message limit; prefer paragraph then word boundaries."""
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


COMMAND_REGISTRY: list[TelegramCommandSpec] = [
    TelegramCommandSpec(
        name="start",
        summary="Onboarding, quick wins, and next steps",
        usage="/start",
        category="Getting started",
    ),
    TelegramCommandSpec(
        name="help",
        summary="Short guide with practical examples",
        usage="/help",
        category="Getting started",
    ),
    TelegramCommandSpec(
        name="commands",
        summary="Browse all slash commands (paged or filtered)",
        usage="/commands [page] [filter]",
        category="Getting started",
    ),
    TelegramCommandSpec(
        name="status",
        summary="Current mood, memory, model, and tool snapshot",
        usage="/status",
        category="Introspection",
    ),
    TelegramCommandSpec(
        name="feelings",
        summary="Detailed emotional read",
        usage="/feelings",
        category="Introspection",
    ),
    TelegramCommandSpec(
        name="memories",
        summary="Recent episodic memory summaries",
        usage="/memories [count]",
        category="Introspection",
    ),
    TelegramCommandSpec(
        name="me",
        summary="How I currently model our relationship",
        usage="/me",
        category="Introspection",
    ),
    TelegramCommandSpec(
        name="models",
        summary="Inference and runtime model configuration",
        usage="/models",
        category="Runtime",
    ),
    TelegramCommandSpec(
        name="ping",
        summary="Quick liveness check",
        usage="/ping",
        category="Runtime",
    ),
    TelegramCommandSpec(
        name="reset",
        summary="Clear rolling chat turns (keeps SQLite memory)",
        usage="/reset",
        category="Session",
    ),
]


def command_menu_items() -> list[tuple[str, str]]:
    """(name, description) list suitable for Telegram setMyCommands."""
    return [(c.name, c.summary[:256]) for c in COMMAND_REGISTRY]


def format_start_html(entity_name: str, app_version: str, *, first_name: str | None = None) -> str:
    en = html.escape(entity_name)
    who = html.escape((first_name or "there").strip() or "there")
    return (
        f"◈ <b>{en}</b> · bumblebee v{html.escape(app_version)}\n\n"
        f"Hey, {who}. I'm not a task bot; I'm a persistent presence that keeps context and memory over time.\n\n"
        "<b>Best ways to use me</b>\n"
        "• Talk naturally, like a real ongoing conversation.\n"
        "• Send photos (with an optional caption) and I can inspect them.\n"
        "• Use slash commands for introspection and controls.\n\n"
        "<b>Try these now</b>\n"
        "• <code>/status</code> for a live internal snapshot\n"
        "• <code>/memories 3</code> for recent memory traces\n"
        "• <code>/me</code> for how I currently know you\n\n"
        "Tap the <b>/</b> menu anytime, or use <code>/commands</code> to browse everything."
    )


def format_help_html(entity_name: str) -> str:
    en = html.escape(entity_name)
    return (
        f"<b>{en}</b> works best when messages have context and intent.\n\n"
        "<b>Natural chat examples</b>\n"
        "• \"Give me a 15-minute plan to learn X\"\n"
        "• \"I sent a screenshot — what stands out?\"\n"
        "• \"Summarize what we decided earlier today\"\n\n"
        "<b>Useful commands</b>\n"
        "• <code>/status</code>, <code>/feelings</code>, <code>/memories [count]</code>\n"
        "• <code>/me</code> to inspect relationship state\n"
        "• <code>/models</code> and <code>/ping</code> for runtime checks\n"
        "• <code>/reset</code> to clear rolling chat turns only\n\n"
        "<b>Important</b>\n"
        "• <code>/reset</code> does <i>not</i> wipe SQLite memories.\n"
        "• Full wipe (episodes, beliefs, relationships, etc.) is host-side:\n"
        "<code>bumblebee wipe &lt;entity&gt; --yes</code>."
    )


def format_commands_page(
    page: int,
    *,
    per_page: int = 5,
    query: str | None = None,
) -> tuple[str, int, int]:
    """
    page: 0-based index.
    query: optional case-insensitive filter over command, summary, usage, category.
    Returns (html_body, page_index, total_pages).
    """
    q = (query or "").strip().lower()
    rows = COMMAND_REGISTRY
    if q:
        rows = [
            c
            for c in COMMAND_REGISTRY
            if q in c.name.lower()
            or q in c.summary.lower()
            or q in c.usage.lower()
            or q in c.category.lower()
        ]
    if not rows:
        return (
            f"<b>Commands</b>\n\nNo matches for <code>{html.escape(query or '')}</code>.\n"
            "Try <code>/commands</code> for the full list.",
            0,
            1,
        )

    total = max(1, (len(rows) + per_page - 1) // per_page)
    page = max(0, min(page, total - 1))
    start = page * per_page
    page_rows = rows[start : start + per_page]

    title = "<b>Commands</b> — page {}/{}".format(page + 1, total)
    if q:
        title += f" · filter <code>{html.escape(query or '')}</code>"
    lines = [title, ""]
    for c in page_rows:
        lines.append(f"<b>/{html.escape(c.name)}</b> — {html.escape(c.summary)}")
        lines.append(f"<i>usage:</i> <code>{html.escape(c.usage)}</code>")
        lines.append("")
    if q:
        lines.append(
            "<i>Next page:</i> <code>/commands {} {}</code>".format(
                page + 2 if page + 1 < total else 1,
                html.escape(query or ""),
            )
        )
    else:
        lines.append(
            "<i>Next page:</i> <code>/commands {}</code>".format(
                page + 2 if page + 1 < total else 1
            )
        )
    return "\n".join(lines), page, total


async def build_status_html(entity: "Entity", app_version: str) -> str:
    async with entity.store.session() as db:
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
        f"◈ <b>{en}</b> · v{html.escape(app_version)}\n"
        f"mood · {html.escape(snap.mood_label)} · {html.escape(snap.drive_line)}\n"
        f"memory · {snap.episode_count} episodes · knows {snap.people_count} people\n"
        f"model · {html.escape(snap.reflex_model)} · context · {snap.max_context_tokens // 1000}k\n"
        f"tools · {snap.tool_count} active · awake · {html.escape(snap.awake_summary)}"
    )


def format_memories_html(entity_name: str, summaries: list[str]) -> str:
    en = html.escape(entity_name)
    if not summaries:
        return f"<i>{en}</i> reaches for recent memories; the shelf is still sparse."
    lines = [f"<i>{en}</i> leafs through recent memories…", ""]
    for i, s in enumerate(summaries, 1):
        lines.append(f"{i}. {html.escape(s[:500])}")
    return "\n".join(lines)


def format_feelings_html(entity_name: str, state: Any) -> str:
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


def format_me_html(entity_name: str, relationship: Any | None) -> str:
    en = html.escape(entity_name)
    if relationship is None:
        return (
            f"<i>{en}</i> does not have a relationship profile for you yet.\n\n"
            "Talk a little more and then check again with <code>/me</code>."
        )
    first_met_ago = _fmt_duration(max(0.0, time.time() - float(relationship.first_met)))
    last_seen_ago = _fmt_duration(max(0.0, time.time() - float(relationship.last_interaction)))
    return (
        "<b>How I hold you in memory</b>\n"
        f"name · {html.escape(str(relationship.name) or 'unknown')}\n"
        f"dynamic · {html.escape(str(relationship.dynamic) or 'forming')}\n"
        f"familiarity · {relationship.familiarity:.2f}\n"
        f"warmth · {relationship.warmth:.2f}\n"
        f"trust · {relationship.trust:.2f}\n"
        f"interactions · {relationship.interaction_count}\n"
        f"first met · {html.escape(first_met_ago)} ago\n"
        f"last interaction · {html.escape(last_seen_ago)} ago"
    )


def format_models_html(entity: "Entity") -> str:
    cfg = entity.config
    think = "on" if cfg.cognition.thinking_mode else "off"
    return (
        "<b>Runtime models</b>\n"
        f"reflex · <code>{html.escape(cfg.cognition.reflex_model)}</code>\n"
        f"deliberate · <code>{html.escape(cfg.cognition.deliberate_model)}</code>\n"
        f"embedding · <code>{html.escape(cfg.harness.models.embedding)}</code>\n"
        f"thinking mode · {html.escape(think)}\n"
        f"context window · {cfg.cognition.max_context_tokens // 1000}k tokens\n"
        f"tools registered · {len(entity.tools.openai_tools())}"
    )


def format_ping_html(app_version: str) -> str:
    return (
        f"pong · bumblebee v{html.escape(app_version)}\n"
        f"server_time_unix · <code>{int(time.time())}</code>"
    )


def format_reset_html(entity_name: str) -> str:
    en = html.escape(entity_name)
    return (
        f"{en} cleared rolling chat turns for this runtime.\n\n"
        "Persistent SQLite memories are untouched."
    )


def format_unknown_command(command_text: str) -> str:
    cmd = html.escape(command_text.strip() or "that command")
    return (
        f"I do not recognize <code>{cmd}</code>.\n\n"
        "Use <code>/commands</code> to browse everything, or <code>/help</code> for quick guidance."
    )


def format_media_unavailable(kind: str) -> str:
    k = html.escape(kind)
    return (
        f"I cannot process <b>{k}</b> in this channel yet.\n"
        "Try plain text or a <b>photo</b> (caption optional)."
    )


def format_access_denied() -> str:
    return (
        "◈ This instance is locked to approved people.\n"
        "If you should have access, ask the operator to add your Telegram user id."
    )
