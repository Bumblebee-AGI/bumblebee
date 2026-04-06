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
        name="tools",
        summary="List all currently active tools for this entity",
        usage="/tools",
        category="Runtime",
    ),
    TelegramCommandSpec(
        name="routines",
        summary="Saved scheduled routines (automations) and their status",
        usage="/routines",
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
    TelegramCommandSpec(
        name="whoami",
        summary="Show your Telegram user id (for operator / privacy setup)",
        usage="/whoami",
        category="Session",
    ),
    TelegramCommandSpec(
        name="privacy",
        summary="Lock the bot to allowed users, or open it again (operators)",
        usage="/privacy [status|lock|open|allow ID|deny ID|help]",
        category="Session",
    ),
    TelegramCommandSpec(
        name="private",
        summary="Turn private mode on or off (operators; same as privacy lock/open)",
        usage="/private on  or  /private off",
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
        f"🐝 <b>{en}</b> · bumblebee v{html.escape(app_version)}\n\n"
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
        "• <code>/models</code>, <code>/tools</code>, <code>/routines</code>, and <code>/ping</code> for runtime checks\n"
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
        f"🐝 <b>{en}</b> · v{html.escape(app_version)}\n"
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


def _relative_time(ts: float | None) -> str:
    """Unix time (seconds) → shortest relative label for a past instant."""
    if ts is None or ts <= 0:
        return "—"
    delta = max(0.0, time.time() - float(ts))
    if delta < 45:
        return "just now"
    if delta < 3600:
        m = max(1, int(delta // 60))
        return f"{m}m ago"
    if delta < 86400:
        h = max(1, int(delta // 3600))
        return f"{h}h ago"
    if delta < 7 * 86400:
        d = max(1, int(delta // 86400))
        return f"{d}d ago"
    w = max(1, int(delta // (7 * 86400)))
    return f"{w}w ago"


def _routine_emoji(automation: dict[str, Any], is_paused: bool) -> str:
    if is_paused:
        return "⏸"
    origin = str(automation.get("origin", "") or "").strip().lower()
    name = str(automation.get("name", "") or "")
    desc = str(automation.get("description", "") or "")
    text = f"{name} {desc}".lower()
    if origin == "internal":
        return "📓"
    if origin == "user":
        return "⏰"
    # self-created
    if any(k in text for k in ("check in", "check-in", "reach out", "how they", "deadline")):
        return "💬"
    if any(k in text for k in ("reddit", "browse", "subreddit", "r/")):
        return "🔶"
    if any(k in text for k in ("watch", "monitor")):
        return "👀"
    if any(k in text for k in ("news", "headline")):
        return "⏰"
    if any(k in text for k in ("reflect", "journal")):
        return "📓"
    return "⏰"


def _possessive_entity_title(entity_name: str) -> str:
    n = (entity_name or "").strip() or "Entity"
    esc = html.escape(n)
    if n.lower().endswith("s"):
        return f"{esc}'"
    return f"{esc}'s"


def _origin_display_label(r: Any) -> str:
    o = getattr(r, "origin", None)
    v = o.value if o is not None and hasattr(o, "value") else str(o or "").strip().lower()
    if v == "internal":
        return "Internal"
    if v == "self":
        return "Self-created"
    if v == "user":
        return "User-created"
    return v[:1].upper() + v[1:] if v else "Unknown"


def _delivery_platform_label(r: Any) -> str | None:
    raw = getattr(r, "deliver_to", None)
    plat_attr = getattr(r, "deliver_platform", None)
    s = (str(raw).strip() if raw else "") or ""
    plat = ""
    if ":" in s:
        plat = s.split(":", 1)[0].strip().lower()
    elif plat_attr:
        plat = str(plat_attr).strip().lower()
    if plat == "telegram":
        return "Telegram"
    if plat == "discord":
        return "Discord"
    if plat == "cli":
        return "CLI"
    return None


def _automation_to_emoji_dict(r: Any) -> dict[str, Any]:
    o = getattr(r, "origin", None)
    ov = o.value if o is not None and hasattr(o, "value") else str(o or "")
    return {
        "origin": ov,
        "name": str(getattr(r, "name", "") or ""),
        "description": str(getattr(r, "description", "") or ""),
    }


def _format_one_routine_html(r: Any) -> str:
    enabled = bool(getattr(r, "enabled", False))
    is_paused = not enabled
    name_raw = str(getattr(r, "name", "") or "").strip() or "Untitled routine"
    natural = (getattr(r, "schedule_natural", None) or "").strip() or "Schedule not set"
    desc = (getattr(r, "description", None) or "").strip()
    runs = int(getattr(r, "run_count", 0) or 0)
    last_run = getattr(r, "last_run", None)
    lr = float(last_run) if last_run is not None else None

    emoji = _routine_emoji(_automation_to_emoji_dict(r), is_paused)
    title = f"<b>{emoji}  {html.escape(name_raw)}</b>"

    delivery = _delivery_platform_label(r)
    origin_lbl = _origin_display_label(r)
    if delivery:
        sched_line = f"    {html.escape(natural)} → {delivery}"
    else:
        sched_line = f"    {html.escape(natural)} · {origin_lbl}"
    if is_paused:
        sched_line = f"{sched_line} · Paused"

    if runs == 0:
        stats_line = "    Never run"
    else:
        last_part = _relative_time(lr) if lr else "—"
        ran_what = "Ran 1 time" if runs == 1 else f"Ran {runs} times"
        stats_line = f"    {ran_what} · Last: {last_part}"

    o = getattr(r, "origin", None)
    ov = o.value if o is not None and hasattr(o, "value") else str(o or "")
    show_desc = bool(desc) and ov in ("user", "self")
    desc_block = ""
    if show_desc:
        short = desc if len(desc) <= 80 else desc[:77] + "…"
        desc_block = f"\n\n    \"{html.escape(short)}\""

    return f"{title}\n{sched_line}\n{stats_line}{desc_block}"


def format_routines_html(
    entity_name: str,
    routines: list[Any],
    *,
    automations_enabled: bool,
    scheduler_ready: bool,
) -> str:
    """Render saved automations (routines) for Telegram HTML — scannable, no raw cron."""
    poss = _possessive_entity_title(entity_name)
    header = f"🐝 {poss} Routines"

    if not routines:
        body = (
            f"{header}\n\n"
            "No routines yet.\n\n"
            "Your entity can create its own — or ask it\n"
            "to set one up for you."
        )
    else:
        n_active = sum(1 for r in routines if getattr(r, "enabled", False))
        n_paused = len(routines) - n_active

        def _sort_key(r: Any) -> tuple[int, float]:
            en = bool(getattr(r, "enabled", False))
            nxt = getattr(r, "next_run", None)
            try:
                nxt_f = float(nxt) if nxt is not None else float("inf")
            except (TypeError, ValueError):
                nxt_f = float("inf")
            return (0 if en else 1, nxt_f)

        ordered = sorted(routines, key=_sort_key)
        blocks = [_format_one_routine_html(r) for r in ordered]
        body = f"{header}\n\n{n_active} active · {n_paused} paused\n\n" + "\n\n".join(blocks)

    footers: list[str] = []
    if not automations_enabled:
        footers.append(
            "ℹ️ Automations are configured but scheduling\n"
            "   is disabled in this entity's config."
        )
    if not scheduler_ready:
        footers.append(
            "ℹ️ Routines are saved but won't fire until\n"
            "   the daemon is running (bumblebee run)."
        )
    if footers:
        body = body + "\n\n" + "\n\n".join(footers)
    return body


def format_tools_html(entity: "Entity") -> str:
    """Render active runtime tools for this entity, including dynamic MCP tools."""
    rows: list[tuple[str, str]] = []
    list_fn = getattr(entity.tools, "list_tools", None)
    if callable(list_fn):
        rows = list_fn()
    else:
        for t in entity.tools.openai_tools():
            fn = t.get("function", {}) if isinstance(t, dict) else {}
            rows.append(
                (
                    str(fn.get("name") or ""),
                    str(fn.get("description") or ""),
                )
            )
        rows = sorted([r for r in rows if r[0]], key=lambda r: r[0])

    lines = [f"<b>Active tools</b> · {len(rows)} total", ""]
    if not rows:
        lines.append("No tools are currently registered.")
        return "\n".join(lines)

    for name, desc in rows:
        summary = (desc or "").strip()
        if len(summary) > 180:
            summary = summary[:177] + "..."
        lines.append(f"• <code>{html.escape(name)}</code>")
        if summary:
            lines.append(f"  {html.escape(summary)}")
    return "\n".join(lines)


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
        "🐝 This instance is locked to approved people.\n"
        "If you should have access, ask the operator to add your Telegram user id."
    )


def format_whoami_html(*, user_id: int, full_name: str, username: str | None) -> str:
    uid = html.escape(str(user_id))
    fn = html.escape(full_name.strip() or "—")
    un = f"@{html.escape(username.strip())}" if (username or "").strip() else "—"
    return (
        "<b>Your Telegram id</b>\n"
        f"<code>{uid}</code>\n\n"
        f"<b>Name</b> {fn}\n"
        f"<b>Username</b> {un}\n\n"
        "Share the id with the host if they need to list you under "
        "<code>operator_user_ids</code> or use <code>/privacy allow</code>."
    )


def format_privacy_help_html() -> str:
    return (
        "<b>Privacy</b>\n\n"
        "<b>Read</b>\n"
        "• <code>/privacy</code> or <code>/privacy status</code>\n\n"
        "<b>Operators only</b> (see entity YAML <code>operator_user_ids</code>)\n"
        "• <code>/private on</code> / <code>/private off</code> — quick close or open the bot\n"
        "• <code>/privacy lock</code> — same as <code>/private on</code>\n"
        "• <code>/privacy allow &lt;id&gt;</code> — add a user id\n"
        "• <code>/privacy deny &lt;id&gt;</code> — remove a user id\n"
        "• <code>/privacy open</code> — same as <code>/private off</code>\n"
        "• <code>/privacy help</code>\n\n"
        "Use <code>/whoami</code> to see your numeric id."
    )


def format_private_usage_html() -> str:
    return (
        "<b>Private mode</b> (operators only)\n\n"
        "• <code>/private on</code> — only operators (and anyone you <code>/privacy allow</code>) can chat\n"
        "• <code>/private off</code> — public again (unless YAML <code>allowed_user_ids</code> is set)\n\n"
        "Same as <code>/privacy lock</code> and <code>/privacy open</code>."
    )


def format_privacy_operator_required_html() -> str:
    return (
        "Only configured <b>operators</b> can change privacy.\n"
        "The host must set <code>operator_user_ids: [ … ]</code> on the Telegram platform in your entity YAML."
    )


def format_privacy_no_operators_html() -> str:
    return (
        "Telegram privacy commands are disabled until the host sets "
        "<code>operator_user_ids</code> (non-empty list of Telegram user ids) in the entity YAML.\n\n"
        "Use <code>/whoami</code> to read your id."
    )


def format_privacy_status_html(
    *,
    enforced: bool,
    allowed_ids: list[int],
    yaml_restricted: bool,
    operators_configured: bool,
) -> str:
    lines = [
        "<b>Privacy status</b>\n",
        f"• <b>DB lock</b>: {'<code>on</code>' if enforced else '<code>off</code>'}\n",
    ]
    if enforced:
        preview = ", ".join(html.escape(str(i)) for i in sorted(allowed_ids)[:24])
        if len(allowed_ids) > 24:
            preview += ", …"
        lines.append(f"• <b>Allowed user ids</b>: {preview or '—'}\n")
    lines.append(
        f"• <b>YAML allowlist</b>: {'<code>yes</code>' if yaml_restricted else '<code>no</code>'} "
        f"(ignored while DB lock is on)\n"
    )
    lines.append(
        f"• <b>Operators in YAML</b>: {'<code>yes</code>' if operators_configured else '<code>no</code>'}\n"
    )
    if not enforced and not yaml_restricted:
        lines.append("\nAnyone can use this bot until you <code>/privacy lock</code>.\n")
    return "".join(lines)


def format_privacy_locked_html(count: int) -> str:
    return (
        f"🔒 <b>Locked.</b> {count} Telegram user id(s) can use this bot.\n"
        "Use <code>/privacy allow</code> / <code>/deny</code> to adjust, or <code>/privacy open</code> to go public."
    )


def format_privacy_opened_html() -> str:
    return (
        "🔓 <b>Open.</b> Anyone can chat again "
        "(unless you still have a static <code>allowed_user_ids</code> list in YAML).\n"
        "Use <code>/privacy lock</code> to restrict."
    )


def format_privacy_allow_deny_usage_html() -> str:
    return "Usage: <code>/privacy allow &lt;telegram_user_id&gt;</code> or <code>/privacy deny &lt;id&gt;</code>"


def format_privacy_invalid_id_html() -> str:
    return "That doesn’t look like a numeric Telegram user id."


def format_privacy_cannot_deny_last_html() -> str:
    return "Refusing to remove the last allowed user while locked — use <code>/privacy open</code> first."
