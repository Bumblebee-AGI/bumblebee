"""Rich renderers for the CLI — startup, shutdown, introspection (kept separate from I/O loop)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from rich.console import Console
from rich.rule import Rule
from rich.text import Text

from bumblebee.identity.drives import Drive
from bumblebee.models import EmotionalState


# Restrained palette: warm gold for identity, dim gray for labels, default for values.
STYLE_MARK = "dim #b8a06a"
STYLE_ENTITY = "bold #d4a656"
STYLE_LABEL = "dim"
STYLE_VALUE = ""
STYLE_RULE = "dim"
STYLE_ERROR = "#c45c5c"
STYLE_WHISPER = "dim italic"


@dataclass
class CLIHeaderSnapshot:
    app_version: str
    entity_name: str
    mood_label: str
    drive_line: str
    episode_count: int
    reflex_model: str
    max_context_tokens: int
    tool_count: int
    awake_summary: str
    people_count: int


@dataclass
class SessionShutdownSummary:
    entity_name: str
    duration_seconds: float
    exchange_count: int
    mood_start: str
    mood_end: str
    episodes_saved: int


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)} seconds"
    if seconds < 3600:
        m = int(seconds // 60)
        return f"{m} minute{'s' if m != 1 else ''}"
    if seconds < 86400 * 2:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        if m:
            return f"{h}h {m}m"
        return f"{h} hour{'s' if h != 1 else ''}"
    d = int(seconds // 86400)
    return f"{d} day{'s' if d != 1 else ''}"


def _parse_created_iso(raw: object) -> float | None:
    if not raw or not isinstance(raw, str):
        return None
    s = raw.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except ValueError:
        return None


def compute_awake_summary(
    *,
    created_raw: object,
    first_episode_ts: float | None,
    now: float | None = None,
) -> str:
    t = now or time.time()
    origin = _parse_created_iso(created_raw)
    if origin is None and first_episode_ts is not None:
        origin = first_episode_ts
    if origin is None:
        return "just awakened"
    return _fmt_duration(max(0.0, t - origin))


def _drive_word(level: float) -> str:
    if level < 0.22:
        return "quiet"
    if level < 0.48:
        return "stirring"
    if level < 0.72:
        return "building"
    return "pressing"


def dominant_drive_line(drives: list[Drive]) -> str:
    if not drives:
        return "drives: still"
    top = max(drives, key=lambda d: d.level)
    return f"{top.name}: {_drive_word(top.level)}"


def dominant_drive_toolbar(drives: list[Drive]) -> str:
    if not drives:
        return "inner drives are still"
    top = max(drives, key=lambda d: d.level)
    return f"{top.name} is {_drive_word(top.level)}"


def render_startup(console: Console, snap: CLIHeaderSnapshot) -> None:
    mark = Text("◈", style=STYLE_MARK)
    title = Text.assemble(
        mark,
        Text(" bumblebee ", style=STYLE_LABEL),
        Text(f"v{snap.app_version}", style=STYLE_VALUE),
    )
    console.print(title)
    console.print()
    console.print(Text(snap.entity_name, style=STYLE_ENTITY))
    line1 = Text.assemble(
        Text("mood: ", style=STYLE_LABEL),
        Text(snap.mood_label, style=STYLE_VALUE),
        Text(" · ", style=STYLE_LABEL),
        Text(snap.drive_line, style=STYLE_VALUE),
        Text(" · ", style=STYLE_LABEL),
        Text("memory: ", style=STYLE_LABEL),
        Text(f"{snap.episode_count} episodes", style=STYLE_VALUE),
    )
    line2 = Text.assemble(
        Text("model: ", style=STYLE_LABEL),
        Text(snap.reflex_model, style=STYLE_VALUE),
        Text(" · ", style=STYLE_LABEL),
        Text("context: ", style=STYLE_LABEL),
        Text(f"{snap.max_context_tokens // 1000}k", style=STYLE_VALUE),
        Text(" · ", style=STYLE_LABEL),
        Text("tools: ", style=STYLE_LABEL),
        Text(f"{snap.tool_count} active", style=STYLE_VALUE),
    )
    line3 = Text.assemble(
        Text("awake for ", style=STYLE_LABEL),
        Text(snap.awake_summary, style=STYLE_VALUE),
        Text(" · ", style=STYLE_LABEL),
        Text("knows ", style=STYLE_LABEL),
        Text(str(snap.people_count), style=STYLE_VALUE),
        Text(" people", style=STYLE_LABEL),
    )
    console.print(line1)
    console.print(line2)
    console.print(line3)
    console.print()
    console.print(Rule(style=STYLE_RULE, characters="─"))


def render_shutdown(console: Console, summary: SessionShutdownSummary) -> None:
    console.print()
    console.print(Rule(style=STYLE_RULE, characters="─"))
    console.print()
    rest = Text.assemble(
        Text(summary.entity_name, style=STYLE_ENTITY),
        Text(" is resting", style=STYLE_LABEL),
    )
    console.print(rest)
    dur = _fmt_duration(summary.duration_seconds)
    sess = Text.assemble(
        Text("session: ", style=STYLE_LABEL),
        Text(dur, style=STYLE_VALUE),
        Text(" · ", style=STYLE_LABEL),
        Text(f"{summary.exchange_count} exchanges", style=STYLE_VALUE),
        Text(" · ", style=STYLE_LABEL),
        Text("mood shifted: ", style=STYLE_LABEL),
        Text(f"{summary.mood_start} → {summary.mood_end}", style=STYLE_VALUE),
    )
    console.print(sess)
    mem = Text.assemble(
        Text("memory: ", style=STYLE_LABEL),
        Text(f"{summary.episodes_saved} new episodes saved", style=STYLE_VALUE),
    )
    console.print(mem)
    console.print()
    console.print(Text("◈", style=STYLE_MARK))
    console.print()


def render_error(console: Console, message: str) -> None:
    console.print(Text(message, style=STYLE_ERROR))


def render_memories_introspection(
    console: Console,
    entity_name: str,
    summaries: list[str],
) -> None:
    head = Text.assemble(
        Text(entity_name, style=STYLE_ENTITY),
        Text(" leafs through recent memories…", style=STYLE_WHISPER),
    )
    console.print(head)
    console.print()
    if not summaries:
        console.print(Text("…nothing much on the top of the stack yet.", style=STYLE_WHISPER))
        return
    for i, s in enumerate(summaries, 1):
        line = Text.assemble(Text(f"  {i}. ", style=STYLE_LABEL), Text(s, style=STYLE_VALUE))
        console.print(line)
    console.print()


def render_feelings_introspection(
    console: Console,
    entity_name: str,
    state: EmotionalState,
) -> None:
    primary = state.primary.value
    pi = f"{state.intensity:.1f}"
    parts: list[str] = [
        f"{entity_name} is feeling {primary} ({pi})",
    ]
    if state.secondary is not None:
        parts.append(
            f"with an undercurrent of {state.secondary.value} ({state.secondary_intensity:.1f})"
        )
    ago = _fmt_duration(max(0.0, time.time() - state.last_transition))
    parts.append(f"— been in this shade for about {ago}")
    body = " ".join(parts) + "."
    console.print(Text(body, style=STYLE_VALUE))
    console.print()


