"""Private journal read/write."""

from __future__ import annotations

import json

from bumblebee.presence.tools.registry import tool
from bumblebee.presence.tools.runtime import require_tool_runtime


@tool(
    name="read_journal",
    description="Read your own journal entries (private reflections from routines).",
)
async def read_journal(n: int = 5) -> str:
    ctx = require_tool_runtime()
    ent = ctx.entity
    if not ent._tool_enabled("journal", True) or not ent.config.automations.journal.enabled:
        return json.dumps({"error": "journal disabled"})
    k = max(1, min(30, int(n or 5)))
    chunks = await ent.journal.read_recent(k)
    return json.dumps({"entries": chunks}, ensure_ascii=False)


@tool(
    name="write_journal",
    description="Write a private journal entry — reflection, processing, or a note to your future self.",
)
async def write_journal(content: str) -> str:
    ctx = require_tool_runtime()
    ent = ctx.entity
    if not ent._tool_enabled("journal", True) or not ent.config.automations.journal.enabled:
        return json.dumps({"ok": False, "error": "journal disabled"})
    raw = (content or "").strip()
    if not raw:
        return json.dumps({"ok": False, "error": "empty content"})
    await ent.journal.write_entry(raw, tags=["manual"])
    return json.dumps({"ok": True}, ensure_ascii=False)
