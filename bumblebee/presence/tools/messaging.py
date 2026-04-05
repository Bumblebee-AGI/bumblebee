"""Cross-platform proactive messaging tool."""

from __future__ import annotations

import json

from bumblebee.presence.tools.registry import tool
from bumblebee.presence.tools.runtime import require_tool_runtime


@tool(
    name="send_message_to",
    description="Send a message to someone on a specific platform. Use this to reach out to people proactively — check in, share something you found, or continue a conversation on a different platform.",
)
async def send_message_to(platform: str, target: str, message: str) -> str:
    pf = (platform or "").strip().lower()
    tgt = (target or "").strip()
    msg = (message or "").strip()
    if not pf or not tgt or not msg:
        return json.dumps({"error": "platform, target, and message are required"})
    ctx = require_tool_runtime()
    entity = ctx.entity
    sender = getattr(entity, "send_message_to_platform", None)
    if not callable(sender):
        return json.dumps({"error": "entity messaging bridge unavailable"})
    try:
        await sender(pf, tgt, msg)
        return json.dumps({"ok": True, "platform": pf, "target": tgt}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e), "platform": pf, "target": tgt})
