"""Send a workspace file to the user as a Telegram/Discord attachment."""

from __future__ import annotations

import json
import mimetypes
from pathlib import PurePosixPath

from bumblebee.presence.tools.registry import tool
from bumblebee.presence.tools.runtime import require_tool_runtime
from bumblebee.presence.tools.execution_rpc import (
    get_execution_client,
    read_only_workspace_fs_allowed,
    local_tool_block_message,
)


@tool(
    "send_file",
    "Send a file from your workspace to the user as an attachment in the current "
    "chat. Works on Telegram and Discord. The user receives the file as a download.",
)
async def send_file(path: str, message: str = "") -> str:
    ctx = require_tool_runtime()
    if not read_only_workspace_fs_allowed(ctx.entity):
        return json.dumps({"error": local_tool_block_message(ctx.entity)})

    platform = ctx.platform
    if platform is None:
        return json.dumps({"error": "no active platform to send files to"})

    inp = ctx.inp
    if inp is None:
        return json.dumps({"error": "no active conversation"})

    client = get_execution_client()
    res = await client.call("read_file", {"path": path or ".", "max_bytes": 256_000})
    if not res.get("ok"):
        return json.dumps({"error": res.get("error") or "could not read file"})

    content = str(res.get("content") or "")
    if not content:
        return json.dumps({"error": "file is empty"})

    data = content.encode("utf-8")
    filename = PurePosixPath(path).name or "file.txt"
    content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"

    send_attach = getattr(platform, "send_attachment_bytes", None)
    if not callable(send_attach):
        return json.dumps({"error": f"platform {inp.platform} does not support file attachments"})

    try:
        await send_attach(
            inp.channel,
            data,
            content_type=content_type,
            filename=filename,
        )
    except Exception as e:
        return json.dumps({"error": f"failed to send: {e}"})

    if message:
        try:
            await platform.send_message(inp.channel, message)
        except Exception:
            pass

    return json.dumps({"ok": True, "sent": filename, "size_bytes": len(data)})
