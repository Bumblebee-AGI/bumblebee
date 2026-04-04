"""Read-only file access — shared paths people point the entity at."""

from __future__ import annotations

import json
from pathlib import Path

from bumblebee.presence.tools.registry import tool


@tool(
    name="read_file",
    description=(
        "Read a local text or UTF-8 file the user shared or referenced (read-only). "
        "You are not a file manager — only open paths you were given."
    ),
)
async def read_file(path: str, max_bytes: int = 48000) -> str:
    p = Path(path).expanduser()
    try:
        p = p.resolve()
    except OSError as e:
        return json.dumps({"error": str(e)})
    try:
        data = p.read_bytes()[: max(1024, min(max_bytes, 256_000))]
        return data.decode("utf-8", errors="replace")
    except OSError as e:
        return json.dumps({"error": str(e)})


async def read_file_safe(path: str, max_bytes: int = 32000) -> str:
    """Backward-compatible name for tests and legacy callers."""
    return await read_file(path, max_bytes=max_bytes)
