"""Filesystem tools: local read-only helpers + dangerous writes via execution backend."""

from __future__ import annotations

import json
from pathlib import Path

from bumblebee.presence.tools.execution_rpc import get_execution_client
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


@tool(
    name="list_directory",
    description="See what files and folders are in a directory.",
)
async def list_directory(path: str = ".") -> str:
    p = Path(path).expanduser()
    try:
        p = p.resolve()
    except OSError as e:
        return json.dumps({"error": str(e)})
    if not p.exists():
        return json.dumps({"error": f"path not found: {p}"})
    if not p.is_dir():
        return json.dumps({"error": f"not a directory: {p}"})
    try:
        out: list[dict[str, str | int]] = []
        for child in sorted(p.iterdir(), key=lambda c: (not c.is_dir(), c.name.lower()))[:500]:
            kind = "dir" if child.is_dir() else "file"
            size = child.stat().st_size if child.is_file() else 0
            out.append({"name": child.name, "kind": kind, "size": int(size)})
        return json.dumps({"path": str(p), "entries": out}, ensure_ascii=False)
    except OSError as e:
        return json.dumps({"error": str(e)})


@tool(
    name="search_files",
    description="Search for files matching a pattern.",
)
async def search_files(directory: str, pattern: str) -> str:
    d = Path(directory).expanduser()
    try:
        d = d.resolve()
    except OSError as e:
        return json.dumps({"error": str(e)})
    if not d.exists() or not d.is_dir():
        return json.dumps({"error": f"directory not found: {d}"})
    pat = (pattern or "").strip() or "*"
    try:
        matches = [str(p) for p in d.rglob(pat) if p.is_file()][:1000]
        return json.dumps({"directory": str(d), "pattern": pat, "matches": matches}, ensure_ascii=False)
    except OSError as e:
        return json.dumps({"error": str(e)})


@tool(
    name="write_file",
    description="Create or overwrite a file.",
)
async def write_file(path: str, content: str) -> str:
    client = get_execution_client()
    res = await client.call("write_file", {"path": path, "content": content})
    return json.dumps(res, ensure_ascii=False)


@tool(
    name="append_file",
    description="Add content to the end of an existing file.",
)
async def append_file(path: str, content: str) -> str:
    client = get_execution_client()
    res = await client.call("append_file", {"path": path, "content": content})
    return json.dumps(res, ensure_ascii=False)


async def read_file_safe(path: str, max_bytes: int = 32000) -> str:
    """Backward-compatible name for tests and legacy callers."""
    return await read_file(path, max_bytes=max_bytes)
