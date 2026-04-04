"""Bounded local file read."""

from __future__ import annotations

import json
from pathlib import Path


async def read_file_safe(path: str, max_bytes: int = 32000) -> str:
    p = Path(path).expanduser().resolve()
    try:
        data = p.read_bytes()[:max_bytes]
        return data.decode("utf-8", errors="replace")
    except Exception as e:
        return json.dumps({"error": str(e)})
