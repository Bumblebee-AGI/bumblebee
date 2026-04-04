"""Web fetch (minimal; search stub)."""

from __future__ import annotations

import json
from urllib.parse import quote_plus

import aiohttp


async def fetch_url(url: str, timeout: float = 15.0) -> str:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                text = await resp.text()
                return text[:8000]
    except Exception as e:
        return json.dumps({"error": str(e)})


async def search_duckduckgo_lite(query: str) -> str:
    """Lightweight HTML scrape — best-effort, no API key."""
    q = quote_plus(query)
    url = f"https://lite.duckduckgo.com/lite/?q={q}"
    html = await fetch_url(url, timeout=20.0)
    if len(html) < 50:
        return json.dumps({"note": "no results", "query": query})
    return json.dumps({"query": query, "snippet": html[:2000]})
