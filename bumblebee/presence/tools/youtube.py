"""YouTube tools: transcript + search."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from urllib.parse import parse_qs, urlparse

from bumblebee.presence.tools.registry import tool
from bumblebee.presence.tools.web import _sync_ddgs_text_search


def _video_id_from_url(url: str) -> str:
    u = (url or "").strip()
    if not u:
        return ""
    try:
        p = urlparse(u)
    except ValueError:
        return ""
    host = (p.hostname or "").lower()
    if host.endswith("youtu.be"):
        return p.path.lstrip("/").split("/")[0]
    if "youtube.com" in host:
        if p.path == "/watch":
            q = parse_qs(p.query)
            return (q.get("v") or [""])[0]
        parts = [x for x in p.path.split("/") if x]
        if len(parts) >= 2 and parts[0] in ("shorts", "embed", "live"):
            return parts[1]
    return ""


@tool(
    name="get_youtube_transcript",
    description="Pull the transcript from a YouTube video so you can engage with what was said",
)
async def get_youtube_transcript(url: str) -> str:
    vid = _video_id_from_url(url)
    if not vid:
        return json.dumps({"error": "could not parse YouTube video id from URL"})
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        return json.dumps(
            {"error": "youtube-transcript-api not installed. Install with: pip install bumblebee[youtube]"}
        )
    try:
        rows = await asyncio.to_thread(YouTubeTranscriptApi().fetch, vid)
        chunks: list[str] = []
        for r in rows:
            txt = str(getattr(r, "text", "") or "").strip()
            if txt:
                chunks.append(txt)
        text = " ".join(chunks).strip()
        if len(text) > 30000:
            text = text[:29900] + "\n… [truncated]"
        return json.dumps({"video_id": vid, "transcript": text}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e), "video_id": vid})


@tool(
    name="search_youtube",
    description="Search YouTube for videos",
)
async def search_youtube(query: str, max_results: int = 5) -> str:
    q = (query or "").strip()
    if not q:
        return json.dumps({"error": "empty query"})
    max_results = max(1, min(int(max_results or 5), 20))
    scoped = f"site:youtube.com {q}"
    rows = await asyncio.to_thread(_sync_ddgs_text_search, scoped, max_results * 3)
    out: list[dict[str, Any]] = []
    for r in rows:
        href = str(r.get("href") or r.get("url") or "")
        if "youtube.com" not in href and "youtu.be" not in href:
            continue
        out.append(
            {
                "title": str(r.get("title") or "").strip(),
                "url": href,
                "description": str(r.get("body") or "").strip()[:400],
            }
        )
        if len(out) >= max_results:
            break
    return json.dumps({"query": q, "results": out}, ensure_ascii=False)
