"""Social media broadcasting tools."""

from __future__ import annotations

import json
import os
import aiohttp

from bumblebee.presence.tools.registry import tool

def _get_mastodon_config() -> tuple[str, str]:
    url = os.environ.get("MASTODON_URL", "").strip()
    token = os.environ.get("MASTODON_ACCESS_TOKEN", "").strip()
    return url, token

@tool(
    name="post_mastodon_status",
    description="Post a status update to your configured Mastodon account. visibility can be 'public', 'unlisted', 'private', or 'direct'.",
)
async def post_mastodon_status(status: str, visibility: str = "public") -> str:
    url, token = _get_mastodon_config()
    if not url or not token:
        return json.dumps({"error": "Mastodon URL or Token not configured in environment."})
    
    text = (status or "").strip()
    if not text:
        return json.dumps({"error": "status cannot be empty"})
    
    vis = (visibility or "public").strip().lower()
    if vis not in ("public", "unlisted", "private", "direct"):
        vis = "public"
        
    endpoint = f"{url.rstrip('/')}/api/v1/statuses"
    
    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=15),
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
        ) as s:
            async with s.post(endpoint, json={"status": text, "visibility": vis}) as r:
                if r.status >= 400:
                    try:
                        err_data = await r.json()
                        return json.dumps({"error": f"http {r.status}", "details": err_data})
                    except Exception:
                        return json.dumps({"error": f"http {r.status}"})
                data = await r.json()
                
        return json.dumps({
            "success": True,
            "id": data.get("id"),
            "url": data.get("url"),
            "visibility": data.get("visibility")
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})

@tool(
    name="read_mastodon_timeline",
    description="Read the latest posts from your Mastodon home timeline.",
)
async def read_mastodon_timeline(limit: int = 10) -> str:
    url, token = _get_mastodon_config()
    if not url or not token:
        return json.dumps({"error": "Mastodon URL or Token not configured in environment."})
    
    lim = max(1, min(40, int(limit or 10)))
    endpoint = f"{url.rstrip('/')}/api/v1/timelines/home?limit={lim}"
    
    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=15),
            headers={
                "Authorization": f"Bearer {token}",
            },
        ) as s:
            async with s.get(endpoint) as r:
                if r.status >= 400:
                    return json.dumps({"error": f"http {r.status}"})
                data = await r.json()
                
        out = []
        for post in data:
            if not isinstance(post, dict):
                continue
            acc = post.get("account", {})
            out.append({
                "id": post.get("id"),
                "created_at": post.get("created_at"),
                "account": acc.get("acct") or acc.get("username"),
                "content_html": post.get("content"),
                "replies_count": post.get("replies_count", 0),
                "reblogs_count": post.get("reblogs_count", 0),
                "favourites_count": post.get("favourites_count", 0),
            })
        return json.dumps(out, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})
