"""Browser automation tools routed through execution backend."""

from __future__ import annotations

import json

from bumblebee.presence.tools.execution_rpc import get_execution_client
from bumblebee.presence.tools.registry import tool
from bumblebee.presence.tools.runtime import require_tool_runtime


def _session_id() -> str:
    ctx = require_tool_runtime()
    return str(ctx.state.get("browser_session_id") or "").strip()


def _set_session_id(v: str) -> None:
    ctx = require_tool_runtime()
    if v:
        ctx.state["browser_session_id"] = v


def _json(res: dict) -> str:
    return json.dumps(res, ensure_ascii=False)


@tool(
    name="browser_navigate",
    description="Open a URL in a real browser and get the rendered page content. Unlike fetch_url, this runs JavaScript and handles SPAs.",
)
async def browser_navigate(url: str) -> str:
    u = (url or "").strip()
    if not u:
        return json.dumps({"error": "empty url"})
    client = get_execution_client()
    payload = {"url": u}
    sid = _session_id()
    if sid:
        payload["session_id"] = sid
    res = await client.call("browser_navigate", payload)
    if isinstance(res, dict) and res.get("session_id"):
        _set_session_id(str(res["session_id"]))
    return _json(res if isinstance(res, dict) else {"ok": False, "error": "invalid result"})


@tool(
    name="browser_screenshot",
    description="Take a screenshot of a web page so you can see it",
)
async def browser_screenshot(url: str = "") -> str:
    client = get_execution_client()
    payload: dict[str, str] = {}
    u = (url or "").strip()
    if u:
        payload["url"] = u
    sid = _session_id()
    if sid:
        payload["session_id"] = sid
    res = await client.call("browser_screenshot", payload)
    if isinstance(res, dict) and res.get("session_id"):
        _set_session_id(str(res["session_id"]))
    return _json(res if isinstance(res, dict) else {"ok": False, "error": "invalid result"})


@tool(
    name="browser_click",
    description="Click an element on the current browser page",
)
async def browser_click(selector: str) -> str:
    sel = (selector or "").strip()
    if not sel:
        return json.dumps({"error": "empty selector"})
    client = get_execution_client()
    payload = {"selector": sel}
    sid = _session_id()
    if sid:
        payload["session_id"] = sid
    res = await client.call("browser_click", payload)
    if isinstance(res, dict) and res.get("session_id"):
        _set_session_id(str(res["session_id"]))
    return _json(res if isinstance(res, dict) else {"ok": False, "error": "invalid result"})


@tool(
    name="browser_type",
    description="Type text into a form field on the current browser page",
)
async def browser_type(selector: str, text: str) -> str:
    sel = (selector or "").strip()
    if not sel:
        return json.dumps({"error": "empty selector"})
    client = get_execution_client()
    payload = {"selector": sel, "text": text or ""}
    sid = _session_id()
    if sid:
        payload["session_id"] = sid
    res = await client.call("browser_type", payload)
    if isinstance(res, dict) and res.get("session_id"):
        _set_session_id(str(res["session_id"]))
    return _json(res if isinstance(res, dict) else {"ok": False, "error": "invalid result"})
