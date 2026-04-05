"""Cross-platform proactive messaging tool."""

from __future__ import annotations

import json

from bumblebee.presence.tools.registry import tool
from bumblebee.presence.tools.runtime import require_tool_runtime


def _known_contacts(entity: object, platform: str = "") -> list[dict[str, object]]:
    lister = getattr(entity, "list_known_person_routes", None)
    if not callable(lister):
        return []
    try:
        rows = lister(platform)
    except Exception:
        rows = []
    out: list[dict[str, object]] = []
    for r in rows:
        out.append(
            {
                "person_name": str(r.get("person_name") or ""),
                "person_id": str(r.get("person_id") or ""),
                "platform": str(r.get("platform") or ""),
                "chat_type": str(r.get("chat_type") or ""),
                "target": str(r.get("channel") or ""),
                "last_seen_at": float(r.get("at") or 0.0),
            }
        )
    return out


@tool(
    name="send_message_to",
    description="Send a message to someone on a platform. You can provide explicit platform+target, or target_person to resolve from known contacts. For target_person, first call returns a confirmation payload; call again with confirm=true to actually send.",
)
async def send_message_to(
    message: str,
    platform: str = "",
    target: str = "",
    target_person: str = "",
    confirm: bool = False,
    prefer_private: bool = True,
) -> str:
    msg = (message or "").strip()
    if not msg:
        return json.dumps({"error": "message is required"})

    pf = (platform or "").strip().lower()
    if pf == "auto":
        pf = ""
    tgt = (target or "").strip()
    person = (target_person or "").strip()

    ctx = require_tool_runtime()
    entity = ctx.entity

    # Friendly shorthand: if only target is provided with no platform, treat it as a person reference.
    if not person and tgt and not pf:
        person = tgt
        tgt = ""

    resolved: dict[str, object] | None = None
    if person:
        resolver = getattr(entity, "resolve_person_route", None)
        if not callable(resolver):
            return json.dumps({"error": "entity route resolver unavailable"})
        route = resolver(person, platform=pf, prefer_private=bool(prefer_private))
        if not isinstance(route, dict):
            return json.dumps(
                {
                    "ok": False,
                    "error": f"unknown target_person: {person}",
                    "known_contacts": _known_contacts(entity, pf),
                },
                ensure_ascii=False,
            )
        pf = str(route.get("platform") or "").strip().lower()
        tgt = str(route.get("channel") or "").strip()
        resolved = {
            "person_name": str(route.get("person_name") or ""),
            "person_id": str(route.get("person_id") or ""),
            "chat_type": str(route.get("chat_type") or ""),
        }
        if not bool(confirm):
            return json.dumps(
                {
                    "ok": False,
                    "needs_confirmation": True,
                    "platform": pf,
                    "target": tgt,
                    "resolved": resolved,
                    "message_preview": msg[:300],
                    "instruction": (
                        "Call send_message_to again with the same arguments and confirm=true to send."
                    ),
                },
                ensure_ascii=False,
            )

    if not pf or not tgt:
        return json.dumps(
            {
                "error": "Need destination. Provide platform+target, or provide target_person.",
                "known_contacts": _known_contacts(entity, pf),
            },
            ensure_ascii=False,
        )

    sender = getattr(entity, "send_message_to_platform", None)
    if not callable(sender):
        return json.dumps({"error": "entity messaging bridge unavailable"})
    try:
        await sender(pf, tgt, msg)
        return json.dumps(
            {"ok": True, "platform": pf, "target": tgt, "resolved": resolved},
            ensure_ascii=False,
        )
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e), "platform": pf, "target": tgt})


@tool(
    name="list_known_contacts",
    description="List known people/routes this entity can currently message based on prior interactions.",
)
async def list_known_contacts(platform: str = "") -> str:
    ctx = require_tool_runtime()
    entity = ctx.entity
    pf = (platform or "").strip().lower()
    return json.dumps(
        {
            "contacts": _known_contacts(entity, pf),
            "platform_filter": pf,
        },
        ensure_ascii=False,
    )
