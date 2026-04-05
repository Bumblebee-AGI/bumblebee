"""Startup / liveness checks for API and operator tooling."""

from __future__ import annotations

from typing import Any

from bumblebee.config import EntityConfig


async def check_inference(entity: EntityConfig, client: Any) -> dict[str, Any]:
    h = await client.health()
    return {"subsystem": "inference", **h}


async def check_database(store: Any) -> dict[str, Any]:
    try:
        async with store.session() as conn:
            n = await store.count_episodes(conn)
        return {"subsystem": "database", "ok": True, "dialect": getattr(store, "dialect", "unknown"), "episodes": n}
    except Exception as e:
        return {"subsystem": "database", "ok": False, "error": str(e)[:400]}
