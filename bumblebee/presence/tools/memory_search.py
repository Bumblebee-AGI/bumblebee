"""Semantic search over episodic memory (tool-facing wrapper around CLI recall)."""

from __future__ import annotations

import json

from bumblebee.models import EmotionCategory
from bumblebee.presence.tools.registry import tool
from bumblebee.presence.tools.runtime import require_tool_runtime


@tool(
    name="search_past_conversations",
    description=(
        "Search your episodic memory — past conversation summaries — by semantic similarity. "
        "Use when you need to recall what was discussed before, people mentioned, or prior context. "
        "Pass a natural-language query (not a keyword list). Results are narrative episodes, not raw chat logs."
    ),
)
async def search_past_conversations(query: str, limit: int = 8) -> str:
    ctx = require_tool_runtime()
    ent = ctx.entity
    q = (query or "").strip()
    if not q:
        return json.dumps({"error": "query is empty"}, ensure_ascii=False)
    lim = max(1, min(20, int(limit or 8)))
    model = ent.config.harness.models.embedding or ""
    if not model:
        return json.dumps({"error": "No embedding model configured."}, ensure_ascii=False)
    try:
        qe = await ent.client.embed(model, q)
        if not qe:
            return json.dumps({"error": "Embedding failed (inference unavailable?)."}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"embed failed: {e}"}, ensure_ascii=False)

    mood = ent.emotions.get_state().primary
    try:
        async with ent.store.session() as db:
            pairs = await ent.episodic.recall(
                db,
                q,
                qe,
                limit=lim,
                min_significance=0.0,
                current_mood=mood if mood != EmotionCategory.NEUTRAL else None,
            )
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

    rows: list[dict[str, object]] = []
    for ep, _imprints in pairs:
        rows.append(
            {
                "id": ep.id,
                "timestamp": ep.timestamp,
                "summary": (ep.summary or "")[:2000],
                "participants": ep.participants,
                "significance": ep.significance,
                "tags": ep.tags,
                "emotional_imprint": getattr(ep.emotional_imprint, "value", str(ep.emotional_imprint)),
            }
        )
    return json.dumps({"ok": True, "query": q, "results": rows}, ensure_ascii=False)
