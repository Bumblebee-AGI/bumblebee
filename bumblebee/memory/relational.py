"""Per-person relationship models."""

from __future__ import annotations

import json
import math
import time
from typing import Optional

from bumblebee.memory.store import MemoryStore
from bumblebee.models import Relationship


class RelationalMemory:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    async def get(self, conn, person_id: str) -> Optional[Relationship]:
        cur = await conn.execute("SELECT * FROM relationships WHERE person_id = ?", (person_id,))
        row = await cur.fetchone()
        if not row:
            return None
        return self._row_to_rel(row)

    def _row_to_rel(self, row) -> Relationship:
        return Relationship(
            person_id=row[0],
            name=row[1],
            first_met=float(row[2]),
            last_interaction=float(row[3]),
            interaction_count=int(row[4]),
            familiarity=float(row[5]),
            warmth=float(row[6]),
            trust=float(row[7]),
            dynamic=row[8] or "neutral",
            notes=json.loads(row[9] or "[]"),
            topics_shared=json.loads(row[10] or "[]"),
            unresolved=json.loads(row[11] or "[]"),
        )

    async def upsert_interaction(
        self,
        conn,
        person_id: str,
        name: str,
        *,
        warmth_delta: float = 0.0,
        trust_delta: float = 0.0,
        note: Optional[str] = None,
    ) -> Relationship:
        now = time.time()
        existing = await self.get(conn, person_id)
        if existing:
            n = existing.interaction_count + 1
            fam = min(1.0, math.log1p(n) / math.log1p(50))
            warmth = max(-1.0, min(1.0, existing.warmth + warmth_delta))
            trust = max(0.0, min(1.0, existing.trust + trust_delta))
            notes = list(existing.notes)
            if note:
                notes.append(note)
            rel = Relationship(
                person_id=person_id,
                name=name or existing.name,
                first_met=existing.first_met,
                last_interaction=now,
                interaction_count=n,
                familiarity=fam,
                warmth=warmth,
                trust=trust,
                dynamic=existing.dynamic,
                notes=notes[-50:],
                topics_shared=existing.topics_shared,
                unresolved=existing.unresolved,
            )
        else:
            rel = Relationship(
                person_id=person_id,
                name=name,
                first_met=now,
                last_interaction=now,
                interaction_count=1,
                familiarity=0.15,
                warmth=0.1 + warmth_delta,
                trust=0.2 + trust_delta,
                dynamic="forming",
                notes=[note] if note else [],
                topics_shared=[],
                unresolved=[],
            )
        await conn.execute(
            """INSERT OR REPLACE INTO relationships
            (person_id, name, first_met, last_interaction, interaction_count, familiarity,
             warmth, trust, dynamic, notes, topics_shared, unresolved)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                rel.person_id,
                rel.name,
                rel.first_met,
                rel.last_interaction,
                rel.interaction_count,
                rel.familiarity,
                rel.warmth,
                rel.trust,
                rel.dynamic,
                json.dumps(rel.notes),
                json.dumps(rel.topics_shared),
                json.dumps(rel.unresolved),
            ),
        )
        await conn.commit()
        return rel

    async def list_recent(self, conn, limit: int = 15) -> list[Relationship]:
        cur = await conn.execute(
            "SELECT * FROM relationships ORDER BY last_interaction DESC LIMIT ?",
            (limit,),
        )
        rows = await cur.fetchall()
        return [self._row_to_rel(r) for r in rows]

    def blurb(self, rel: Relationship) -> str:
        return (
            f"{rel.name} — familiarity {rel.familiarity:.2f}, warmth {rel.warmth:.2f}, "
            f"trust {rel.trust:.2f}. Dynamic: {rel.dynamic}. "
            f"Recent notes: {'; '.join(rel.notes[-3:])}"
        )
