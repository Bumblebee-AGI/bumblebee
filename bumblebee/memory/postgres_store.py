"""PostgreSQL relational store (asyncpg) with SQLite-compatible conn shim."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

import numpy as np

from bumblebee.memory.pg_translate import translate_sql
from bumblebee.memory.store import _unpack_embedding, cosine_sim
from bumblebee.models import EmotionCategory

try:
    import asyncpg
except ImportError as e:  # pragma: no cover - optional dependency
    asyncpg = None  # type: ignore[misc, assignment]
    _ASYNCPG_ERR = e
else:
    _ASYNCPG_ERR = None

SCHEMA_PG = """
CREATE TABLE IF NOT EXISTS episodes (
    id TEXT PRIMARY KEY,
    timestamp DOUBLE PRECISION,
    summary TEXT,
    participants TEXT,
    emotional_imprint TEXT,
    emotional_intensity DOUBLE PRECISION,
    significance DOUBLE PRECISION,
    tags TEXT,
    raw_context TEXT,
    self_reflection TEXT,
    embedding BYTEA
);

CREATE TABLE IF NOT EXISTS relationships (
    person_id TEXT PRIMARY KEY,
    name TEXT,
    first_met DOUBLE PRECISION,
    last_interaction DOUBLE PRECISION,
    interaction_count INTEGER,
    familiarity DOUBLE PRECISION,
    warmth DOUBLE PRECISION,
    trust DOUBLE PRECISION,
    dynamic TEXT,
    notes TEXT,
    topics_shared TEXT,
    unresolved TEXT
);

CREATE TABLE IF NOT EXISTS beliefs (
    id TEXT PRIMARY KEY,
    category TEXT,
    content TEXT,
    confidence DOUBLE PRECISION,
    source TEXT,
    formed_at DOUBLE PRECISION,
    last_reinforced DOUBLE PRECISION,
    times_reinforced INTEGER,
    embedding BYTEA
);

CREATE TABLE IF NOT EXISTS imprints (
    id TEXT PRIMARY KEY,
    episode_id TEXT,
    emotion TEXT,
    intensity DOUBLE PRECISION,
    trigger TEXT,
    embedding BYTEA
);

CREATE TABLE IF NOT EXISTS narrative (
    id TEXT PRIMARY KEY,
    timestamp DOUBLE PRECISION,
    self_story TEXT,
    trait_snapshot TEXT
);

CREATE TABLE IF NOT EXISTS inner_voice (
    id TEXT PRIMARY KEY,
    timestamp DOUBLE PRECISION,
    summary TEXT,
    emotional_context TEXT,
    embedding BYTEA
);

CREATE TABLE IF NOT EXISTS entity_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS evolution_log (
    id TEXT PRIMARY KEY,
    timestamp DOUBLE PRECISION,
    changes_json TEXT,
    reasoning TEXT
);

CREATE TABLE IF NOT EXISTS reminders (
    id TEXT PRIMARY KEY,
    message TEXT NOT NULL,
    due_at DOUBLE PRECISION NOT NULL,
    target_person TEXT,
    platform TEXT,
    channel TEXT,
    person_id TEXT,
    status TEXT NOT NULL,
    created_at DOUBLE PRECISION NOT NULL,
    fired_at DOUBLE PRECISION
);
"""

_EXPERIENCE_TABLES = (
    "imprints",
    "episodes",
    "relationships",
    "beliefs",
    "narrative",
    "inner_voice",
    "entity_state",
    "evolution_log",
    "reminders",
)


class PgCursor:
    def __init__(self, rows: list[Any]) -> None:
        self._rows = rows

    async def fetchall(self) -> list[tuple[Any, ...]]:
        return [tuple(r) for r in self._rows]

    async def fetchone(self) -> tuple[Any, ...] | None:
        if not self._rows:
            return None
        return tuple(self._rows[0])


class PgShimConnection:
    def __init__(self, raw: Any) -> None:
        self._c = raw

    async def execute(self, sql: str, params: tuple[Any, ...] | list[Any] = ()) -> PgCursor:
        sql_t = translate_sql(sql)
        p = tuple(params)
        head = sql_t.lstrip().split(None, 1)[0].upper() if sql_t.strip() else ""
        if head in ("SELECT", "WITH"):
            rows = await self._c.fetch(sql_t, *p)
            return PgCursor(list(rows))
        await self._c.execute(sql_t, *p)
        return PgCursor([])

    async def commit(self) -> None:
        return None


class PostgresMemoryStore:
    dialect: str = "postgres"

    def __init__(self, database_url: str, *, display_name: str | None = None) -> None:
        if asyncpg is None:
            raise RuntimeError(
                "PostgreSQL backend requires asyncpg. Install with: pip install bumblebee[railway]"
            ) from _ASYNCPG_ERR
        self._dsn = database_url
        self.db_path = display_name or "[postgresql]"
        self._pool: asyncpg.Pool | None = None

    async def _ensure_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            self._pool = await asyncpg.create_pool(self._dsn, min_size=1, max_size=10)
            async with self._pool.acquire() as conn:
                await conn.execute(SCHEMA_PG)
        return self._pool

    @asynccontextmanager
    async def session(self) -> AsyncIterator[PgShimConnection]:
        pool = await self._ensure_pool()
        async with pool.acquire() as raw:
            async with raw.transaction():
                yield PgShimConnection(raw)

    async def wipe_experience_tables(self, conn: PgShimConnection) -> None:
        for table in _EXPERIENCE_TABLES:
            await conn.execute(f"DELETE FROM {table}")

    async def aclose(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def search_episodes_by_embedding(
        self,
        conn: PgShimConnection,
        query_emb: list[float],
        limit: int = 5,
        min_significance: float = 0.0,
    ) -> list[tuple[str, float]]:
        cur = await conn.execute(
            "SELECT id, significance, embedding FROM episodes WHERE embedding IS NOT NULL"
        )
        rows = await cur.fetchall()
        scored: list[tuple[str, float]] = []
        for eid, sig, emb_blob in rows:
            if sig is not None and float(sig) < min_significance:
                continue
            emb = _unpack_embedding(bytes(emb_blob) if emb_blob is not None else None)
            if emb:
                scored.append((eid, cosine_sim(query_emb, emb)))
        scored.sort(key=lambda x: -x[1])
        return scored[:limit]

    async def search_episodes_by_embedding_biased(
        self,
        conn: PgShimConnection,
        query_emb: list[float],
        mood: EmotionCategory,
        now: float,
        *,
        imprint_weight: float,
        imprint_half_life: float,
        limit: int = 10,
        min_significance: float = 0.0,
    ) -> list[tuple[str, float]]:
        from bumblebee.memory.imprints import affinity_multiplier_for_recall

        cur = await conn.execute(
            "SELECT id, timestamp, significance, embedding FROM episodes WHERE embedding IS NOT NULL"
        )
        rows = await cur.fetchall()
        eids = [r[0] for r in rows]
        imprint_map: dict[str, list[tuple[str, float]]] = {e: [] for e in eids}
        if eids:
            ph = ",".join(f"${i + 1}" for i in range(len(eids)))
            sql = f"SELECT episode_id, emotion, intensity FROM imprints WHERE episode_id IN ({ph})"
            icur = await conn.execute(sql, eids)
            for ep_id, emo, intens in await icur.fetchall():
                imprint_map.setdefault(ep_id, []).append((str(emo), float(intens)))

        scored: list[tuple[str, float]] = []
        for eid, ts, sig, emb_blob in rows:
            if sig is not None and float(sig) < min_significance:
                continue
            emb = _unpack_embedding(bytes(emb_blob) if emb_blob is not None else None)
            if not emb:
                continue
            base = cosine_sim(query_emb, emb)
            age = max(0.0, now - float(ts))
            mem_decay = 0.5 ** (age / imprint_half_life) if imprint_half_life > 0 else 1.0
            imprint_bonus = 0.0
            for emo_str, intens in imprint_map.get(eid, []):
                aff = affinity_multiplier_for_recall(mood, emo_str)
                imprint_bonus += float(intens) * (aff - 1.0) * mem_decay
            imprint_bonus = min(1.5, imprint_bonus)
            adj = base * (1.0 + imprint_weight * imprint_bonus)
            scored.append((eid, adj))
        scored.sort(key=lambda x: -x[1])
        return scored[:limit]

    async def count_episodes(self, conn: PgShimConnection) -> int:
        cur = await conn.execute("SELECT COUNT(*) FROM episodes")
        row = await cur.fetchone()
        return int(row[0]) if row else 0

    async def count_relationships(self, conn: PgShimConnection) -> int:
        cur = await conn.execute("SELECT COUNT(*) FROM relationships")
        row = await cur.fetchone()
        return int(row[0]) if row else 0

    async def min_episode_timestamp(self, conn: PgShimConnection) -> float | None:
        cur = await conn.execute("SELECT MIN(timestamp) FROM episodes")
        row = await cur.fetchone()
        if row and row[0] is not None:
            return float(row[0])
        return None
