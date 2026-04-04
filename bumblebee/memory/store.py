"""SQLite storage + numpy cosine similarity for embeddings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import aiosqlite
import numpy as np

from bumblebee.models import EmotionCategory

SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    id TEXT PRIMARY KEY,
    timestamp REAL,
    summary TEXT,
    participants TEXT,
    emotional_imprint TEXT,
    emotional_intensity REAL,
    significance REAL,
    tags TEXT,
    raw_context TEXT,
    self_reflection TEXT,
    embedding BLOB
);

CREATE TABLE IF NOT EXISTS relationships (
    person_id TEXT PRIMARY KEY,
    name TEXT,
    first_met REAL,
    last_interaction REAL,
    interaction_count INTEGER,
    familiarity REAL,
    warmth REAL,
    trust REAL,
    dynamic TEXT,
    notes TEXT,
    topics_shared TEXT,
    unresolved TEXT
);

CREATE TABLE IF NOT EXISTS beliefs (
    id TEXT PRIMARY KEY,
    category TEXT,
    content TEXT,
    confidence REAL,
    source TEXT,
    formed_at REAL,
    last_reinforced REAL,
    times_reinforced INTEGER,
    embedding BLOB
);

CREATE TABLE IF NOT EXISTS imprints (
    id TEXT PRIMARY KEY,
    episode_id TEXT,
    emotion TEXT,
    intensity REAL,
    trigger TEXT,
    embedding BLOB
);

CREATE TABLE IF NOT EXISTS narrative (
    id TEXT PRIMARY KEY,
    timestamp REAL,
    self_story TEXT,
    trait_snapshot TEXT
);

CREATE TABLE IF NOT EXISTS inner_voice (
    id TEXT PRIMARY KEY,
    timestamp REAL,
    summary TEXT,
    emotional_context TEXT,
    embedding BLOB
);

CREATE TABLE IF NOT EXISTS entity_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS evolution_log (
    id TEXT PRIMARY KEY,
    timestamp REAL,
    changes_json TEXT,
    reasoning TEXT
);
"""


def _pack_embedding(vec: list[float]) -> bytes:
    arr = np.array(vec, dtype=np.float32)
    return arr.tobytes()


def _unpack_embedding(blob: bytes | None) -> Optional[list[float]]:
    if not blob:
        return None
    arr = np.frombuffer(blob, dtype=np.float32)
    return arr.tolist()


def cosine_sim(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    va = np.array(a, dtype=np.float64)
    vb = np.array(b, dtype=np.float64)
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


class MemoryStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = str(Path(db_path).expanduser())
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    async def connect(self) -> aiosqlite.Connection:
        conn = await aiosqlite.connect(self.db_path)
        await conn.executescript(SCHEMA)
        await conn.commit()
        return conn

    async def search_episodes_by_embedding(
        self,
        conn: aiosqlite.Connection,
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
            emb = _unpack_embedding(emb_blob)
            if emb:
                scored.append((eid, cosine_sim(query_emb, emb)))
        scored.sort(key=lambda x: -x[1])
        return scored[:limit]

    async def search_episodes_by_embedding_biased(
        self,
        conn: aiosqlite.Connection,
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
            ph = ",".join("?" * len(eids))
            icur = await conn.execute(
                f"SELECT episode_id, emotion, intensity FROM imprints WHERE episode_id IN ({ph})",
                eids,
            )
            for ep_id, emo, intens in await icur.fetchall():
                imprint_map.setdefault(ep_id, []).append((str(emo), float(intens)))

        scored: list[tuple[str, float]] = []
        for eid, ts, sig, emb_blob in rows:
            if sig is not None and float(sig) < min_significance:
                continue
            emb = _unpack_embedding(emb_blob)
            if not emb:
                continue
            base = cosine_sim(query_emb, emb)
            age = max(0.0, now - float(ts))
            mem_decay = (
                0.5 ** (age / imprint_half_life) if imprint_half_life > 0 else 1.0
            )
            imprint_bonus = 0.0
            for emo_str, intens in imprint_map.get(eid, []):
                aff = affinity_multiplier_for_recall(mood, emo_str)
                imprint_bonus += float(intens) * (aff - 1.0) * mem_decay
            imprint_bonus = min(1.5, imprint_bonus)
            adj = base * (1.0 + imprint_weight * imprint_bonus)
            scored.append((eid, adj))
        scored.sort(key=lambda x: -x[1])
        return scored[:limit]

    async def count_episodes(self, conn: aiosqlite.Connection) -> int:
        cur = await conn.execute("SELECT COUNT(*) FROM episodes")
        row = await cur.fetchone()
        return int(row[0]) if row else 0

    async def count_relationships(self, conn: aiosqlite.Connection) -> int:
        cur = await conn.execute("SELECT COUNT(*) FROM relationships")
        row = await cur.fetchone()
        return int(row[0]) if row else 0

    async def min_episode_timestamp(self, conn: aiosqlite.Connection) -> float | None:
        cur = await conn.execute("SELECT MIN(timestamp) FROM episodes")
        row = await cur.fetchone()
        if row and row[0] is not None:
            return float(row[0])
        return None
