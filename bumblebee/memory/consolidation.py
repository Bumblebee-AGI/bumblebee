"""Background decay, strengthening, narrative triggers."""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import structlog

log = structlog.get_logger("bumblebee.memory.consolidation")


def apply_episode_decay_sync(db_path: str, delta: float) -> None:
    """Sync SQLite decay — avoids extra aiosqlite worker alongside live perceive/tick."""
    p = Path(db_path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(p)) as conn:
        conn.execute(
            """
            UPDATE episodes SET significance = CASE
                WHEN significance - ? < 0 THEN 0
                ELSE significance - ?
            END
            WHERE significance > 0
            """,
            (delta, delta),
        )
        conn.commit()


class ConsolidationJob:
    def __init__(self, entity_config) -> None:
        self._cfg = entity_config
        self._run_count = 0

    async def run(self, conn, entity_facade=None) -> None:
        rate = self._cfg.harness.memory.memory_decay_rate
        delta = rate * 10
        await conn.execute(
            """
            UPDATE episodes SET significance = CASE
                WHEN significance - ? < 0 THEN 0
                ELSE significance - ?
            END
            WHERE significance > 0
            """,
            (delta, delta),
        )
        await conn.commit()
        log.info("consolidation_tick", module="memory", decay_applied=rate)

        self._run_count += 1
        n_every = max(1, self._cfg.harness.memory.narrative_every_n_consolidations)
        if entity_facade and self._run_count % n_every == 0:
            try:
                await entity_facade.run_narrative_cycle(conn)
            except Exception as e:
                log.warning("narrative_cycle_failed", module="memory", error=str(e))

    async def run_for_daemon(self, entity_facade) -> None:
        """Decay via thread + optional narrative on a short-lived aiosqlite connection."""
        rate = self._cfg.harness.memory.memory_decay_rate
        delta = rate * 10
        await asyncio.to_thread(apply_episode_decay_sync, entity_facade.store.db_path, delta)
        log.info("consolidation_tick", module="memory", decay_applied=rate)

        self._run_count += 1
        n_every = max(1, self._cfg.harness.memory.narrative_every_n_consolidations)
        if not entity_facade or self._run_count % n_every != 0:
            return
        db = await entity_facade.store.connect()
        try:
            try:
                await entity_facade.run_narrative_cycle(db)
            except Exception as e:
                log.warning("narrative_cycle_failed", module="memory", error=str(e))
        finally:
            await db.close()
