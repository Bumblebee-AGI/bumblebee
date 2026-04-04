"""Always-on heartbeat: emotions, drives, consolidation, initiative."""

from __future__ import annotations

import asyncio
import time

import structlog

from bumblebee.config import EntityConfig
from bumblebee.memory.consolidation import ConsolidationJob
from bumblebee.presence.initiative import InitiativeEngine

log = structlog.get_logger("bumblebee.presence.daemon")


class PresenceDaemon:
    def __init__(self, entity_facade) -> None:
        self.entity = entity_facade
        self.cfg: EntityConfig = entity_facade.config
        self._task: asyncio.Task | None = None
        self._running = False
        # Defer first consolidation so we don't stack a second aiosqlite open on startup.
        self._last_consolidation: float = time.time()
        self._consolidation = ConsolidationJob(self.cfg)
        self._initiative = InitiativeEngine(self.cfg, entity_facade.client)

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _loop(self) -> None:
        interval = self.cfg.harness.presence.heartbeat_interval
        while self._running:
            try:
                await self.entity.tick()
                now = time.time()
                silence = now - getattr(self.entity, "_last_user_message_at", now)
                drives = self.entity.drives.tick(silence_seconds=silence)
                cooldown = float(self.cfg.drives.initiative_cooldown)
                if drives and self.entity.drives.can_initiate(now, cooldown):
                    d = drives[0]
                    ctx = f"Drive: {d.name}. Topics: {self.cfg.drives.curiosity_topics}"
                    msg = await self._initiative.compose_proactive(d, ctx)
                    if msg:
                        self.entity.drives.register_initiative_time(now)
                        self.entity.drives.satisfy(d.name, 0.5)
                        await self.entity.broadcast_proactive(msg)
                        log.info(
                            "initiative_sent",
                            module="presence",
                            drive=d.name,
                        )
                cons_every = float(self.cfg.harness.memory.consolidation_interval)
                if now - self._last_consolidation >= cons_every:
                    await self._consolidation.run_for_daemon(self.entity)
                    self._last_consolidation = now
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.exception("daemon_tick_error", module="presence", error=str(e))
            await asyncio.sleep(interval)
