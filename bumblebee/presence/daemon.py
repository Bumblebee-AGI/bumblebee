"""Always-on scheduling: heartbeat, drives, initiative, MCP refresh, consolidation."""

from __future__ import annotations

import time

import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from bumblebee.config import EntityConfig
from bumblebee.memory.consolidation import ConsolidationJob
from bumblebee.presence.initiative import InitiativeEngine

log = structlog.get_logger("bumblebee.presence.daemon")


class PresenceDaemon:
    def __init__(self, entity_facade) -> None:
        self.entity = entity_facade
        self.cfg: EntityConfig = entity_facade.config
        self._scheduler: AsyncIOScheduler | None = None
        self._consolidation = ConsolidationJob(self.cfg)
        self._initiative = InitiativeEngine(self.cfg, entity_facade.client)
        self._last_consolidation: float = time.time()

    def _daemon_dict(self) -> dict:
        return dict(self.cfg.presence.daemon or {})

    def _heartbeat_seconds(self) -> int:
        d = self._daemon_dict()
        return int(d.get("heartbeat_interval", self.cfg.harness.presence.heartbeat_interval))

    def _consolidation_seconds(self) -> int:
        d = self._daemon_dict()
        return int(d.get("memory_consolidation", self.cfg.harness.memory.consolidation_interval))

    async def start(self) -> None:
        if self._scheduler and self._scheduler.running:
            return
        self._scheduler = AsyncIOScheduler()
        hb = max(5, self._heartbeat_seconds())
        cons = max(30, self._consolidation_seconds())
        self._scheduler.add_job(
            self._heartbeat_tick,
            "interval",
            seconds=hb,
            id="bumblebee_heartbeat",
            replace_existing=True,
        )
        self._scheduler.add_job(
            self._consolidation_tick,
            "interval",
            seconds=cons,
            id="bumblebee_consolidation",
            replace_existing=True,
        )
        self._scheduler.start()
        log.info(
            "daemon_scheduler_started",
            module="presence",
            platform="daemon",
            heartbeat_seconds=hb,
            consolidation_seconds=cons,
        )

    async def stop(self) -> None:
        if self._scheduler:
            try:
                self._scheduler.shutdown(wait=False)
            except Exception as e:
                log.warning("scheduler_shutdown", module="presence", error=str(e))
            self._scheduler = None

    async def _heartbeat_tick(self) -> None:
        structlog.contextvars.bind_contextvars(
            entity_name=self.cfg.name,
            module="presence",
            platform="daemon",
        )
        try:
            await self.entity.tick()
            now = time.time()
            silence = now - getattr(self.entity, "_last_user_message_at", now)
            drives = self.entity.drives.tick(silence_seconds=silence)
            cooldown = float(self.cfg.drives.initiative_cooldown)
            if drives and self.entity.drives.can_initiate(now, cooldown):
                d = drives[0]
                lc = getattr(self.entity, "_last_conversation", {}) or {}
                ctx = (
                    f"Drive: {d.name}. Topics: {self.cfg.drives.curiosity_topics}. "
                    f"Last conversation platform={lc.get('platform')} channel={lc.get('channel')} "
                    f"person_id={lc.get('person_id')}."
                )
                msg = await self._initiative.compose_proactive(d, ctx)
                if msg:
                    self.entity.drives.register_initiative_time(now)
                    self.entity.drives.satisfy(d.name, 0.5)
                    await self.entity.broadcast_proactive(msg)
                    log.info(
                        "initiative_sent",
                        module="presence",
                        platform="daemon",
                        drive=d.name,
                    )
            try:
                await self.entity.refresh_mcp_servers()
            except Exception as e:
                log.warning("mcp_heartbeat_refresh_failed", module="presence", error=str(e))
        except Exception as e:
            log.exception("daemon_heartbeat_error", module="presence", error=str(e))

    async def _consolidation_tick(self) -> None:
        structlog.contextvars.bind_contextvars(
            entity_name=self.cfg.name,
            module="presence",
            platform="daemon",
        )
        try:
            await self._consolidation.run_for_daemon(self.entity)
            self._last_consolidation = time.time()
        except Exception as e:
            log.exception("daemon_consolidation_error", module="presence", error=str(e))
