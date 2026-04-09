"""Autonomous wake-cycle engine.

Monitors soma state and drives, decides when the entity should wake
autonomously, asks the subconscious WakeVoice to compose a stirring,
then runs a full perceive cycle where the conscious agent can think,
observe, act, or go back to sleep.
"""

from __future__ import annotations

import random
import time
from typing import Any

import structlog

from bumblebee.config import AutonomySettings, EntityConfig
from bumblebee.identity.drives import Drive
from bumblebee.models import Input

log = structlog.get_logger("bumblebee.presence.wake_cycle")


class WakeCycleEngine:
    """Evaluates wake conditions and runs autonomous perceive cycles."""

    def __init__(self, cfg: EntityConfig) -> None:
        self.cfg = cfg
        self._auto: AutonomySettings = cfg.harness.autonomy
        self._last_cycle_at: float = time.time()
        self._cycle_count_this_hour: int = 0
        self._hour_start: float = time.time()
        self._running: bool = False
        self._next_timer_wake: float = self._schedule_next_timer()

    def _schedule_next_timer(self) -> float:
        lo = max(1, self._auto.base_wake_interval_min) * 60.0
        hi = max(lo + 60, self._auto.base_wake_interval_max * 60.0)
        return time.time() + random.uniform(lo, hi)

    def _reset_hourly_counter(self) -> None:
        now = time.time()
        if now - self._hour_start >= 3600:
            self._cycle_count_this_hour = 0
            self._hour_start = now

    def should_wake(
        self,
        *,
        tonic: Any,
        drives: list[Drive],
        silence_seconds: float,
        dormant: bool = False,
    ) -> str | None:
        """Return a wake reason string, or None if no wake condition is met."""
        if not self._auto.enabled or dormant or self._running:
            return None

        now = time.time()
        if now - self._last_cycle_at < self._auto.min_cycle_gap_seconds:
            return None

        if silence_seconds < self._auto.silence_threshold_seconds:
            return None

        self._reset_hourly_counter()
        if self._cycle_count_this_hour >= self._auto.max_cycles_per_hour:
            return None

        if self._auto.impulse_wake:
            active = [
                i for i in tonic.bars._active_impulses
                if not i.get("on_cooldown")
            ]
            if active:
                labels = ", ".join(i.get("label", "?") for i in active)
                return f"impulse:{labels}"

        if self._auto.drive_wake:
            crossed = [d for d in drives if d.level >= d.threshold]
            if crossed:
                names = ", ".join(d.name for d in crossed)
                return f"drive:{names}"

        if self._auto.conflict_wake:
            intense = [
                c for c in tonic.bars._active_conflicts
                if c.get("intensity", 0) > 0.4
            ]
            if intense:
                labels = ", ".join(c.get("label", "?") for c in intense)
                return f"conflict:{labels}"

        if self._auto.noise_wake:
            for frag in tonic.noise.current_fragments()[-3:]:
                fl = frag.lower()
                if "?" in fl or "should" in fl or "wonder" in fl or "why" in fl:
                    return f"noise:{frag[:60]}"

        if now >= self._next_timer_wake:
            self._next_timer_wake = self._schedule_next_timer()
            return "timer"

        return None

    async def run(
        self,
        *,
        entity: Any,
        tonic: Any,
        reason: str,
    ) -> None:
        """Run a full autonomous wake cycle."""
        if self._running:
            return
        self._running = True
        self._last_cycle_at = time.time()
        self._cycle_count_this_hour += 1
        self._next_timer_wake = self._schedule_next_timer()

        log.info(
            "autonomous_wake",
            module="presence",
            reason=reason,
            cycle_num=self._cycle_count_this_hour,
        )

        try:
            stirring = await self._compose_stirring(entity, tonic)
            context = self._build_context(entity, tonic, stirring, reason)
            await self._run_perceive(entity, context)
        except Exception as e:
            log.exception("autonomous_cycle_failed", error=str(e))
        finally:
            self._running = False

    async def _compose_stirring(self, entity: Any, tonic: Any) -> str:
        """Ask the subconscious WakeVoice to compose the wake prompt."""
        journal_tail = ""
        if hasattr(entity, "journal") and entity.journal.path.is_file():
            try:
                raw = entity.journal.path.read_text(encoding="utf-8", errors="replace")
                journal_tail = raw[-500:] if len(raw) > 500 else raw
            except Exception:
                pass

        conv_tail = ""
        hist = getattr(entity, "_history", [])
        for m in hist[-3:]:
            role = m.get("role", "")
            content = str(m.get("content", ""))[:200]
            conv_tail += f"{role}: {content}\n"

        relationship_blurb = ""
        try:
            async with entity.store.session() as db:
                rels = await entity.relational.list_recent(db, 5)
                if rels:
                    blurbs = []
                    for r in rels:
                        blurbs.append(
                            f"{r.name} (warmth={r.warmth:.1f}, trust={r.trust:.1f})"
                        )
                    relationship_blurb = ", ".join(blurbs)
        except Exception:
            pass

        last_mood = ""
        lc = getattr(entity, "_last_conversation", {}) or {}
        last_turn_at = getattr(entity, "_last_turn_completed_at", None)
        now = time.time()
        minutes_since_wake = (now - self._last_cycle_at) / 60.0 if self._last_cycle_at > 0 else 0.0
        minutes_since_conversation = (now - float(lc.get("at", 0))) / 60.0 if lc.get("at") else 0.0

        reflex_model = (
            entity.config.cognition.reflex_model
            or entity.config.cognition.deliberate_model
        )

        stirring = await tonic.compose_wake_stirring(
            entity.client,
            reflex_model,
            entity_name=entity.config.name,
            journal_tail=journal_tail,
            conversation_tail=conv_tail.strip(),
            relationship_blurb=relationship_blurb,
            last_mood=last_mood,
            minutes_since_wake=minutes_since_wake,
            minutes_since_conversation=minutes_since_conversation,
            num_ctx=entity.config.effective_ollama_num_ctx(),
        )

        if not stirring:
            stirring = "(the subconscious is quiet — nothing specific stirs)"

        return stirring

    def _build_context(
        self,
        entity: Any,
        tonic: Any,
        stirring: str,
        reason: str,
    ) -> str:
        """Assemble the full autonomous context around the WakeVoice stirring."""
        import datetime
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

        platform_channels: list[str] = []
        platforms = getattr(entity, "_platforms", {})
        for name in platforms:
            platform_channels.append(name)

        channels_str = ", ".join(platform_channels) if platform_channels else "(none active)"

        parts = [
            f"[AUTONOMOUS WAKE — {now_str}, reason: {reason}]",
            "",
            "[Your subconscious stirring]",
            stirring,
            "",
            f"[World]",
            f"Platforms you're present on: {channels_str}",
            "",
            "You just woke up. Your body woke you — read what stirs and decide what "
            "to do. Use think() to reason privately. Use say() to send a message. "
            "Use end_turn() when you're done — record your mood and a parting thought. "
            "Doing nothing is valid. Silence is valid. Observation alone is valid.",
        ]

        return "\n".join(parts)

    async def _run_perceive(self, entity: Any, context: str) -> None:
        """Run entity.perceive() with the autonomous context."""
        inp = Input(
            text=context,
            person_id="self",
            person_name="Self",
            channel="autonomous",
            platform="autonomous",
        )

        tool_names: list[str] = []

        try:
            reply_text, _ = await entity.perceive(
                inp,
                stream=None,
                record_user_message=False,
                meaningful_override=False,
                reply_platform=None,
                preserve_conversation_route=True,
                routine_history=False,
                skip_relational_upsert=True,
                skip_drive_interaction=True,
                skip_episode=False,
                tool_names_log=tool_names,
            )
        except Exception as e:
            log.exception("autonomous_perceive_failed", error=str(e))
            return

        _OUTBOUND_TOOLS = {"say", "send_dm", "send_message_to"}
        sent_messages = any(n in _OUTBOUND_TOOLS for n in tool_names)
        used_end_turn = "end_turn" in tool_names

        tonic = getattr(entity, "tonic", None)
        if tonic is not None and not sent_messages:
            tonic.emit({"type": "idle_cycle"})

        log.info(
            "autonomous_cycle_complete",
            module="presence",
            tools_used=tool_names,
            sent_messages=sent_messages,
            used_end_turn=used_end_turn,
            reply_len=len(reply_text or ""),
        )
