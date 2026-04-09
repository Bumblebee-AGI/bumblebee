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

from bumblebee.cognition.poker_grounding import compose_grounded_poker_disposition
from bumblebee.cognition.poker_prompts import (
    load_poker_deck,
    resolve_deck_path,
    select_poker_prompt,
)
from bumblebee.config import AutonomySettings, EntityConfig
from bumblebee.identity.desire import infer_desires
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
        self._last_desires: list[dict[str, Any]] = []

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

        self._last_desires = infer_desires(
            tonic=tonic,
            drives=drives,
            max_items=max(1, int(self._auto.max_desires_considered or 3)),
        )

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

        if self._auto.desire_wake and self._last_desires:
            top = self._last_desires[0]
            urgency = float(top.get("urgency", 0.0) or 0.0)
            if urgency >= float(self._auto.desire_wake_threshold):
                kind = str(top.get("kind") or "act").strip() or "act"
                target = str(top.get("target") or "").strip()
                if target:
                    return f"desire:{kind}:{target[:40]}"
                return f"desire:{kind}"

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
        now = time.time()
        previous_wake_at = self._last_cycle_at
        self._last_cycle_at = now
        self._cycle_count_this_hour += 1
        self._next_timer_wake = self._schedule_next_timer()

        log.info(
            "autonomous_wake",
            module="presence",
            reason=reason,
            cycle_num=self._cycle_count_this_hour,
        )

        try:
            stirring, poker_disposition = await self._compose_stirring(
                entity, tonic, previous_wake_at=previous_wake_at
            )
            context = self._build_context(
                entity, tonic, stirring, reason, poker_disposition=poker_disposition
            )
            await self._run_perceive(entity, context)
        except Exception as e:
            log.exception("autonomous_cycle_failed", error=str(e))
        finally:
            self._running = False

    async def _compose_stirring(
        self, entity: Any, tonic: Any, *, previous_wake_at: float
    ) -> tuple[str, str | None]:
        """Compose wake stirring and optional poker disposition (internal prompt)."""
        pp = self.cfg.harness.autonomy.poker_prompts
        poker_disposition: str | None = None
        replace_voice = bool(pp.enabled) and str(pp.mode or "").strip().lower() == "replace_wake_voice"

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

        reflex_model = (
            entity.config.cognition.reflex_model
            or entity.config.cognition.deliberate_model
        )
        num_ctx = entity.config.effective_ollama_num_ctx()

        poker_seed: str | None = None
        if pp.enabled:
            path = resolve_deck_path(entity.config.name, pp.prompts_path)
            deck = load_poker_deck(path)
            picked = select_poker_prompt(
                deck,
                time_weighted=bool(pp.time_weighted),
            )
            if picked:
                poker_seed = picked.text
                log.info(
                    "poker_prompt_selected",
                    energy=picked.energy,
                    mode=pp.mode,
                    deck=str(path),
                )

        if poker_seed:
            if bool(pp.ground_with_gen):
                gm = (pp.grounding_model or "").strip() or reflex_model
                client = getattr(entity, "client", None)
                if gm and client:
                    poker_disposition = await compose_grounded_poker_disposition(
                        client,
                        gm,
                        tonic,
                        seed=poker_seed,
                        entity_name=entity.config.name,
                        journal_tail=journal_tail,
                        conversation_tail=conv_tail.strip(),
                        relationship_blurb=relationship_blurb,
                        temperature=float(pp.grounding_temperature),
                        max_tokens=int(pp.grounding_max_tokens),
                        num_ctx=num_ctx,
                    )
                    log.info("poker_prompt_grounded", used_gen=True, model=gm)
                else:
                    poker_disposition = poker_seed
            else:
                poker_disposition = poker_seed

        if replace_voice and poker_disposition:
            return poker_disposition, None

        last_mood = ""
        lc = getattr(entity, "_last_conversation", {}) or {}
        now = time.time()
        minutes_since_wake = (now - previous_wake_at) / 60.0 if previous_wake_at > 0 else 0.0
        minutes_since_conversation = (now - float(lc.get("at", 0))) / 60.0 if lc.get("at") else 0.0

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
            num_ctx=num_ctx,
        )

        if not stirring:
            stirring = "(the subconscious is quiet — nothing specific stirs)"

        if replace_voice and not poker_disposition:
            log.warning(
                "poker_prompt_replace_mode_no_prompt",
                deck=str(resolve_deck_path(entity.config.name, pp.prompts_path)),
            )

        if poker_disposition and not replace_voice:
            return stirring, poker_disposition

        return stirring, None

    def _build_context(
        self,
        entity: Any,
        tonic: Any,
        stirring: str,
        reason: str,
        *,
        poker_disposition: str | None = None,
    ) -> str:
        """Assemble the full autonomous context around optional poker + WakeVoice stirring."""
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
        ]
        if poker_disposition:
            pp = self.cfg.harness.autonomy.poker_prompts
            gen_note = ""
            if bool(pp.ground_with_gen):
                gen_note = (
                    " It may have been woven from a deck impulse plus your generative noise (GEN) and recent "
                    "experience — taste, not a script."
                )
            parts.extend(
                [
                    "[Internal disposition — private; not shown to users. Treat it as loose permission to exercise "
                    "agency: explore the real digital world with your tools when it resonates — find, do, research, "
                    "build, reach out, learn, try something new."
                    + gen_note
                    + " Turn it into a concrete step only when it fits; otherwise ignore or only use think().]",
                    poker_disposition,
                    "",
                ]
            )
        parts.extend(
            [
                "[Your subconscious stirring]",
                stirring,
                "",
                f"[World]",
            ]
        )
        parts.extend(
            [
                f"Platforms you're present on: {channels_str}",
                "",
                "[Top desire pressures]",
                self._render_desire_block(),
                "",
                "You just woke up. Your body woke you — read what stirs and decide what to do.",
            ]
        )

        if self._auto.allow_tool_calls_on_wake:
            parts.extend(
                [
                    "You may venture out with tools when it feels natural: search, fetch, inspect, "
                    "write notes, schedule, or message someone. Prefer purposeful action tied to your "
                    "active desire pressure, not generic busywork.",
                    "Use think() to reason privately. Use any relevant tools. Use say() only when you "
                    "actually want to communicate. Use end_turn() when done — include mood and parting thought.",
                    "Doing nothing is valid. Silence is valid. Observation alone is valid.",
                ]
            )
        else:
            parts.extend(
                [
                    "Wake mode is constrained right now: reason, observe, and optionally speak, but do not "
                    "venture into tool use.",
                    "Use think() to reason privately. Use say() to send a message. Use end_turn() when done.",
                ]
            )

        return "\n".join(parts)

    def _render_desire_block(self) -> str:
        if not self._last_desires:
            return "(no strong pressures)"
        lines: list[str] = []
        for d in self._last_desires[: max(1, int(self._auto.max_desires_considered or 3))]:
            kind = str(d.get("kind") or "act")
            urgency = float(d.get("urgency", 0.0) or 0.0)
            target = str(d.get("target") or "").strip()
            why = str(d.get("why") or "").strip()
            msg = f"- {kind} ({urgency:.2f})"
            if target:
                msg += f" -> {target[:80]}"
            if why:
                msg += f" [{why[:120]}]"
            lines.append(msg)
        return "\n".join(lines) if lines else "(no strong pressures)"

    async def _run_perceive(self, entity: Any, context: str) -> None:
        """Run entity.perceive() with the autonomous context."""
        channel, reply_platform = self._resolve_wake_delivery(entity)
        inp = Input(
            text=context,
            person_id="self",
            person_name="Self",
            channel=channel,
            platform="autonomous",
        )

        tool_names: list[str] = []

        try:
            reply_text, _ = await entity.perceive(
                inp,
                stream=None,
                record_user_message=False,
                meaningful_override=False,
                reply_platform=reply_platform,
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

    @staticmethod
    def _resolve_wake_delivery(entity: Any) -> tuple[str, Any | None]:
        """
        Pick where autonomous `say()` should deliver:
        1) last active conversation route if still available,
        2) first active non-CLI platform with its last known channel,
        3) fallback to autonomous/no platform (messages cannot be sent).
        """
        platforms = getattr(entity, "_platforms", {}) or {}
        last_conv = getattr(entity, "_last_conversation", {}) or {}

        last_name = str(last_conv.get("platform") or "").strip().lower()
        last_channel = str(last_conv.get("channel") or "").strip()
        if last_name and last_channel:
            last_pf = platforms.get(last_name)
            if last_pf is not None:
                return last_channel, last_pf

        for name, pf in platforms.items():
            n = str(name or "").strip().lower()
            if n == "cli":
                continue
            ch = ""
            ch_attr = getattr(pf, "last_chat_id", None)
            if ch_attr:
                ch = str(ch_attr).strip()
            if not ch:
                ch_attr = getattr(pf, "last_channel_id", None)
                if ch_attr:
                    ch = str(ch_attr).strip()
            if ch:
                return ch, pf

        return "autonomous", None
