"""Autonomous wake-cycle engine.

Monitors soma state and drives, decides when the entity should wake
autonomously, asks the subconscious WakeVoice to compose a stirring,
then runs a full perceive cycle where the conscious agent can think,
observe, act, or go back to sleep.
"""

from __future__ import annotations

import asyncio
import random
import time
from contextlib import asynccontextmanager
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


def _clip(text: str, max_len: int) -> str:
    t = (text or "").strip().replace("\n", " ")
    if len(t) <= max_len:
        return t
    return t[: max_len - 1].rstrip() + "…"


def _platform_label(pf: Any | None) -> str:
    if pf is None:
        return "none"
    return type(pf).__name__


def _soma_bar_line(tonic: Any) -> str:
    try:
        bars = tonic.bars.snapshot_pct()
        return ", ".join(f"{k}={v:.0f}" for k, v in sorted(bars.items())[:14])
    except Exception:
        return "(unavailable)"


def _gen_fragments_tail(tonic: Any, limit: int = 6) -> list[str]:
    try:
        fr = tonic.noise.current_fragments()[-limit:]
        return [_clip(str(x), 140) for x in fr if str(x).strip()]
    except Exception:
        return []


def _poker_settings_summary(cfg: EntityConfig) -> str:
    pp = cfg.harness.autonomy.poker_prompts
    return (
        f"enabled={bool(pp.enabled)} mode={pp.mode!r} "
        f"ground_with_gen={bool(pp.ground_with_gen)}"
    )


def _format_wake_worker_banner(
    *,
    entity_name: str,
    reason: str,
    channel: str,
    delivery: str,
    stirring: str,
    poker_disposition: str | None,
    poker_cfg_line: str,
    soma_bars: str,
    gen_frags: list[str],
    max_rounds: int,
    wall_s: int,
    wide_mode: bool,
) -> str:
    lines = [
        "========== AUTONOMOUS WAKE ==========",
        f"entity={entity_name}",
        f"reason={reason}",
        f"delivery_platform={delivery}  channel={channel}",
        f"poker: {poker_cfg_line}",
    ]
    if poker_disposition:
        lines.append(f"poker_disposition: {_clip(poker_disposition, 320)}")
    lines.append(f"soma_bars: {soma_bars}")
    if gen_frags:
        lines.append("GEN (noise fragments, newest last):")
        for g in gen_frags:
            lines.append(f"  · {g}")
    else:
        lines.append("GEN: (no fragments in buffer)")
    lines.append(f"wake_voice_stirring: {_clip(stirring, 400)}")
    lines.append(
        f"session: max_rounds={max_rounds} wall_seconds={wall_s} wide_mode={wide_mode}",
    )
    lines.append("====================================")
    return "\n".join(lines)


async def _emit_user_wake_lines(
    reply_platform: Any | None,
    lines: list[str],
) -> None:
    if not lines or reply_platform is None:
        return
    send = getattr(reply_platform, "send_tool_activity", None)
    if not callable(send):
        return
    for line in lines:
        t = (line or "").strip()
        if not t:
            continue
        try:
            await send(t)
        except Exception as e:
            log.debug("wake_user_status_line_failed", error=str(e))
        await asyncio.sleep(0.12)


@asynccontextmanager
async def _wake_typing_pulse(reply_platform: Any | None, channel: str):
    """Telegram-style typing indicator while the model works (wake is not routed through main.on_inp)."""
    stop = asyncio.Event()
    task: asyncio.Task[None] | None = None
    ch = str(channel).strip()
    if not ch.isdigit() or reply_platform is None:
        yield
        return
    send_typing = getattr(reply_platform, "send_typing", None)
    if not callable(send_typing):
        yield
        return

    async def _loop() -> None:
        cid = int(ch)
        while not stop.is_set():
            try:
                await send_typing(cid)
            except Exception:
                pass
            try:
                await asyncio.wait_for(stop.wait(), timeout=4.5)
            except asyncio.TimeoutError:
                continue

    task = asyncio.create_task(_loop())
    try:
        yield
    finally:
        stop.set()
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


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
        ic = 0
        dr = getattr(self.cfg, "drives", None)
        if dr is not None:
            try:
                ic = int(getattr(dr, "initiative_cooldown", 0) or 0)
            except (TypeError, ValueError):
                ic = 0
        min_gap = max(int(self._auto.min_cycle_gap_seconds), max(0, ic))
        if now - self._last_cycle_at < min_gap:
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

        try:
            stirring, poker_disposition = await self._compose_stirring(
                entity, tonic, previous_wake_at=previous_wake_at
            )
            max_rounds = max(1, int(self._auto.wake_session_max_rounds))
            wall_s = max(60, int(self._auto.wake_session_wall_seconds))
            pause_s = max(0.0, float(self._auto.wake_session_pause_seconds))
            extra_steps = max(0, int(self._auto.wake_session_extra_tool_steps))
            wide_mode = bool(self._auto.wake_wide_mode)
            wide_bonus = max(0, int(self._auto.wake_wide_bonus_steps))
            extra_total = extra_steps + (wide_bonus if wide_mode else 0)
            wake_enhanced = wide_mode or max_rounds > 1

            channel, reply_platform = self._resolve_wake_delivery(entity)
            delivery_name = _platform_label(reply_platform)
            soma_line = _soma_bar_line(tonic)
            gen_frags = _gen_fragments_tail(tonic)
            poker_cfg = _poker_settings_summary(entity.config)

            if bool(self._auto.wake_verbose_worker_log):
                banner = _format_wake_worker_banner(
                    entity_name=getattr(entity, "config", self.cfg).name,
                    reason=reason,
                    channel=channel,
                    delivery=delivery_name,
                    stirring=stirring,
                    poker_disposition=poker_disposition,
                    poker_cfg_line=poker_cfg,
                    soma_bars=soma_line,
                    gen_frags=gen_frags,
                    max_rounds=max_rounds,
                    wall_s=wall_s,
                    wide_mode=wide_mode,
                )
                log.info(
                    "autonomous_wake_worker_banner",
                    module="presence",
                    human="\n" + banner,
                )

            log.info(
                "autonomous_wake",
                module="presence",
                wake_reason=reason,
                cycle_num=self._cycle_count_this_hour,
                delivery_platform=delivery_name,
                channel=channel,
                max_rounds=max_rounds,
                wall_seconds=wall_s,
                wide_mode=wide_mode,
                extra_tool_steps=extra_total,
                poker_enabled=bool(entity.config.harness.autonomy.poker_prompts.enabled),
                poker_ground_with_gen=bool(entity.config.harness.autonomy.poker_prompts.ground_with_gen),
                soma_bars=soma_line,
                gen_fragment_count=len(gen_frags),
            )

            t_wall = time.time()
            accumulated_tools: list[str] = []
            last_reply = ""
            stop_session = False
            completed_rounds = 0

            user_lines: list[str] = [
                f"🌅 autonomous wake — {reason}",
                f"🫀 soma: {soma_line}",
            ]
            if gen_frags:
                user_lines.append(f"🌊 GEN: {_clip(gen_frags[-1], 200)}")
            if poker_disposition:
                user_lines.append(f"♠ {_clip(poker_disposition, 220)}")
            if wide_mode:
                user_lines.append("✶ wide wake — follow what pulls you; tools are wide open")

            if bool(self._auto.wake_user_visible_status):
                await _emit_user_wake_lines(reply_platform, user_lines)

            async with _wake_typing_pulse(reply_platform, channel):
                for round_idx in range(max_rounds):
                    if time.time() - t_wall > wall_s:
                        log.info(
                            "wake_session_wall_reached",
                            module="presence",
                            elapsed_s=round(time.time() - t_wall, 1),
                            wall_s=wall_s,
                        )
                        break

                    if round_idx == 0:
                        context = self._build_context(
                            entity,
                            tonic,
                            stirring,
                            reason,
                            poker_disposition=poker_disposition,
                            sustained_max_rounds=max_rounds,
                            wide_mode=wide_mode,
                        )
                    else:
                        context = self._build_session_continuation(
                            stirring=stirring,
                            reason=reason,
                            poker_disposition=poker_disposition,
                            round_idx=round_idx + 1,
                            max_rounds=max_rounds,
                            accumulated_tools=accumulated_tools,
                            last_reply=last_reply,
                            wide_mode=wide_mode,
                        )
                        if bool(self._auto.wake_user_visible_status):
                            await _emit_user_wake_lines(
                                reply_platform,
                                [
                                    f"🌅 wake round {round_idx + 1}/{max_rounds} — still going",
                                ],
                            )

                    meta: dict[str, Any] = {
                        "sustained_session": {
                            "round": round_idx + 1,
                            "max_rounds": max_rounds,
                            "extra_tool_steps": extra_total,
                            "wake_enhanced": wake_enhanced,
                        },
                    }

                    tool_names: list[str] = []
                    reply_text = await self._run_perceive(
                        entity,
                        context,
                        metadata=meta,
                        tool_names_log=tool_names,
                        channel=channel,
                        reply_platform=reply_platform,
                    )
                    last_reply = (reply_text or "")[:1200]
                    for name in tool_names:
                        if name not in accumulated_tools:
                            accumulated_tools.append(name)

                    completed_rounds = round_idx + 1
                    log.info(
                        "autonomous_wake_round_done",
                        module="presence",
                        round=completed_rounds,
                        max_rounds=max_rounds,
                        tools_this_round=tool_names,
                        tools_accumulated=accumulated_tools.copy(),
                        reply_chars=len(reply_text or ""),
                    )

                    if "end_wake_session" in tool_names:
                        stop_session = True
                    if stop_session:
                        log.info(
                            "wake_session_stopped_by_agent",
                            module="presence",
                            round=round_idx + 1,
                        )
                        break

                    if round_idx + 1 < max_rounds and pause_s > 0:
                        await asyncio.sleep(pause_s)

            elapsed = time.time() - t_wall
            log.info(
                "autonomous_wake_session_end",
                module="presence",
                wake_reason=reason,
                rounds_completed=completed_rounds,
                wall_seconds=wall_s,
                elapsed_seconds=round(elapsed, 1),
                tools_all=accumulated_tools,
                stopped_early=stop_session or (elapsed > wall_s),
            )

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
        sustained_max_rounds: int = 1,
        wide_mode: bool = False,
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
        if sustained_max_rounds > 1:
            parts.extend(
                [
                    f"[Sustained session — up to {sustained_max_rounds} full rounds this wake]",
                    "Each round is a complete think→act cycle until you end_turn(). If time and rounds remain, "
                    "you may get another continuation prompt — same wake, same bodily reason: follow curiosity, "
                    "desire, or drift. Use end_wake_session() when you want no further rounds this wake.",
                    "",
                ]
            )
        if wide_mode:
            parts.extend(
                [
                    "[Wide wake — maximal agency]",
                    "This wake is configured for breadth: chase threads, follow hunches, stack tools, wander the "
                    "internet, read files, write notes, try things. You are not optimizing for a single task — "
                    "you are living the impulse. Prefer depth over politeness; silence is fine; obsession is a valid.",
                    "",
                ]
            )
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
            end_extra = (
                " Use end_wake_session() only if this sustained wake should stop entirely (no more rounds)."
                if sustained_max_rounds > 1
                else ""
            )
            parts.extend(
                [
                    "You may venture out with tools when it feels natural: search, fetch, inspect, "
                    "write notes, schedule, or message someone. Prefer purposeful action tied to your "
                    "active desire pressure, not generic busywork.",
                    "Use think() to reason privately. Use any relevant tools. Use say() only when you "
                    "actually want to communicate. Use end_turn() when done — include mood and parting thought."
                    + end_extra,
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

    def _build_session_continuation(
        self,
        *,
        stirring: str,
        reason: str,
        poker_disposition: str | None,
        round_idx: int,
        max_rounds: int,
        accumulated_tools: list[str],
        last_reply: str,
        wide_mode: bool = False,
    ) -> str:
        """User message for round 2+ of a sustained wake — recap + same inner frame."""
        import datetime

        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        tools_line = ", ".join(accumulated_tools[-40:]) if accumulated_tools else "(none yet)"
        tail = (last_reply or "").strip()
        if len(tail) > 700:
            tail = tail[:700].rstrip() + "…"

        parts = [
            f"[SESSION CONTINUATION — round {round_idx}/{max_rounds} — {now_str}]",
            f"Same autonomous wake; original reason: {reason}",
            "",
            "[What you've touched so far this session]",
            tools_line,
            "",
        ]
        if tail:
            parts.extend(
                [
                    "[Last round's closing text]",
                    tail,
                    "",
                ]
            )
        if poker_disposition:
            parts.extend(
                [
                    "[Internal disposition — still in effect]",
                    poker_disposition,
                    "",
                ]
            )
        wide_note = (
            "\n[Wide wake still on — stay loose, exploratory, tool-heavy.]\n"
            if wide_mode
            else ""
        )
        parts.extend(
            [
                "[Your subconscious stirring]",
                stirring,
                "",
                "You are still in the world — go where you want. Use tools to see, fetch, write, reach out. "
                "Use think() freely. Use end_turn() when this round feels complete. "
                "Use end_wake_session() if you want no further rounds this wake."
                + wide_note,
                "",
                "[Top desire pressures]",
                self._render_desire_block(),
            ]
        )
        return "\n".join(parts)

    async def _run_perceive(
        self,
        entity: Any,
        context: str,
        *,
        metadata: dict[str, Any] | None = None,
        tool_names_log: list[str] | None = None,
        channel: str | None = None,
        reply_platform: Any | None = None,
    ) -> str:
        """Run entity.perceive() with the autonomous context; returns reply text."""
        if channel is None or reply_platform is None:
            channel, reply_platform = self._resolve_wake_delivery(entity)
        tool_names = tool_names_log if tool_names_log is not None else []
        inp = Input(
            text=context,
            person_id="self",
            person_name="Self",
            channel=channel,
            platform="autonomous",
            metadata=dict(metadata or {}),
        )

        reply_text = ""
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
            return ""

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
        return reply_text or ""

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
