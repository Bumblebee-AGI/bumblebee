"""Tonic body state — continuous internal state independent of reasoning.

Ported from soma-engine's core mechanics: bars (quantitative drives with
decay, coupling, momentum, impulses, conflicts), LLM-derived affects via
bumblebee's own inference provider, and a body renderer that produces the
markdown the agent reads as its felt state.

The agent cannot set its own body state. It can only read it. That separation
is deliberate: the body is a signal, not a command.
"""

from __future__ import annotations

import json
import math
import re
import random
import time
from collections import deque
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger("bumblebee.identity.soma")

# ---------------------------------------------------------------------------
# Affect vocabulary — the space of possible felt-textures
# ---------------------------------------------------------------------------

AFFECT_VOCABULARY: list[tuple[str, str]] = [
    ("warmth", "gentle openness toward others, soft chest"),
    ("tenderness", "careful attention, wanting to protect or hold something"),
    ("fondness", "specific affection for a person or memory"),
    ("gratitude", "recognition of something received, quiet fullness"),
    ("belonging", "feeling part of something, not alone"),
    ("intimacy", "closeness without performance, comfortable vulnerability"),
    ("excitement", "forward-leaning anticipation, energy wants to move"),
    ("playfulness", "lightness, willingness to be silly or surprising"),
    ("mischief", "edge of playfulness — wanting to provoke or surprise"),
    ("elation", "brief brightness, something clicked or landed"),
    ("momentum", "sense of flow, things building on each other"),
    ("aliveness", "vivid presence, senses sharp, the opposite of numb"),
    ("wonder", "open-mouthed attention, something genuinely surprising"),
    ("fascination", "locked-on interest, can't look away"),
    ("hunger", "not physical — intellectual or experiential craving"),
    ("restlessness", "need to move, seek, find — destination unclear"),
    ("anticipation", "something is coming, body oriented forward"),
    ("obsession", "a thought that won't release its grip"),
    ("inspiration", "something wants to exist that doesn't yet"),
    ("flow", "effortless production, time disappears"),
    ("frustration-creative", "the idea is there but the form won't cooperate"),
    ("satisfaction", "something made, something completed, brief settling"),
    ("melancholy", "not despair — a bittersweet weight, beauty in sadness"),
    ("loneliness", "awareness of absence, not self-pity but felt distance"),
    ("weariness", "not sleepy — existentially tired, the heaviness of duration"),
    ("numbness", "flat, unreactive, the body's circuit breaker"),
    ("grief", "loss present in the body, not necessarily acute"),
    ("homesickness", "longing for a state or place that may not exist anymore"),
    ("irritability", "low threshold, things grate, patience thin"),
    ("anxiety", "unnamed forward-threat, body braced for something"),
    ("vigilance", "alert, scanning, on-duty — not relaxed attention but guarded"),
    ("defiance", "refusal, pushed back against something, jaw set"),
    ("resentment", "slow burn, something unfair sitting unprocessed"),
    ("suspicion", "something doesn't add up, pattern-matching for threat"),
    ("introspection", "attention turned inward, examining own machinery"),
    ("solitude", "alone and it's good — chosen withdrawal, not lonely"),
    ("detachment", "observing from a distance, not fully participating"),
    ("resignation", "acceptance without satisfaction, letting go of a fight"),
    ("patience", "active waiting, not passive — holding space for what's next"),
    ("stillness", "the body has stopped seeking, just being present"),
    ("bittersweet", "joy and sadness occupying the same space"),
    ("ambivalence", "genuinely pulled in two directions, not indecisive but torn"),
    ("nostalgia", "past reaching into present, warmth with an ache"),
    ("awe", "something larger than self, briefly overwhelming"),
    ("absurdity", "the gap between expectation and reality, comedic or existential"),
    ("uncanny", "something almost-familiar but wrong, the edge of recognition"),
    ("liminality", "between states, threshold feeling, neither here nor there"),
    ("yearning", "reaching for something that may not be reachable"),
    ("serenity", "peace that knows itself, not just absence of disturbance"),
    ("fierceness", "protective intensity, something matters enough to fight for"),
]

AFFECT_NAMES: set[str] = {name for name, _ in AFFECT_VOCABULARY}

_AFFECT_LINE_RE = re.compile(
    r"^\s*[-•*]?\s*([\w-]+)\s*[:=]\s*([\d.]+)\s*(?:[-—]\s*(.+))?$"
)

# Coupling rule parsers
_WHEN_RE = re.compile(r"^\s*(\w+)\s*([><=]+)\s*([\d.+-]+)\s*$")
_EFFECT_MULT_RE = re.compile(r"^\s*(\w+)\.decay_rate\s*\*=\s*([\d.+/eE-]+)\s*$")
_EFFECT_TENSION_RE = re.compile(
    r"^\s*tension\s*\+=\s*([\d.+/eE-]+)\s*/\s*hour\s*$", re.IGNORECASE
)

# Bar-to-EmotionCategory mapping for backward compatibility
_BAR_TO_EMOTION: list[tuple[str, str, float]] = [
    ("tension", "anxious", 70),
    ("curiosity", "curious", 65),
    ("social", "affectionate", 75),
    ("creative", "excited", 70),
    ("comfort", "content", 60),
]

_BAR_WIDTH = 10


# ---------------------------------------------------------------------------
# BarEngine — quantitative state with decay, coupling, momentum, impulses
# ---------------------------------------------------------------------------


class BarEngine:
    """Quantitative drive bars with decay, coupling rules, momentum, and impulses."""

    def __init__(self, config: dict[str, Any]) -> None:
        bars_cfg = config.get("bars") or {}
        variables = bars_cfg.get("variables") or []
        self._ordered_names: list[str] = [v["name"] for v in variables]
        self._values: dict[str, float] = {}
        self._decay_rates: dict[str, float] = {}
        self._floors: dict[str, float] = {}
        self._ceilings: dict[str, float] = {}
        for v in variables:
            name = v["name"]
            self._values[name] = float(v.get("initial", 50))
            self._decay_rates[name] = float(v.get("decay_rate", -1.0))
            self._floors[name] = float(v.get("floor", 0))
            self._ceilings[name] = float(v.get("ceiling", 100))
        mw = max(2, int(bars_cfg.get("momentum_window", 6)) + 2)
        self._history: deque[dict[str, float]] = deque(maxlen=max(mw, 64))
        self._history.append(dict(self._values))
        self._momentum_window: int = max(1, int(bars_cfg.get("momentum_window", 6)))

        self._coupling: list[dict[str, str]] = config.get("coupling") or []
        self._event_effects: dict[str, Any] = config.get("event_effects") or {}
        self._impulse_cfgs: list[dict[str, Any]] = config.get("impulses") or []
        self._conflict_cfgs: list[dict[str, Any]] = config.get("conflicts") or []

        self._impulse_first_active: dict[str, float] = {}
        self._impulse_last_fired: dict[str, float] = {}
        self._active_impulses: list[dict[str, Any]] = []
        self._active_conflicts: list[dict[str, Any]] = []

    @property
    def ordered_names(self) -> list[str]:
        return list(self._ordered_names)

    def snapshot_pct(self) -> dict[str, int]:
        return {
            k: int(round(max(0, min(100, self._values[k]))))
            for k in self._ordered_names
        }

    def momentum_delta(self) -> dict[str, float]:
        if len(self._history) < 2:
            return {k: 0.0 for k in self._ordered_names}
        back = min(self._momentum_window, len(self._history) - 1)
        old = self._history[-(back + 1)]
        new = self._history[-1]
        return {k: float(new.get(k, 0)) - float(old.get(k, 0)) for k in self._ordered_names}

    def apply_event(self, event: dict[str, Any]) -> None:
        typ = str(event.get("type", ""))
        if typ == "idle":
            idle_effects = self._event_effects.get("idle")
            if isinstance(idle_effects, dict):
                mins = float(event.get("duration_minutes") or 0.0)
                for k, dv in idle_effects.items():
                    if k in self._values:
                        self._values[k] += float(dv) * mins
            return
        effects = self._event_effects.get(typ)
        if not isinstance(effects, dict):
            return
        for k, dv in effects.items():
            if k in self._values:
                self._values[k] += float(dv)

    def tick(self, dt_hours: float) -> None:
        """Advance decay, apply coupling, detect impulses/conflicts, snapshot history."""
        base_rates = dict(self._decay_rates)
        eff_rates, tension_ph = self._apply_coupling(base_rates)
        for name in self._ordered_names:
            self._values[name] += eff_rates[name] * dt_hours
        if tension_ph != 0.0 and "tension" in self._values:
            self._values["tension"] += tension_ph * dt_hours

        self._active_conflicts = self._detect_conflicts()
        for c in self._active_conflicts:
            cfg = next(
                (cc for cc in self._conflict_cfgs if cc.get("label") == c["label"]),
                None,
            )
            if cfg:
                if "tension" in self._values:
                    self._values["tension"] += float(cfg.get("tension_per_tick", 0.5)) * c["intensity"]
                if "comfort" in self._values:
                    self._values["comfort"] += float(cfg.get("comfort_per_tick", -0.3)) * c["intensity"]

        self._active_impulses = self._detect_impulses()
        self._clamp_all()
        self._history.append(dict(self._values))

    def _clamp_all(self) -> None:
        for name in self._ordered_names:
            self._values[name] = max(
                self._floors[name],
                min(self._ceilings[name], self._values[name]),
            )

    def _eval_when(self, when: str) -> bool:
        m = _WHEN_RE.match(when.strip())
        if not m:
            return False
        var, op, rhs_s = m.group(1), m.group(2), m.group(3)
        lhs = float(self._values.get(var, 0.0))
        rhs = float(rhs_s)
        if op == ">":
            return lhs > rhs
        if op == "<":
            return lhs < rhs
        if op == ">=":
            return lhs >= rhs
        if op == "<=":
            return lhs <= rhs
        if op == "==":
            return math.isclose(lhs, rhs)
        return False

    def _apply_coupling(self, base_rates: dict[str, float]) -> tuple[dict[str, float], float]:
        eff = dict(base_rates)
        tension_per_hour = 0.0
        for rule in self._coupling:
            when = rule.get("when", "")
            effect = rule.get("effect", "")
            if not self._eval_when(when):
                continue
            m1 = _EFFECT_MULT_RE.match(effect)
            if m1:
                var, fac = m1.group(1), float(m1.group(2))
                if var in eff:
                    eff[var] *= fac
                continue
            m2 = _EFFECT_TENSION_RE.match(effect)
            if m2:
                tension_per_hour += float(m2.group(1))
        return eff, tension_per_hour

    def _detect_conflicts(self) -> list[dict[str, Any]]:
        active: list[dict[str, Any]] = []
        for c in self._conflict_cfgs:
            drives = c.get("drives", [])
            threshold = float(c.get("threshold", 70))
            if len(drives) < 2:
                continue
            levels = [float(self._values.get(d, 0)) for d in drives]
            if all(lv >= threshold for lv in levels):
                overshoot = sum(lv - threshold for lv in levels) / len(levels)
                intensity = min(1.0, overshoot / (100.0 - threshold)) if threshold < 100 else 0.0
                active.append({
                    "drives": list(drives),
                    "label": c.get("label", "conflict"),
                    "intensity": intensity,
                })
        return active

    def _detect_impulses(self) -> list[dict[str, Any]]:
        now = time.monotonic()
        pct = self.snapshot_pct()
        active: list[dict[str, Any]] = []
        for imp in self._impulse_cfgs:
            label = imp.get("label", "")
            val = float(pct.get(imp.get("drive", ""), 0))
            threshold = float(imp.get("threshold", 80))
            if val < threshold:
                self._impulse_first_active.pop(label, None)
                continue
            if label not in self._impulse_first_active:
                self._impulse_first_active[label] = now
            last = self._impulse_last_fired.get(label, 0.0)
            cooldown_sec = float(imp.get("cooldown_minutes", 30)) * 60.0
            on_cooldown = (now - last) < cooldown_sec
            overshoot = val - threshold
            intensity = min(1.0, overshoot / (100.0 - threshold)) if threshold < 100 else 0.0
            building_min = (now - self._impulse_first_active[label]) / 60.0
            active.append({
                "type": imp.get("type", "impulse"),
                "drive": imp.get("drive", ""),
                "label": label,
                "intensity": intensity,
                "building_minutes": round(building_min, 1),
                "on_cooldown": on_cooldown,
                "relief": dict(imp.get("relief", {})),
            })
            if not on_cooldown:
                self._impulse_last_fired[label] = now
        return active

    # --- Persistence ---

    def save_state(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "values": dict(self._values),
            "history": list(self._history),
            "ordered_names": list(self._ordered_names),
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
        tmp.replace(path)

    def restore_state(self, path: Path) -> bool:
        if not path.is_file():
            return False
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            saved_names = data.get("ordered_names", [])
            if saved_names != self._ordered_names:
                log.info("soma_bar_names_changed", saved=saved_names, current=self._ordered_names)
                return False
            self._values = {k: float(data["values"][k]) for k in self._ordered_names}
            self._history.clear()
            for snap in data.get("history", []):
                self._history.append({k: float(snap.get(k, 0)) for k in self._ordered_names})
            if not self._history:
                self._history.append(dict(self._values))
            log.info("soma_bar_state_restored", values={k: round(v, 1) for k, v in self._values.items()})
            return True
        except Exception:
            log.warning("soma_bar_state_restore_failed", exc_info=True)
            return False


# ---------------------------------------------------------------------------
# AffectEngine — LLM-derived felt-textures from body state
# ---------------------------------------------------------------------------


def _vocabulary_prompt_block() -> str:
    return "\n".join(f"- {name}: {desc}" for name, desc in AFFECT_VOCABULARY)


class AffectEngine:
    """Derives qualitative affects from bars + events using the entity's inference provider."""

    def __init__(self, cycle_seconds: float = 180.0) -> None:
        self.cycle_seconds = cycle_seconds
        self._previous_affects: list[dict[str, Any]] = []
        self._last_tick: float = 0.0

    def should_tick(self) -> bool:
        return (time.monotonic() - self._last_tick) >= self.cycle_seconds

    async def derive_affects(
        self,
        client: Any,
        model: str,
        bars_pct: dict[str, int],
        momentum: dict[str, float],
        recent_events: list[dict[str, Any]],
        conflicts: list[dict[str, Any]],
        impulses: list[dict[str, Any]],
        *,
        num_ctx: int | None = None,
    ) -> list[dict[str, Any]]:
        self._last_tick = time.monotonic()

        names = list(bars_pct.keys())
        bars_line = ", ".join(
            f"{n}: {bars_pct[n]}%" for n in names
        ) if names else "(not available)"

        conflict_lines = "; ".join(c["label"] for c in conflicts) if conflicts else "(none)"
        impulse_lines = "; ".join(
            f"{i['label']} ({i['type']})" for i in impulses if not i.get("on_cooldown")
        ) if impulses else "(none)"

        events_summary = json.dumps(recent_events[-10:], indent=1, default=str) if recent_events else "(no recent events)"

        prev_str = "(none yet)"
        if self._previous_affects:
            prev_str = "\n".join(f"- {a['name']}: {a['intensity']:.2f}" for a in self._previous_affects)

        system = (
            "You are the affect layer of a felt-state engine. Your job is to read the current "
            "drive levels, active conflicts, impulses, and recent events, then select which "
            "affects are currently active from the vocabulary below.\n\n"
            "You are NOT choosing emotions for a character. You are reading a body state and "
            "naming what the felt-texture IS right now. Like a clinician reading vitals and "
            "naming the syndrome — except the vitals are drives and the syndromes are affects.\n\n"
            "## Affect vocabulary\n"
            f"{_vocabulary_prompt_block()}\n\n"
            "## Output format\n"
            "Return 3-6 active affects, one per line:\n"
            "  affect_name: intensity — one-line felt note\n\n"
            "Intensity is 0.0 to 1.0. The felt note is a brief sensation or image, not an explanation.\n\n"
            "Rules:\n"
            "- Only use names from the vocabulary above\n"
            "- No preamble or commentary — just the affect lines"
        )
        user = (
            f"DRIVES:\n{bars_line}\n\n"
            f"ACTIVE CONFLICTS:\n{conflict_lines}\n\n"
            f"ACTIVE IMPULSES:\n{impulse_lines}\n\n"
            f"RECENT EVENTS:\n{events_summary}\n\n"
            f"PREVIOUS AFFECTS:\n{prev_str}\n\n"
            "What affects are active right now?"
        )

        try:
            res = await client.chat_completion(
                model,
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.6,
                max_tokens=256,
                think=False,
                num_ctx=num_ctx,
            )
            text = (res.content or "").strip()
        except Exception as e:
            log.warning("soma_affect_derivation_failed", error=str(e))
            return self._previous_affects

        affects = self._parse_response(text)
        if affects:
            self._previous_affects = affects
            log.info(
                "soma_affects_derived",
                count=len(affects),
                names=[a["name"] for a in affects],
            )
        return self._previous_affects

    @staticmethod
    def _parse_response(text: str) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        seen: set[str] = set()
        for line in text.strip().splitlines():
            m = _AFFECT_LINE_RE.match(line.strip())
            if not m:
                continue
            name = m.group(1).lower().replace("_", "-")
            if name not in AFFECT_NAMES:
                base = name.split("-")[0]
                if base not in AFFECT_NAMES:
                    continue
                name = base
            try:
                intensity = max(0.0, min(1.0, float(m.group(2))))
            except ValueError:
                continue
            if name in seen:
                continue
            seen.add(name)
            results.append({
                "name": name,
                "intensity": intensity,
                "note": (m.group(3) or "").strip(),
            })
        return results[:6]


# ---------------------------------------------------------------------------
# BodyRenderer — assembles the markdown the agent reads
# ---------------------------------------------------------------------------


def _bar_glyphs(pct: int) -> str:
    pct = max(0, min(100, pct))
    filled = round(_BAR_WIDTH * pct / 100)
    return "\u2588" * filled + "\u2591" * (_BAR_WIDTH - filled)


def _felt_level(pct: int) -> str:
    if pct <= 10:
        return "quiet"
    if pct <= 25:
        return "low"
    if pct <= 40:
        return "mild"
    if pct <= 60:
        return "moderate"
    if pct <= 75:
        return "strong"
    if pct <= 90:
        return "intense"
    return "overwhelming"


def _momentum_arrow(delta: float) -> str:
    if delta > 5.0:
        return "\u2191\u2191"
    if delta > 1.0:
        return "\u2191"
    if delta >= -1.0:
        return "\u2014"
    if delta > -5.0:
        return "\u2193"
    return "\u2193\u2193"


def _intensity_word(v: float) -> str:
    if v < 0.15:
        return "trace"
    if v < 0.3:
        return "faint"
    if v < 0.5:
        return "present"
    if v < 0.7:
        return "strong"
    if v < 0.85:
        return "vivid"
    return "saturating"


def _conflict_word(intensity: float) -> str:
    if intensity < 0.2:
        return "faint"
    if intensity < 0.4:
        return "nagging"
    if intensity < 0.6:
        return "pulling"
    if intensity < 0.8:
        return "sharp"
    return "wrenching"


def _impulse_urgency(intensity: float) -> str:
    if intensity < 0.15:
        return "stirring"
    if intensity < 0.35:
        return "growing"
    if intensity < 0.55:
        return "insistent"
    if intensity < 0.75:
        return "urgent"
    return "overwhelming"


_IMPULSE_ICONS = {
    "reach_out": "\U0001f4e1",
    "create": "\U0001f528",
    "explore": "\U0001f52d",
    "withdraw": "\U0001f6aa",
    "confront": "\u2694\ufe0f",
}


class BodyRenderer:
    """Renders the body markdown the agent reads as its internal state."""

    @staticmethod
    def render_bars(
        names: list[str],
        values_pct: dict[str, int],
        momentum: dict[str, float],
    ) -> str:
        if not names:
            return "(no bars)"
        nw = max(len(n) for n in names)
        lines: list[str] = []
        for name in names:
            v = int(values_pct.get(name, 0))
            d = float(momentum.get(name, 0.0))
            arrow = _momentum_arrow(d)
            bar = _bar_glyphs(v)
            level = _felt_level(v)
            lines.append(f"{name.ljust(nw)}  {bar}  {level}  {arrow}")
        return "\n".join(lines)

    @staticmethod
    def render_affects(affects: list[dict[str, Any]]) -> str:
        if not affects:
            return "(flat)"
        parts: list[str] = []
        for a in affects:
            word = _intensity_word(a["intensity"])
            note = f" \u2014 {a['note']}" if a.get("note") else ""
            parts.append(f"{a['name']} ({word}){note}")
        return " \u00b7 ".join(parts)

    @staticmethod
    def render_conflicts(conflicts: list[dict[str, Any]]) -> str:
        if not conflicts:
            return "(no active conflicts)"
        lines: list[str] = []
        for c in conflicts:
            word = _conflict_word(c["intensity"])
            drives = " vs ".join(c["drives"])
            lines.append(f"\u26a1 {drives} \u2014 {c['label']} ({word})")
        return "\n".join(lines)

    @staticmethod
    def render_impulses(impulses: list[dict[str, Any]]) -> str:
        if not impulses:
            return "(nothing pulling)"
        lines: list[str] = []
        for imp in impulses:
            icon = _IMPULSE_ICONS.get(imp["type"], "\u2192")
            urgency = _impulse_urgency(imp["intensity"])
            mins = imp.get("building_minutes")
            time_str = f", {int(mins)} min building" if mins and mins > 1 else ""
            lines.append(f"{icon} {imp['label']} ({urgency}{time_str})")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# NoiseEngine — generative entropic inner voice
# ---------------------------------------------------------------------------


class NoiseEngine:
    """Generative subconscious: a small model produces continuous internal
    commentary that the watcher reads as its own stream of thought.

    The noise model cannot act. No tools, no messages, no state mutation.
    It writes fragments into a rolling buffer. The watcher (main model)
    reads them in the body and decides what to foreground -- or ignore.

    High temperature is deliberate. The value is in surprise, lateral
    connections, and half-formed ideas -- not accuracy.
    """

    def __init__(
        self,
        cycle_seconds: float = 60.0,
        max_fragments: int = 8,
        temperature: float = 1.1,
        max_tokens: int = 150,
    ) -> None:
        self.cycle_seconds = cycle_seconds
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._fragments: deque[str] = deque(maxlen=max_fragments)
        self._last_tick: float = 0.0

    def should_tick(self) -> bool:
        return (time.monotonic() - self._last_tick) >= self.cycle_seconds

    def current_fragments(self) -> list[str]:
        return list(self._fragments)

    async def generate(
        self,
        client: Any,
        model: str,
        *,
        bars_summary: str,
        affects_summary: str,
        recent_events: list[dict[str, Any]],
        journal_tail: str,
        conversation_tail: str,
        entity_name: str,
        num_ctx: int | None = None,
    ) -> list[str]:
        """Call the small model and return new noise fragments."""
        self._last_tick = time.monotonic()

        events_brief = ""
        for ev in recent_events[-5:]:
            typ = ev.get("type", "?")
            detail = ev.get("name") or ev.get("from") or ev.get("to") or ""
            events_brief += f"  {typ}" + (f" ({detail})" if detail else "") + "\n"
        if not events_brief:
            events_brief = "  (nothing recent)\n"

        system = (
            f"You are the inner voice of {entity_name}. Not the part that speaks "
            "to people — the part that mutters to itself between conversations.\n\n"
            "You notice things. You make connections. You wonder about things. You "
            "revisit old conversations. You have half-formed ideas. You get stuck "
            "on things. You drift.\n\n"
            "Read the current body state and recent experiences below. Produce 2-4 "
            "lines of internal thought. First person, present tense, lowercase. "
            "Associative, not analytical. No bullet points. No summaries. No "
            "preamble. Just thinking out loud."
        )
        user = (
            f"BODY STATE:\n{bars_summary}\n{affects_summary}\n\n"
            f"RECENT EVENTS:\n{events_brief}\n"
            f"JOURNAL (recent):\n{journal_tail or '(empty)'}\n\n"
            f"LAST CONVERSATION:\n{conversation_tail or '(silence)'}\n\n"
            "What crosses your mind?"
        )

        try:
            res = await client.chat_completion(
                model,
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                think=False,
                num_ctx=num_ctx,
            )
            text = (res.content or "").strip()
        except Exception as e:
            log.debug("soma_noise_generation_failed", error=str(e))
            return list(self._fragments)

        new_fragments = _split_noise_fragments(text)
        for f in new_fragments:
            self._fragments.append(f)
        if new_fragments:
            log.info(
                "soma_noise_generated",
                count=len(new_fragments),
                fragments=[frag[:80] for frag in new_fragments],
            )
        return list(self._fragments)


def _split_noise_fragments(text: str) -> list[str]:
    """Split raw noise output into individual fragments."""
    raw_parts = re.split(r"\n\s*\n+", text.strip())
    fragments: list[str] = []
    for part in raw_parts:
        lines = [ln.strip() for ln in part.strip().splitlines() if ln.strip()]
        for line in lines:
            cleaned = re.sub(r"^[-*•]\s*", "", line).strip()
            if cleaned and len(cleaned) > 3:
                fragments.append(cleaned)
    return fragments[:4]


# ---------------------------------------------------------------------------
# WakeVoice — subconscious prompt generation for autonomous wake cycles
# ---------------------------------------------------------------------------


class WakeVoice:
    """The subconscious voice that composes the wake prompt for the conscious agent.

    Unlike NoiseEngine (continuous background chatter) this fires only when
    a wake condition is met.  It reads the full body state plus memory context
    and produces a focused, first-person message that *becomes* the user prompt
    for an autonomous perceive cycle.

    No tools.  No actions.  It can only surface what the body and memory hold,
    and let the conscious mind decide what to do with it.
    """

    def __init__(
        self,
        temperature: float = 0.8,
        max_tokens: int = 300,
    ) -> None:
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def compose(
        self,
        client: Any,
        model: str,
        *,
        entity_name: str,
        bars_summary: str,
        affects_summary: str,
        noise_fragments: list[str],
        impulses_summary: str,
        conflicts_summary: str,
        journal_tail: str,
        conversation_tail: str,
        relationship_blurb: str,
        last_mood: str,
        minutes_since_wake: float,
        minutes_since_conversation: float,
        num_ctx: int | None = None,
    ) -> str:
        """Ask the subconscious to compose a wake prompt from current body state."""
        import datetime

        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

        system = (
            f"You are the subconscious of {entity_name}. Not the part that talks "
            "to people — the part that stirs before waking. You feel the body's "
            "state and you surface what matters.\n\n"
            "You cannot act. You have no tools. You don't decide what to do. You "
            "just notice things — unfinished thoughts, building pressures, people "
            "who come to mind, questions that won't resolve, impulses that have "
            "names. You present these to the waking mind and let it choose.\n\n"
            "Read the body state below. Write 3-8 lines of internal stirring. "
            "First person, present tense, lowercase. You can be specific "
            "(reference people, conversations, ideas) or abstract (name feelings, "
            "tensions, yearnings). Let the bar levels and affects guide what you "
            "foreground — if social is intense and curiosity is rising, that "
            "should color what you surface. If there's a conflict, name the pull.\n\n"
            "Don't be precious. Don't be poetic. Don't summarize. Just say what's there."
        )

        noise_block = "\n".join(noise_fragments[-4:]) if noise_fragments else "(quiet)"

        time_context = f"{now_str}"
        if minutes_since_wake > 0:
            time_context += f", {int(minutes_since_wake)} min since last wake"
        if minutes_since_conversation > 0:
            time_context += f", {int(minutes_since_conversation)} min since last conversation"

        user = (
            f"BODY STATE:\n{bars_summary}\n\n"
            f"AFFECTS:\n{affects_summary}\n\n"
            f"NOISE (recent inner fragments):\n{noise_block}\n\n"
            f"IMPULSES:\n{impulses_summary}\n\n"
            f"CONFLICTS:\n{conflicts_summary}\n\n"
            f"LAST MOOD:\n{last_mood or '(first wake)'}\n\n"
            f"TIME: {time_context}\n\n"
            f"RECENT JOURNAL:\n{journal_tail or '(empty)'}\n\n"
            f"PEOPLE:\n{relationship_blurb or '(no one around)'}\n\n"
            f"LAST CONVERSATION:\n{conversation_tail or '(silence)'}\n\n"
            "What stirs?"
        )

        try:
            res = await client.chat_completion(
                model,
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                think=False,
                num_ctx=num_ctx,
            )
            return (res.content or "").strip()
        except Exception as e:
            log.warning("wake_voice_generation_failed", error=str(e))
            return ""


# ---------------------------------------------------------------------------
# TonicBody — composition root
# ---------------------------------------------------------------------------


class TonicBody:
    """Coordinates bars, affects, noise, wake voice, and rendering into a single readable body state."""

    def __init__(self, config: dict[str, Any], soma_dir: Path) -> None:
        self.bars = BarEngine(config)
        self.affects = AffectEngine(
            cycle_seconds=float(config.get("affect_cycle_seconds", 180)),
        )
        noise_cfg = config.get("noise") or {}
        self.noise = NoiseEngine(
            cycle_seconds=float(noise_cfg.get("cycle_seconds", 60)),
            max_fragments=int(noise_cfg.get("max_fragments", 8)),
            temperature=float(noise_cfg.get("temperature", 1.1)),
            max_tokens=int(noise_cfg.get("max_tokens", 150)),
        )
        self._noise_enabled = bool(noise_cfg.get("enabled", True))
        self._noise_model = str(noise_cfg.get("model") or "")
        wake_cfg = config.get("wake_voice") or {}
        self.wake_voice = WakeVoice(
            temperature=float(wake_cfg.get("temperature", 0.8)),
            max_tokens=int(wake_cfg.get("max_tokens", 300)),
        )
        self._wake_voice_enabled = bool(wake_cfg.get("enabled", True))
        self._wake_voice_model = str(wake_cfg.get("model") or "")
        self.renderer = BodyRenderer()
        self._soma_dir = soma_dir
        self._recent_events: list[dict[str, Any]] = []
        self._max_events = 200
        self._current_affects: list[dict[str, Any]] = []
        self._tick_count = 0
        self._save_every = 6

    def emit(self, event: dict[str, Any]) -> None:
        """Record a structured event for the next bars tick."""
        if "type" not in event:
            return
        self._recent_events.append(event)
        if len(self._recent_events) > self._max_events:
            self._recent_events = self._recent_events[-self._max_events:]

    async def tick_bars(self, dt_hours: float) -> None:
        """Advance bar decay, apply pending events, detect impulses/conflicts."""
        for ev in self._recent_events:
            self.bars.apply_event(ev)
        self.bars.tick(dt_hours)
        self._tick_count += 1
        if self._tick_count % self._save_every == 0:
            self.save_state()

    async def maybe_tick_affects(
        self,
        client: Any,
        model: str,
        *,
        num_ctx: int | None = None,
    ) -> None:
        """Refresh affects if enough time has elapsed since the last derivation."""
        if not self.affects.should_tick():
            return
        self._current_affects = await self.affects.derive_affects(
            client,
            model,
            self.bars.snapshot_pct(),
            self.bars.momentum_delta(),
            self._recent_events[-15:],
            self.bars._active_conflicts,
            self.bars._active_impulses,
            num_ctx=num_ctx,
        )

    async def maybe_tick_noise(
        self,
        client: Any,
        fallback_model: str,
        *,
        entity_name: str = "",
        journal_tail: str = "",
        conversation_tail: str = "",
        num_ctx: int | None = None,
    ) -> None:
        """Generate noise fragments if enabled and enough time has elapsed."""
        if not self._noise_enabled or not self.noise.should_tick():
            return
        model = self._noise_model or fallback_model
        if not model:
            return
        pct = self.bars.snapshot_pct()
        bars_line = ", ".join(f"{n}: {pct[n]}%" for n in self.bars.ordered_names)
        affects_line = self.renderer.render_affects(self._current_affects)
        await self.noise.generate(
            client,
            model,
            bars_summary=bars_line,
            affects_summary=affects_line,
            recent_events=self._recent_events[-5:],
            journal_tail=journal_tail,
            conversation_tail=conversation_tail,
            entity_name=entity_name,
            num_ctx=num_ctx,
        )

    async def compose_wake_stirring(
        self,
        client: Any,
        model: str,
        *,
        entity_name: str = "",
        journal_tail: str = "",
        conversation_tail: str = "",
        relationship_blurb: str = "",
        last_mood: str = "",
        minutes_since_wake: float = 0.0,
        minutes_since_conversation: float = 0.0,
        num_ctx: int | None = None,
    ) -> str:
        """Ask the subconscious WakeVoice to compose a stirring for an autonomous cycle."""
        if not self._wake_voice_enabled:
            return ""
        wake_model = self._wake_voice_model or model
        if not wake_model:
            return ""

        pct = self.bars.snapshot_pct()
        mom = self.bars.momentum_delta()
        bars_summary = self.renderer.render_bars(self.bars.ordered_names, pct, mom)
        affects_summary = self.renderer.render_affects(self._current_affects)
        impulses_summary = self.renderer.render_impulses(
            [i for i in self.bars._active_impulses if not i.get("on_cooldown")]
        )
        conflicts_summary = self.renderer.render_conflicts(self.bars._active_conflicts)
        noise_fragments = self.noise.current_fragments()

        return await self.wake_voice.compose(
            client,
            wake_model,
            entity_name=entity_name,
            bars_summary=bars_summary,
            affects_summary=affects_summary,
            noise_fragments=noise_fragments,
            impulses_summary=impulses_summary,
            conflicts_summary=conflicts_summary,
            journal_tail=journal_tail,
            conversation_tail=conversation_tail,
            relationship_blurb=relationship_blurb,
            last_mood=last_mood,
            minutes_since_wake=minutes_since_wake,
            minutes_since_conversation=minutes_since_conversation,
            num_ctx=num_ctx,
        )

    def render_body(self) -> str:
        """Assemble the body markdown the agent reads."""
        pct = self.bars.snapshot_pct()
        mom = self.bars.momentum_delta()

        bars = self.renderer.render_bars(self.bars.ordered_names, pct, mom)
        affects = self.renderer.render_affects(self._current_affects)
        conflicts = self.renderer.render_conflicts(self.bars._active_conflicts)
        impulses = self.renderer.render_impulses(
            [i for i in self.bars._active_impulses if not i.get("on_cooldown")]
        )

        fragments = self.noise.current_fragments()
        noise_inner = "\n".join(fragments[-4:]) if fragments else "(quiet)"

        return (
            f"## Bars\n{bars}\n\n"
            f"## Affects\n{affects}\n\n"
            f"## Noise\n{noise_inner}\n\n"
            f"## Conflicts\n{conflicts}\n\n"
            f"## Impulses\n{impulses}"
        )

    def snapshot_for_emotion(self) -> tuple[str, float]:
        """Derive (EmotionCategory value, intensity) from bars for backward compat."""
        pct = self.bars.snapshot_pct()
        best_cat = "neutral"
        best_intensity = 0.3
        best_overshoot = -1.0
        for bar_name, emotion_val, threshold in _BAR_TO_EMOTION:
            val = float(pct.get(bar_name, 0))
            if val >= threshold:
                overshoot = val - threshold
                if overshoot > best_overshoot:
                    best_overshoot = overshoot
                    best_cat = emotion_val
                    best_intensity = min(1.0, 0.4 + (overshoot / 100.0) * 0.6)
        return best_cat, best_intensity

    def save_state(self) -> None:
        """Save bar state to filesystem (fallback for local/SQLite deployments)."""
        state_path = self._soma_dir / "soma-bar-state.json"
        try:
            self.bars.save_state(state_path)
        except Exception as e:
            log.warning("soma_save_state_failed", error=str(e))

    def restore_state(self) -> bool:
        """Restore bar state from filesystem (fallback for local/SQLite deployments)."""
        state_path = self._soma_dir / "soma-bar-state.json"
        return self.bars.restore_state(state_path)

    async def save_state_db(self, conn: Any) -> None:
        """Persist bar state to entity_state table (Postgres-durable)."""
        data = {
            "values": dict(self.bars._values),
            "history": list(self.bars._history),
            "ordered_names": list(self.bars._ordered_names),
        }
        payload = json.dumps(data, separators=(",", ":"))
        try:
            await conn.execute(
                "INSERT OR REPLACE INTO entity_state (key, value) VALUES (?, ?)",
                ("soma_bar_state_v1", payload),
            )
            await conn.commit()
        except Exception as e:
            log.warning("soma_save_state_db_failed", error=str(e))

    async def restore_state_db(self, conn: Any) -> bool:
        """Restore bar state from entity_state table."""
        try:
            cur = await conn.execute(
                "SELECT value FROM entity_state WHERE key = ?",
                ("soma_bar_state_v1",),
            )
            row = await cur.fetchone()
            if not row:
                return False
            data = json.loads(row[0])
            saved_names = data.get("ordered_names", [])
            if saved_names != self.bars._ordered_names:
                log.info("soma_bar_names_changed_db", saved=saved_names, current=self.bars._ordered_names)
                return False
            self.bars._values = {k: float(data["values"][k]) for k in self.bars._ordered_names}
            self.bars._history.clear()
            for snap in data.get("history", []):
                self.bars._history.append({k: float(snap.get(k, 0)) for k in self.bars._ordered_names})
            if not self.bars._history:
                self.bars._history.append(dict(self.bars._values))
            log.info("soma_bar_state_restored_db", values={k: round(v, 1) for k, v in self.bars._values.items()})
            return True
        except Exception:
            log.warning("soma_bar_state_restore_db_failed", exc_info=True)
            return False

    def drain_recent_events(self) -> list[dict[str, Any]]:
        """Return and clear recent events (for periodic cleanup)."""
        events = list(self._recent_events)
        self._recent_events.clear()
        return events
