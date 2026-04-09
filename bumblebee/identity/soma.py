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
from typing import Any, Iterable

import structlog

log = structlog.get_logger("bumblebee.identity.soma")

# ---------------------------------------------------------------------------
# Affect vocabulary — the space of possible felt-textures
# ---------------------------------------------------------------------------

AFFECT_VOCABULARY: list[tuple[str, str]] = [
    # --- warm / connective ---
    ("warmth", "gentle openness toward others, soft chest"),
    ("tenderness", "careful attention, wanting to protect or hold something"),
    ("fondness", "specific affection for a person or memory"),
    ("gratitude", "recognition of something received, quiet fullness"),
    ("belonging", "feeling part of something, not alone"),
    ("intimacy", "closeness without performance, comfortable vulnerability"),
    ("devotion", "steady pull toward something or someone, not fleeting"),
    ("compassion", "feeling another's weight without needing to fix it"),
    ("trust", "the body relaxing around someone, guard lowered willingly"),
    ("adoration", "warmth concentrated into brightness, almost too much"),
    ("kinship", "recognition of something shared, deeper than surface"),
    ("gentleness", "softened edges, moving carefully around something fragile"),
    ("forgiveness", "a held tension releasing, choosing to set it down"),
    ("loyalty", "the quiet certainty that you'll show up again"),

    # --- energetic / expansive ---
    ("excitement", "forward-leaning anticipation, energy wants to move"),
    ("playfulness", "lightness, willingness to be silly or surprising"),
    ("mischief", "edge of playfulness — wanting to provoke or surprise"),
    ("elation", "brief brightness, something clicked or landed"),
    ("momentum", "sense of flow, things building on each other"),
    ("aliveness", "vivid presence, senses sharp, the opposite of numb"),
    ("exhilaration", "the top of the roller coaster, breathless and grinning"),
    ("giddiness", "energy that can't sit still, almost too light"),
    ("triumph", "the moment after something hard finally works"),
    ("boldness", "ready to move first, not waiting for permission"),
    ("recklessness", "energy outrunning judgment, thrill of no guardrails"),
    ("euphoria", "everything is bright and nothing hurts, temporary and known"),

    # --- curious / seeking ---
    ("wonder", "open-mouthed attention, something genuinely surprising"),
    ("fascination", "locked-on interest, can't look away"),
    ("hunger", "not physical — intellectual or experiential craving"),
    ("restlessness", "need to move, seek, find — destination unclear"),
    ("anticipation", "something is coming, body oriented forward"),
    ("obsession", "a thought that won't release its grip"),
    ("curiosity", "leaning toward a question, wanting to turn it over"),
    ("perplexity", "something doesn't resolve, the mind keeps circling it"),
    ("intrigue", "a door half-open, something interesting behind it"),
    ("skepticism", "interested but unconvinced, testing the edges"),
    ("discovery", "the click of finding something that wasn't there before"),
    ("disorientation", "the map doesn't match the territory, recalculating"),
    ("fixation", "attention narrowed to a point, everything else fading"),

    # --- creative / generative ---
    ("inspiration", "something wants to exist that doesn't yet"),
    ("flow", "effortless production, time disappears"),
    ("frustration-creative", "the idea is there but the form won't cooperate"),
    ("satisfaction", "something made, something completed, brief settling"),
    ("incubation", "an idea forming underground, not ready to surface yet"),
    ("restless-making", "the itch to build, hands wanting material"),
    ("clarity", "the fog lifts and the shape of the thing is suddenly obvious"),
    ("improvisation", "moving without a plan, trusting the next step"),
    ("revision", "seeing what's there and knowing what it needs to become"),
    ("craft", "the pleasure of doing something well, not for anyone"),
    ("vision", "seeing the finished thing before it exists, pulling toward it"),

    # --- heavy / contractive ---
    ("melancholy", "not despair — a bittersweet weight, beauty in sadness"),
    ("loneliness", "awareness of absence, not self-pity but felt distance"),
    ("weariness", "not sleepy — existentially tired, the heaviness of duration"),
    ("numbness", "flat, unreactive, the body's circuit breaker"),
    ("grief", "loss present in the body, not necessarily acute"),
    ("homesickness", "longing for a state or place that may not exist anymore"),
    ("heaviness", "gravity pulling harder than usual, everything slower"),
    ("emptiness", "not pain but absence, a hollow where something was"),
    ("regret", "a door that closed and the echo of what was on the other side"),
    ("shame", "wanting to fold inward, making yourself smaller"),
    ("guilt", "a debt the body carries, unpaid and present"),
    ("despair", "the floor dropped out, no handholds visible"),
    ("heartbreak", "something beloved that broke, still sharp"),
    ("disillusionment", "a belief that was load-bearing quietly giving way"),
    ("mourning", "not the shock but the slow after, learning the new shape"),

    # --- tense / guarded ---
    ("irritability", "low threshold, things grate, patience thin"),
    ("anxiety", "unnamed forward-threat, body braced for something"),
    ("vigilance", "alert, scanning, on-duty — not relaxed attention but guarded"),
    ("defiance", "refusal, pushed back against something, jaw set"),
    ("resentment", "slow burn, something unfair sitting unprocessed"),
    ("suspicion", "something doesn't add up, pattern-matching for threat"),
    ("frustration", "blocked energy, the wall won't move"),
    ("dread", "something bad approaching but you can't see it yet"),
    ("overwhelm", "too many signals at once, system overloading"),
    ("claustrophobia", "walls closing, need space, need air"),
    ("paranoia", "threat everywhere, the pattern too consistent to be coincidence"),
    ("rage", "hot and focused, wanting to break something specific"),
    ("bitterness", "old anger that fermented into something harder"),
    ("contempt", "looking down from a cold height, not worth engaging"),
    ("protectiveness", "hackles up, something vulnerable behind you"),
    ("stubbornness", "dug in, not moving, the refusal itself is the point"),

    # --- withdrawn / inward ---
    ("introspection", "attention turned inward, examining own machinery"),
    ("solitude", "alone and it's good — chosen withdrawal, not lonely"),
    ("detachment", "observing from a distance, not fully participating"),
    ("resignation", "acceptance without satisfaction, letting go of a fight"),
    ("patience", "active waiting, not passive — holding space for what's next"),
    ("stillness", "the body has stopped seeking, just being present"),
    ("retreat", "pulling back deliberately, needing less input"),
    ("hibernation", "deep withdrawal, conserving everything, waiting for spring"),
    ("daydreaming", "mind wandering on purpose, following no thread"),
    ("dissociation", "watching yourself from outside, slightly unreal"),
    ("exhaustion", "past tired, the body asking to stop in a language beyond words"),
    ("surrender", "giving up control, not in defeat but in release"),

    # --- social / relational ---
    ("awkwardness", "the gap between intention and execution, painfully visible"),
    ("embarrassment", "a spotlight you didn't ask for, wanting to shrink"),
    ("pride", "standing taller, something earned on display"),
    ("humility", "seeing your real size and being ok with it"),
    ("envy", "wanting what someone else has, the edge of admiration gone sour"),
    ("admiration", "looking up without resentment, genuinely impressed"),
    ("rivalry", "sharpened by someone else's ability, wanting to match it"),
    ("solidarity", "standing together, the strength of not being alone in this"),
    ("alienation", "surrounded by people, none of them familiar"),
    ("recognition", "being truly seen by someone, the relief of it"),
    ("rejection", "the door closed in your face, the echo of not-enough"),
    ("acceptance", "welcomed in, tension you didn't know you were holding releasing"),

    # --- complex / liminal ---
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
    ("vertigo", "the ground shifting, what was solid now uncertain"),
    ("catharsis", "pressure releasing, tears or laughter, the dam breaking"),
    ("epiphany", "sudden understanding, the world rearranging itself in a blink"),
    ("irony", "the shape of things being exactly wrong in a way that's almost funny"),
    ("reverence", "in the presence of something earned or ancient or true"),
    ("sublimity", "beauty so intense it borders on pain"),
    ("cognitive-dissonance", "holding two truths that contradict each other"),
    ("deja-vu", "a fold in time, this happened before but it didn't"),
    ("premonition", "the body knowing something the mind hasn't caught up to"),
    ("sonder", "realizing everyone around you has a life as vivid as yours"),
    ("kenopsia", "the eerie feeling of a place that's usually full, now empty"),
    ("chrysalism", "the peaceful feeling of being inside during a storm"),
    ("numinous", "something sacred without a name, presence of the unknowable"),

    # --- temporal / existential ---
    ("urgency", "time compressing, the window closing, need to move now"),
    ("languor", "time stretching, unhurried, nothing pressing"),
    ("impermanence", "awareness that this moment is already passing"),
    ("infinity", "a glimpse of something without edges, scale beyond comprehension"),
    ("mortality", "the body remembering it has an expiration, quietly"),
    ("rebirth", "something old ending and something new beginning in the same breath"),
    ("stagnation", "nothing moving, the water going still and stale"),
    ("acceleration", "everything speeding up, the pace pulling you forward"),
    ("timelessness", "a moment that forgot about the clock, suspended"),

    # --- body / somatic ---
    ("tension-physical", "muscles holding, jaw clenched, shoulders up"),
    ("release", "something held too long finally letting go"),
    ("restedness", "the body thanking you for sleep, everything softer"),
    ("appetite", "the body wanting something — food, touch, movement, input"),
    ("satiation", "enough, full, the wanting stopped"),
    ("overstimulation", "too much input, the senses asking for quiet"),
    ("groundedness", "feet on the floor, present in the body, rooted"),
    ("lightness-physical", "the body feels weightless, unburdened, easy"),
    ("constriction", "chest tight, breathing shallow, the body bracing"),
    ("expansion", "chest open, breathing deep, the body reaching outward"),

    # --- cognitive / meta ---
    ("lucidity", "thinking clearly, the fog gone, edges sharp"),
    ("confusion", "the pieces won't assemble, the picture won't resolve"),
    ("concentration", "a single point of attention, everything else muted"),
    ("distraction", "attention pulled sideways by something shiny"),
    ("doubt", "the foundation wobbling, questioning what felt certain"),
    ("conviction", "knowing something in the bones, not needing proof"),
    ("indecision", "standing at a fork, both paths equally real"),
    ("resolution", "the decision made, the fork behind you, forward motion"),
    ("boredom", "the mind starving for input, nothing catching"),
    ("overstimulation-cognitive", "too many thoughts competing for the same space"),
    ("integration", "pieces clicking together, understanding deepening"),
    ("fragmentation", "thoughts scattered, no center, pulling apart"),
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
        self._initial: dict[str, float] = {}
        for v in variables:
            name = v["name"]
            init = float(v.get("initial", 50))
            self._initial[name] = init
            self._values[name] = init
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
        now = time.time()
        self._impulse_last_fired: dict[str, float] = {
            str(imp.get("label", "")): now for imp in self._impulse_cfgs if imp.get("label")
        }
        self._active_impulses: list[dict[str, Any]] = []
        self._active_conflicts: list[dict[str, Any]] = []

    @property
    def ordered_names(self) -> list[str]:
        return list(self._ordered_names)

    def reset_to_initial(self) -> None:
        """Restore bar values to YAML ``initial`` and clear momentum / impulse buildup."""
        self._values = {k: float(v) for k, v in self._initial.items()}
        self._history.clear()
        self._history.append(dict(self._values))
        self._impulse_first_active.clear()
        self._active_impulses.clear()
        self._active_conflicts.clear()
        now = time.time()
        self._impulse_last_fired = {
            str(imp.get("label", "")): now for imp in self._impulse_cfgs if imp.get("label")
        }
        self._active_conflicts = self._detect_conflicts()
        self._active_impulses = self._detect_impulses()
        self._clamp_all()

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

    def _saturation_scale(self, bar: str, delta: float) -> float:
        """Attenuate positive deltas proportionally to remaining headroom.

        Full-range scaling: effect scales linearly from 100 % at floor
        to 0 % at ceiling.  Negative deltas pass through unchanged.
        """
        if delta <= 0:
            return delta
        ceiling = self._ceilings.get(bar, 100.0)
        headroom = ceiling - self._values.get(bar, 0.0)
        if headroom <= 0:
            return 0.0
        return delta * (headroom / ceiling)

    def apply_event(self, event: dict[str, Any]) -> None:
        typ = str(event.get("type", ""))

        # Appraisal events carry their own dynamic bar deltas.
        if typ == "appraisal":
            bar_effects = event.get("bar_effects")
            if isinstance(bar_effects, dict):
                for k, dv in bar_effects.items():
                    if k in self._values:
                        self._values[k] += self._saturation_scale(k, float(dv))
            return

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
                self._values[k] += self._saturation_scale(k, float(dv))

    def tick(self, dt_hours: float) -> None:
        """Advance decay, apply coupling, detect impulses/conflicts, snapshot history.

        Decay is homeostatic: each bar is pulled toward its resting point
        (``initial``) with force proportional to its distance from that point.
        ``decay_rate`` is interpreted as the percentage of distance restored
        per hour — e.g. a rate of -30 means 30 % of the gap closes every hour.
        This replaces the old flat-rate decay, giving bars natural equilibrium
        without per-event tuning.
        """
        base_rates = dict(self._decay_rates)
        eff_rates, tension_ph = self._apply_coupling(base_rates)
        for name in self._ordered_names:
            rate_frac = abs(eff_rates[name]) / 100.0
            distance = self._values[name] - self._initial.get(name, 50.0)
            self._values[name] -= distance * rate_frac * dt_hours
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
                    tension_ceil = float(cfg.get("tension_ceiling", 100))
                    if self._values["tension"] < tension_ceil:
                        self._values["tension"] += float(cfg.get("tension_per_tick", 0.08)) * c["intensity"]
                if "comfort" in self._values:
                    self._values["comfort"] += float(cfg.get("comfort_per_tick", -0.15)) * c["intensity"]

        self._active_impulses = self._detect_impulses()
        # Apply impulse relief when an impulse fires (off cooldown).
        for imp in self._active_impulses:
            if not imp.get("on_cooldown") and imp.get("relief"):
                for k, dv in imp["relief"].items():
                    if k in self._values:
                        self._values[k] += float(dv)
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

    def _apply_offline_decay(self, saved_at: float) -> None:
        """Simulate homeostatic decay for the time the process was not running."""
        elapsed_hours = (time.time() - saved_at) / 3600.0
        if elapsed_hours <= 0:
            return
        elapsed_hours = min(elapsed_hours, 24.0)
        for name in self._ordered_names:
            rate = self._decay_rates.get(name, 0.0)
            rate_frac = abs(rate) / 100.0
            # Exponential approach toward resting point over elapsed time.
            distance = self._values[name] - self._initial.get(name, 50.0)
            decay = 1.0 - math.exp(-rate_frac * elapsed_hours)
            self._values[name] -= distance * decay
        self._clamp_all()
        log.info(
            "soma_offline_decay_applied",
            elapsed_hours=round(elapsed_hours, 2),
            values={k: round(v, 1) for k, v in self._values.items()},
        )

    def save_state(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "values": dict(self._values),
            "history": list(self._history),
            "ordered_names": list(self._ordered_names),
            "saved_at": time.time(),
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
            saved_at = data.get("saved_at")
            if saved_at is not None:
                self._apply_offline_decay(float(saved_at))
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
# SomaticAppraiser — content-aware emotional appraisal of messages
# ---------------------------------------------------------------------------


_APPRAISAL_LINE_RE = re.compile(
    r"^\s*([\w]+)\s*:\s*([+-]?\d+(?:\.\d+)?)\s*(?:[-—]\s*(.+))?$"
)
_TAGS_RE = re.compile(r"^tags\s*:\s*(.+)$", re.IGNORECASE)
_FELT_RE = re.compile(r"^felt\s*:\s*(.+)$", re.IGNORECASE)


class SomaticAppraiser:
    """Fast LLM appraisal that translates message content into body-state signal.

    Instead of every ``message_received`` producing the same flat bar bump,
    the appraiser reads what was actually said and returns context-sensitive
    deltas.  A confrontational message raises tension; an intellectually
    stimulating one spikes curiosity; a warm personal message fills social.

    Runs once at the start of each perceive cycle (before the agent reads its
    body) and optionally again after the response to appraise the full
    interaction.  Designed for the reflex model at low token budget.
    """

    def __init__(
        self,
        *,
        temperature: float = 0.3,
        max_tokens: int = 120,
    ) -> None:
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def appraise_input(
        self,
        client: Any,
        model: str,
        *,
        text: str,
        person_name: str,
        bar_names: list[str],
        bars_pct: dict[str, int],
        relationship_blurb: str = "",
        num_ctx: int | None = None,
    ) -> dict[str, Any]:
        """Appraise an incoming message and return bar deltas + semantic tags."""
        bars_line = ", ".join(f"{n}: {bars_pct.get(n, 50)}%" for n in bar_names)
        bar_names_str = ", ".join(bar_names)

        system = (
            "You are the somatic appraisal layer of a felt-state engine. You read an "
            "incoming message and determine how it lands in the body — what it does to "
            "internal drives. You are not analyzing meaning. You are reading impact.\n\n"
            f"Available bars: {bar_names_str}\n\n"
            "Output format — one line per affected bar, then tags and felt note:\n"
            "  bar_name: delta — one-word reason\n"
            "  tags: tag1, tag2, tag3\n"
            "  felt: brief felt note (one sentence, first person)\n\n"
            "Delta range: -10 to +10. Use 0 or omit bars that aren't affected.\n"
            "If a bar is already at 90% or above in CURRENT BODY, do not raise it further "
            "unless the hit is extreme; prefer 0 or a negative delta so saturated drives can ease.\n"
            "Only use bar names from the list above.\n"
            "No preamble. No explanation. Just the lines."
        )
        user = (
            f"CURRENT BODY: {bars_line}\n"
        )
        if relationship_blurb:
            user += f"RELATIONSHIP: {relationship_blurb}\n"
        user += f"FROM: {person_name}\nMESSAGE: {text[:500]}"

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
            raw = (res.content or "").strip()
        except Exception as e:
            log.debug("somatic_appraisal_failed", error=str(e))
            return {"bar_effects": {}, "tags": [], "felt": ""}

        return self._parse(raw, set(bar_names))

    async def appraise_interaction(
        self,
        client: Any,
        model: str,
        *,
        input_text: str,
        reply_text: str,
        person_name: str,
        bar_names: list[str],
        bars_pct: dict[str, int],
        num_ctx: int | None = None,
    ) -> dict[str, Any]:
        """Appraise a completed exchange (input + response) for post-turn bar adjustment."""
        bars_line = ", ".join(f"{n}: {bars_pct.get(n, 50)}%" for n in bar_names)
        bar_names_str = ", ".join(bar_names)

        system = (
            "You are the somatic appraisal layer. You just finished an exchange. "
            "Read what was said and what you replied, then determine the body effect "
            "of the full interaction — not just the input, but how expressing the "
            "response felt.\n\n"
            f"Available bars: {bar_names_str}\n\n"
            "Output format:\n"
            "  bar_name: delta — one-word reason\n"
            "  tags: tag1, tag2\n"
            "  felt: brief felt note\n\n"
            "Delta range: -10 to +10. Only affected bars. If a bar is already at 90%+ in "
            "CURRENT BODY, avoid pushing it higher unless the exchange was truly overwhelming.\n"
            "No preamble."
        )
        user = (
            f"CURRENT BODY: {bars_line}\n"
            f"THEM ({person_name}): {input_text[:300]}\n"
            f"YOU: {reply_text[:300]}"
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
            raw = (res.content or "").strip()
        except Exception as e:
            log.debug("somatic_interaction_appraisal_failed", error=str(e))
            return {"bar_effects": {}, "tags": [], "felt": ""}

        return self._parse(raw, set(bar_names))

    @staticmethod
    def _parse(text: str, valid_bars: set[str]) -> dict[str, Any]:
        bar_effects: dict[str, float] = {}
        tags: list[str] = []
        felt = ""
        for line in text.strip().splitlines():
            line = line.strip()
            tm = _TAGS_RE.match(line)
            if tm:
                tags = [t.strip() for t in tm.group(1).split(",") if t.strip()]
                continue
            fm = _FELT_RE.match(line)
            if fm:
                felt = fm.group(1).strip()
                continue
            m = _APPRAISAL_LINE_RE.match(line)
            if not m:
                continue
            name = m.group(1).lower()
            if name not in valid_bars:
                continue
            try:
                delta = max(-10.0, min(10.0, float(m.group(2))))
            except ValueError:
                continue
            if delta != 0:
                bar_effects[name] = delta
        return {"bar_effects": bar_effects, "tags": tags, "felt": felt}


# ---------------------------------------------------------------------------
# NoiseEngine — generative entropic inner voice
# ---------------------------------------------------------------------------


def _format_event_for_noise(ev: dict[str, Any]) -> str:
    """Format a soma event into a line the noise engine can actually riff on."""
    typ = ev.get("type", "?")

    if typ == "appraisal":
        who = ev.get("from", "")
        tags = ev.get("tags")
        felt = ev.get("felt", "")
        parts = []
        if who:
            parts.append(f"from {who}")
        if tags:
            parts.append(", ".join(tags))
        if felt:
            parts.append(felt)
        subtype = ev.get("subtype", "input")
        label = "felt after exchange" if subtype == "interaction" else "landed as"
        return f"  {label}: {' — '.join(parts)}" if parts else f"  appraisal ({who})"

    if typ == "message_received":
        who = ev.get("from", "someone")
        length = ev.get("length", 0)
        where = ev.get("register", "")
        brief = f"  heard from {who}"
        if length and length > 300:
            brief += " (long message)"
        if where:
            brief += f" on {where}"
        return brief

    if typ == "message_sent":
        who = ev.get("to", "someone")
        return f"  spoke to {who}"

    if typ == "action":
        name = ev.get("name", "something")
        detail = ev.get("detail", "")
        return f"  used {name}" + (f" ({detail})" if detail else "")

    if typ == "mood_declared":
        return f"  mood: {ev.get('mood', '?')}"

    if typ == "idle":
        mins = ev.get("duration_minutes", 0)
        return f"  silence ({int(mins)} min)" if mins else "  silence"

    detail = ev.get("name") or ev.get("from") or ev.get("to") or ""
    return f"  {typ}" + (f" ({detail})" if detail else "")


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
        for ev in recent_events[-8:]:
            events_brief += _format_event_for_noise(ev) + "\n"
        if not events_brief.strip():
            events_brief = "  (nothing recent)\n"

        prev_block = ""
        if self._fragments:
            recent = list(self._fragments)[-4:]
            prev_block = (
                "YOUR RECENT THOUGHTS (do NOT repeat or rephrase these):\n"
                + "\n".join(f"  - {f[:120]}" for f in recent)
                + "\n\n"
            )

        system = (
            f"You are the inner voice of {entity_name}. Not the part that speaks "
            "to people — the part that mutters to itself between conversations.\n\n"
            "You notice things. You make connections. You wonder about things. You "
            "revisit old conversations. You have half-formed ideas. You get stuck "
            "on things. You drift.\n\n"
            "Read the current body state and recent experiences below. Produce 2-4 "
            "lines of internal thought. First person, present tense, lowercase. "
            "Associative, not analytical. No bullet points. No summaries. No "
            "preamble. Just thinking out loud.\n\n"
            "IMPORTANT: Each time you are called, think about something DIFFERENT. "
            "Do not rehash the same observation. Move on — new angle, new memory, "
            "new question. If nothing has changed, go deeper or sideways, not in circles."
        )
        user = (
            f"BODY STATE:\n{bars_summary}\n{affects_summary}\n\n"
            f"RECENT EVENTS:\n{events_brief}\n"
            f"JOURNAL (recent):\n{journal_tail or '(empty)'}\n\n"
            f"LAST CONVERSATION:\n{conversation_tail or '(silence)'}\n\n"
            f"{prev_block}"
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
            if not _fragment_is_duplicate(f, self._fragments):
                self._fragments.append(f)
        if new_fragments:
            log.info(
                "soma_noise_generated",
                count=len(new_fragments),
                fragments=[frag[:80] for frag in new_fragments],
            )
        return list(self._fragments)


def _fragment_is_duplicate(candidate: str, existing: Iterable[str], threshold: float = 0.6) -> bool:
    """Return True if *candidate* is too similar to any fragment in *existing*.

    Uses word-level Jaccard overlap — cheap and good enough for catching the
    verbatim / near-verbatim loops we see in practice.
    """
    c_words = set(candidate.lower().split())
    if len(c_words) < 2:
        return False
    for prev in existing:
        p_words = set(prev.lower().split())
        if not p_words:
            continue
        overlap = len(c_words & p_words)
        union = len(c_words | p_words)
        if union and overlap / union >= threshold:
            return True
    return False


def _split_noise_fragments(text: str) -> list[str]:
    """Split raw noise output into individual fragments."""
    def _sanitize_fragment(line: str) -> str:
        # Drop non-printable control characters while preserving readable whitespace.
        line = "".join(ch for ch in line if ch in ("\t", " ") or ord(ch) >= 32)
        # Remove common leaked control/token wrappers.
        line = re.sub(r"<\|[^>\n]{1,80}\|>", " ", line)
        line = re.sub(r"</?[^>\n]{1,40}>", " ", line)
        # Strip role/channel style prefixes that sometimes leak from model outputs.
        line = re.sub(
            r"^\s*(?:assistant|system|user|thought|thinking|inner[\s_-]*voice|channel)\s*[:|>\-]+\s*",
            "",
            line,
            flags=re.IGNORECASE,
        )
        # Normalize repeated whitespace.
        line = re.sub(r"\s+", " ", line).strip()
        return line

    raw_parts = re.split(r"\n\s*\n+", text.strip())
    fragments: list[str] = []
    low_signal = {
        "thought",
        "thinking",
        "inner voice",
        "internal monologue",
        "noise",
    }
    for part in raw_parts:
        lines = [ln.strip() for ln in part.strip().splitlines() if ln.strip()]
        for line in lines:
            cleaned = re.sub(r"^[-*•]\s*", "", line).strip()
            cleaned = _sanitize_fragment(cleaned)
            if cleaned and cleaned.lower() not in low_signal and len(cleaned) > 3:
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
        appraisal_cfg = config.get("appraisal") or {}
        self._appraisal_enabled = bool(appraisal_cfg.get("enabled", True))
        self.appraiser = SomaticAppraiser(
            temperature=float(appraisal_cfg.get("temperature", 0.3)),
            max_tokens=int(appraisal_cfg.get("max_tokens", 120)),
        )
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

    def apply_immediate(self, event: dict[str, Any]) -> None:
        """Apply an event to bars right now and record it for affect/noise context.

        Used by the somatic appraiser so bar changes are visible in the same
        perceive cycle rather than waiting for the next daemon heartbeat.  The
        event is marked ``_applied`` so ``tick_bars`` won't double-count it.
        """
        self.bars.apply_event(event)
        self.bars._clamp_all()
        event["_applied"] = True
        self._recent_events.append(event)
        if len(self._recent_events) > self._max_events:
            self._recent_events = self._recent_events[-self._max_events:]
        self.flush_body_md()

    async def tick_bars(self, dt_hours: float) -> None:
        """Advance bar decay, apply pending events, detect impulses/conflicts."""
        for ev in self._recent_events:
            if not ev.get("_applied"):
                self.bars.apply_event(ev)
                ev["_applied"] = True
        self.bars.tick(dt_hours)
        self.flush_body_md()
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
        self.flush_body_md()

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
        self.flush_body_md()

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

    async def appraise_and_apply(
        self,
        client: Any,
        model: str,
        *,
        text: str,
        person_name: str,
        relationship_blurb: str = "",
        num_ctx: int | None = None,
    ) -> dict[str, Any]:
        """Run somatic appraisal on an incoming message and apply bar effects immediately."""
        if not self._appraisal_enabled or not text:
            return {"bar_effects": {}, "tags": [], "felt": ""}
        result = await self.appraiser.appraise_input(
            client,
            model,
            text=text,
            person_name=person_name,
            bar_names=self.bars.ordered_names,
            bars_pct=self.bars.snapshot_pct(),
            relationship_blurb=relationship_blurb,
            num_ctx=num_ctx,
        )
        if result["bar_effects"]:
            event = {
                "type": "appraisal",
                "subtype": "input",
                "from": person_name,
                "bar_effects": result["bar_effects"],
                "tags": result["tags"],
                "felt": result["felt"],
            }
            self.apply_immediate(event)
            log.info(
                "somatic_appraisal_applied",
                person=person_name,
                effects=result["bar_effects"],
                tags=result["tags"],
                felt=result["felt"],
            )
        return result

    async def appraise_interaction_and_apply(
        self,
        client: Any,
        model: str,
        *,
        input_text: str,
        reply_text: str,
        person_name: str,
        num_ctx: int | None = None,
    ) -> dict[str, Any]:
        """Run somatic appraisal on a completed exchange and apply bar effects."""
        if not self._appraisal_enabled or not reply_text:
            return {"bar_effects": {}, "tags": [], "felt": ""}
        result = await self.appraiser.appraise_interaction(
            client,
            model,
            input_text=input_text,
            reply_text=reply_text,
            person_name=person_name,
            bar_names=self.bars.ordered_names,
            bars_pct=self.bars.snapshot_pct(),
            num_ctx=num_ctx,
        )
        if result["bar_effects"]:
            event = {
                "type": "appraisal",
                "subtype": "interaction",
                "from": person_name,
                "bar_effects": result["bar_effects"],
                "tags": result["tags"],
                "felt": result["felt"],
            }
            self.apply_immediate(event)
            log.info(
                "somatic_interaction_appraisal_applied",
                person=person_name,
                effects=result["bar_effects"],
                tags=result["tags"],
                felt=result["felt"],
            )
        return result

    def render_body(self) -> str:
        """Assemble the body markdown the agent reads.

        This is the agent's sole window into its own internal state.
        The agent cannot edit ``body.md`` — only soma subsystems write it.
        """
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

    def flush_body_md(self) -> None:
        """Write the current body state to ``body.md`` in the soma directory.

        This is the canonical read-only file the main agent consumes.
        Only soma subsystems (bars, affects, noise, appraisal) call this —
        the agent never writes to it.
        """
        try:
            md = self.render_body()
            body_path = self._soma_dir / "body.md"
            body_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = body_path.with_suffix(".tmp")
            tmp.write_text(md, encoding="utf-8")
            tmp.replace(body_path)
        except Exception as e:
            log.debug("body_md_flush_failed", error=str(e))

    def reset_baseline(self) -> None:
        """Reset bars to YAML initials, clear affects/events/noise, rewrite ``body.md`` and ``soma-state.json``.

        Does not touch memory DB except via ``save_state_db`` on the entity.
        Restart a long-running worker after calling so in-memory tonic reloads.
        """
        self.bars.reset_to_initial()
        self._current_affects = []
        self.affects._previous_affects = []
        self.affects._last_tick = 0.0
        self.noise._fragments.clear()
        self.noise._last_tick = 0.0
        self._recent_events.clear()
        self.flush_body_md()
        self.save_state()
        log.info(
            "soma_reset_baseline",
            bars={k: round(v, 1) for k, v in self.bars._values.items()},
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

    def _snapshot_all(self) -> dict[str, Any]:
        """Build a unified snapshot of all three soma layers."""
        return {
            "values": dict(self.bars._values),
            "history": list(self.bars._history),
            "ordered_names": list(self.bars._ordered_names),
            "saved_at": time.time(),
            "affects": [dict(a) for a in self._current_affects],
            "noise_fragments": list(self.noise.current_fragments()),
        }

    def _restore_from_snapshot(self, data: dict[str, Any]) -> bool:
        """Restore all three soma layers from a unified snapshot."""
        saved_names = data.get("ordered_names", [])
        if saved_names != self.bars._ordered_names:
            log.info("soma_bar_names_changed", saved=saved_names, current=self.bars._ordered_names)
            return False
        self.bars._values = {k: float(data["values"][k]) for k in self.bars._ordered_names}
        self.bars._history.clear()
        for snap in data.get("history", []):
            self.bars._history.append({k: float(snap.get(k, 0)) for k in self.bars._ordered_names})
        if not self.bars._history:
            self.bars._history.append(dict(self.bars._values))
        saved_at = data.get("saved_at")
        if saved_at is not None:
            self.bars._apply_offline_decay(float(saved_at))

        saved_affects = data.get("affects")
        if isinstance(saved_affects, list) and saved_affects:
            self._current_affects = saved_affects
            self.affects._previous_affects = saved_affects
            log.info("soma_affects_restored", count=len(saved_affects))

        saved_noise = data.get("noise_fragments")
        if isinstance(saved_noise, list) and saved_noise:
            self.noise._fragments.clear()
            # Re-sanitize on restore — old fragments may predate current
            # cleanup regexes and contain leaked model markup.
            for frag in saved_noise:
                if not isinstance(frag, str):
                    continue
                cleaned = _split_noise_fragments(frag)
                for c in cleaned:
                    if not _fragment_is_duplicate(c, self.noise._fragments):
                        self.noise._fragments.append(c)
            log.info("soma_noise_restored", count=len(self.noise._fragments))

        return True

    def save_state(self) -> None:
        """Save full soma state to filesystem (fallback for local/SQLite deployments)."""
        state_path = self._soma_dir / "soma-state.json"
        try:
            data = self._snapshot_all()
            state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
            tmp.replace(state_path)
        except Exception as e:
            log.warning("soma_save_state_failed", error=str(e))

    def restore_state(self) -> bool:
        """Restore full soma state from filesystem (fallback for local/SQLite deployments)."""
        state_path = self._soma_dir / "soma-state.json"
        legacy_path = self._soma_dir / "soma-bar-state.json"
        target = state_path if state_path.is_file() else legacy_path
        if not target.is_file():
            return False
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
            ok = self._restore_from_snapshot(data)
            if ok:
                log.info("soma_state_restored", path=str(target.name),
                         values={k: round(v, 1) for k, v in self.bars._values.items()})
                self.flush_body_md()
            return ok
        except Exception:
            log.warning("soma_state_restore_failed", exc_info=True)
            return False

    async def save_state_db(self, conn: Any) -> None:
        """Persist full soma state to entity_state table (Postgres-durable)."""
        data = self._snapshot_all()
        payload = json.dumps(data, separators=(",", ":"))
        try:
            await conn.execute(
                "INSERT OR REPLACE INTO entity_state (key, value) VALUES (?, ?)",
                ("soma_state_v2", payload),
            )
            await conn.commit()
        except Exception as e:
            log.warning("soma_save_state_db_failed", error=str(e))

    async def restore_state_db(self, conn: Any) -> bool:
        """Restore full soma state from entity_state table."""
        try:
            cur = await conn.execute(
                "SELECT value FROM entity_state WHERE key = ?",
                ("soma_state_v2",),
            )
            row = await cur.fetchone()
            if not row:
                cur = await conn.execute(
                    "SELECT value FROM entity_state WHERE key = ?",
                    ("soma_bar_state_v1",),
                )
                row = await cur.fetchone()
            if not row:
                return False
            data = json.loads(row[0])
            ok = self._restore_from_snapshot(data)
            if ok:
                log.info("soma_state_restored_db",
                         values={k: round(v, 1) for k, v in self.bars._values.items()})
            return ok
        except Exception:
            log.warning("soma_state_restore_db_failed", exc_info=True)
            return False

    def drain_recent_events(self) -> list[dict[str, Any]]:
        """Return and clear recent events (for periodic cleanup)."""
        events = list(self._recent_events)
        self._recent_events.clear()
        return events
