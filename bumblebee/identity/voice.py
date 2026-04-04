"""Expression metadata: typing delay, tone; light drift detection (no full rewrite)."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from bumblebee.config import EntityConfig
from bumblebee.models import EmotionalState


_LEADING_PAREN_BLOCK = re.compile(
    r"^\s*\([^)]{0,500}\)\s*(\n\s*)*",
    re.MULTILINE,
)
_LINE_ONLY_PAREN = re.compile(r"^\s*\([^)]{0,500}\)\s*$")
_LINE_ONLY_ASTERISK_ACTION = re.compile(r"^\s*\*[^*]{1,200}\*\s*$")


@dataclass
class ExpressionMeta:
    typing_delay_seconds: float
    tone_tag: str
    chunk_pause: float


def strip_stage_directions(text: str) -> str:
    """
    Remove roleplay narration the model should not send on text platforms:
    leading ``(Chuckles)``, whole lines that are only parentheses, lone ``*actions*``.
    """
    t = (text or "").strip()
    if not t:
        return t
    while True:
        m = _LEADING_PAREN_BLOCK.match(t)
        if not m:
            break
        t = t[m.end() :].lstrip()
    out_lines: list[str] = []
    for line in t.splitlines():
        s = line.strip()
        if not s:
            if out_lines and out_lines[-1] != "":
                out_lines.append("")
            continue
        if _LINE_ONLY_PAREN.match(line) or _LINE_ONLY_ASTERISK_ACTION.match(line):
            continue
        out_lines.append(line.rstrip())
    t = "\n".join(out_lines).strip()
    while True:
        m = _LEADING_PAREN_BLOCK.match(t)
        if not m:
            break
        t = t[m.end() :].lstrip()
    return t


class VoiceController:
    def __init__(self, entity: EntityConfig) -> None:
        self.entity = entity

    def sanitize_reply(self, text: str) -> str:
        return strip_stage_directions(text)

    def meta_for_response(
        self,
        emotional_state: EmotionalState,
        response_length: int,
        platform: str = "cli",
    ) -> ExpressionMeta:
        h = self.entity.harness.presence
        base_cps = h.typing_speed_base
        var = h.typing_speed_variance
        mood = emotional_state.primary.value
        if mood in ("excited", "amused"):
            cps = base_cps * (1.2 + random.uniform(-var, var))
        elif mood in ("melancholy", "withdrawn", "anxious"):
            cps = base_cps * (0.65 + random.uniform(-var, var))
        else:
            cps = base_cps * (1.0 + random.uniform(-var, var))
        cps = max(5.0, cps)
        delay = response_length / cps
        tone = mood if platform != "telegram" else f"measured_{mood}"
        return ExpressionMeta(
            typing_delay_seconds=min(12.0, max(0.3, delay)),
            tone_tag=tone,
            chunk_pause=h.chunk_delay,
        )
