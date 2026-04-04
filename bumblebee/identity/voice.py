"""Expression metadata: typing delay, tone; light drift detection (no full rewrite)."""

from __future__ import annotations

from dataclasses import dataclass
import random

from bumblebee.config import EntityConfig
from bumblebee.models import EmotionalState


@dataclass
class ExpressionMeta:
    typing_delay_seconds: float
    tone_tag: str
    chunk_pause: float


class VoiceController:
    def __init__(self, entity: EntityConfig) -> None:
        self.entity = entity

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
