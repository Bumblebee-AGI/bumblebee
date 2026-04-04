"""Classify reflex vs deliberate; assemble context package."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from bumblebee.config import EntityConfig
from bumblebee.models import EmotionCategory, EmotionalState, Input
from bumblebee.utils.ollama_client import OllamaClient


RouteKind = Literal["reflex", "deliberate"]


@dataclass
class ContextPackage:
    emotional_state: EmotionalState
    memory_snippets: list[str] = field(default_factory=list)
    relationship_blurb: str = ""
    drives_summary: str = ""
    inner_summary: str = ""
    reflex_hint: str = ""


class CognitionRouter:
    def __init__(self, entity: EntityConfig, client: OllamaClient) -> None:
        self.entity = entity
        self.client = client

    def heuristic_route(self, inp: Input, emotional: EmotionalState) -> RouteKind:
        t = inp.text.strip()
        low = t.lower()
        if len(t) < 4:
            return "reflex"
        if low in ("hi", "hello", "hey", "thanks", "thank you", "ok", "okay", "bye"):
            return "reflex"
        if "?" in t and len(t) < 80 and emotional.intensity < 0.6:
            return "reflex"
        if any(
            w in low
            for w in (
                "why",
                "meaning",
                "feel",
                "afraid",
                "love",
                "death",
                "exist",
                "philosophy",
            )
        ):
            return "deliberate"
        if emotional.primary in (
            EmotionCategory.MELANCHOLY,
            EmotionCategory.ANXIOUS,
            EmotionCategory.AFFECTIONATE,
        ):
            return "deliberate"
        return "deliberate" if len(t) > 200 else "reflex"

    async def classify_with_reflex(self, inp: Input) -> tuple[RouteKind, float]:
        """Use E4B to classify; return (route, uncertainty 0-1)."""
        model = self.entity.cognition.reflex_model
        messages = [
            {
                "role": "system",
                "content": (
                    "Reply with exactly one word: REFLEX or DELIBERATE. "
                    "REFLEX for greetings, thanks, very short chat, simple facts. "
                    "DELIBERATE for emotional depth, complex reasoning, creative work."
                ),
            },
            {"role": "user", "content": inp.text[:1500]},
        ]
        try:
            res = await self.client.chat_completion(
                model,
                messages,
                temperature=0.2,
                max_tokens=8,
                think=False,
            )
            word = (res.content or "").strip().upper()
            if "DELIBERATE" in word:
                return "deliberate", 0.2
            if "REFLEX" in word:
                return "reflex", 0.2
        except Exception:
            pass
        return "deliberate", 0.6

    async def route(
        self,
        inp: Input,
        emotional: EmotionalState,
        context: ContextPackage,
    ) -> tuple[RouteKind, ContextPackage]:
        kind = self.heuristic_route(inp, emotional)
        if kind == "reflex":
            rk, unc = await self.classify_with_reflex(inp)
            thresh = self.entity.harness.cognition.escalation_threshold
            if rk == "deliberate" or unc > 1.0 - thresh:
                kind = "deliberate"
        context.reflex_hint = kind
        return kind, context
