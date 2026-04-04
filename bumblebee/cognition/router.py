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


def _identity_or_self_question(low: str) -> bool:
    """Short questions with ? were reflex-routed and hit reflex_max_tokens — bad for name/self."""
    phrases = (
        "who are you",
        "what are you",
        "what're you",
        "whatre you",
        "your name",
        "what's your name",
        "whats your name",
        "what is your name",
        "who am i talking",
        "what should i call you",
        "do you have a name",
        "tell me about yourself",
        "what do you call yourself",
        "are you an ai",
        "are you a bot",
        "are you human",
        "are you real",
        "are you a person",
    )
    return any(p in low for p in phrases)


def _short_casual_vibe(low: str, char_len: int) -> bool:
    """Keep small talk on reflex; classifier often wrongly picks DELIBERATE for slang."""
    if char_len > 120:
        return False
    markers = (
        "wagwan",
        "wag1",
        "waddup",
        "whaddup",
        "sup ",
        " sup",
        " nm",
        "nm ",
        "nuthin",
        "nothin much",
        "nothing much",
        "chillin",
        "chilling",
        "just vibing",
        "yo ",
        " aye",
        "safe ",
        "innit",
        "lol",
        "lmao",
        "nah ",
        " naw",
    )
    return any(m in low for m in markers)


def _tiny_non_identity_reaction(t: str, low: str) -> bool:
    """Short interjections like ``WHAT?`` — keep reflex so the classifier can't go cosmic."""
    if len(t) > 18 or "?" not in t:
        return False
    return not _identity_or_self_question(low)


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
        if _identity_or_self_question(low):
            return "deliberate"
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
        """Use reflex model to classify; return (route, uncertainty 0-1)."""
        model = self.entity.cognition.reflex_model
        messages = [
            {
                "role": "system",
                "content": (
                    "Reply with exactly one word: REFLEX or DELIBERATE. "
                    "REFLEX for greetings, thanks, very short chat, slang (wagwan, sup, nm), simple facts. "
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
        t = inp.text.strip()
        low = t.lower()
        if kind == "reflex" and (
            _short_casual_vibe(low, len(t)) or _tiny_non_identity_reaction(t, low)
        ):
            context.reflex_hint = kind
            return kind, context
        if kind == "reflex":
            rk, unc = await self.classify_with_reflex(inp)
            thresh = self.entity.harness.cognition.escalation_threshold
            if rk == "deliberate" or unc > 1.0 - thresh:
                kind = "deliberate"
        context.reflex_hint = kind
        return kind, context
