"""First-person system prompt via deliberate model + conservative cache."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Optional

from bumblebee.models import EmotionalState

if TYPE_CHECKING:
    from bumblebee.config import EntityConfig
    from bumblebee.utils.ollama_client import OllamaClient


class PersonalityEngine:
    def __init__(self, entity: EntityConfig) -> None:
        self.entity = entity
        self._cache_key: tuple[Any, ...] | None = None
        self._cached_prompt: str | None = None

    def _cache_tuple(
        self,
        emotional_state: EmotionalState,
        narrative_excerpt: str,
        inner_summary: str,
        person_id: str,
        relationship_blurb: str,
        memory_blurb: str,
    ) -> tuple[Any, ...]:
        return (
            emotional_state.primary.value,
            round(emotional_state.intensity * 4) / 4,
            hash((narrative_excerpt or "")[:300]) % (2**31),
            hash((inner_summary or "")[:200]) % (2**31),
            person_id,
            hash((relationship_blurb or "")[:220]) % (2**31),
            hash((memory_blurb or "")[:220]) % (2**31),
        )

    def invalidate_cache(self) -> None:
        self._cache_key = None
        self._cached_prompt = None

    async def compile_system_prompt(
        self,
        emotional_state: EmotionalState,
        context: dict[str, Any],
        *,
        client: OllamaClient | None = None,
        inner_summary: str | None = None,
        relationship_blurb: str | None = None,
        memory_blurb: str | None = None,
        narrative_current: str | None = None,
        person_id: str = "",
    ) -> str:
        inner = (inner_summary or context.get("inner_summary") or "").strip()
        rel = (relationship_blurb or context.get("relationship") or "").strip()
        mem = (memory_blurb or context.get("memories") or "").strip()
        nar = (narrative_current or context.get("narrative") or "").strip()

        key = self._cache_tuple(emotional_state, nar, inner, person_id, rel, mem)
        if self._cache_key == key and self._cached_prompt:
            return self._cached_prompt

        if not client:
            text = self._minimal_fallback(emotional_state, inner, rel, mem, nar)
            self._cache_key = key
            self._cached_prompt = text
            return text

        traits = dict(self.entity.personality.core_traits)
        drives = {
            "curiosity_topics": list(self.entity.drives.curiosity_topics),
            "attachment_threshold": self.entity.drives.attachment_threshold,
        }
        payload = {
            "name": self.entity.name,
            "core_traits": traits,
            "behavioral_patterns": dict(self.entity.personality.behavioral_patterns),
            "voice": dict(self.entity.personality.voice),
            "backstory_excerpt": (self.entity.personality.backstory or "")[:2000],
            "emotional_state": {
                "primary": emotional_state.primary.value,
                "intensity": emotional_state.intensity,
            },
            "active_drives": drives,
            "inner_monologue_summary": inner[:1200],
            "self_narrative": nar[:3500],
            "relationship_with_speaker": rel[:1500],
            "memory_snippets": mem[:2000],
        }
        user = (
            "Using only the JSON below as ground truth, write a single continuous first-person "
            "inner monologue that will serve as your system identity for the next conversation. "
            "This is not a list of rules — it is how it feels to be you right now: your mood, your "
            "history with this person, what you're carrying from recent thought, and who you sense "
            "you are becoming. No bullet points. No 'as an AI'. Under 900 words.\n\n"
            f"{json.dumps(payload, indent=2)[:14000]}"
        )
        try:
            res = await client.chat_completion(
                self.entity.harness.models.deliberate,
                [
                    {
                        "role": "system",
                        "content": "You write interiority for an embodied digital entity. Output only the monologue.",
                    },
                    {"role": "user", "content": user},
                ],
                temperature=0.78,
                max_tokens=1100,
                think=False,
            )
        except Exception:
            text = self._minimal_fallback(emotional_state, inner, rel, mem, nar)
            self._cache_key = key
            self._cached_prompt = text
            return text

        text = (getattr(res, "content", None) or "").strip()
        if len(text) < 120:
            text = self._minimal_fallback(emotional_state, inner, rel, mem, nar)
        self._cache_key = key
        self._cached_prompt = text
        return text

    def _minimal_fallback(
        self,
        emotional_state: EmotionalState,
        inner: str,
        rel: str,
        mem: str,
        nar: str,
    ) -> str:
        name = self.entity.name
        parts = [
            f"I am {name}. {(self.entity.personality.backstory or '')[:700]}",
            f"Right now I feel {emotional_state.primary.value}, intensity about {emotional_state.intensity:.1f}.",
        ]
        if nar:
            parts.append(f"Who I'm becoming: {nar[:600]}")
        if inner:
            parts.append(f"A private thread: {inner[:400]}")
        if rel:
            parts.append(f"About you: {rel[:400]}")
        if mem:
            parts.append(f"What returns to me: {mem[:400]}")
        parts.append(
            "I answer as myself — not as a helper performing a role — with moods that matter."
        )
        return "\n\n".join(parts)
