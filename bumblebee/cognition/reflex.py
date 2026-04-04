"""Fast E4B path — minimal context."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from bumblebee.config import EntityConfig
from bumblebee.cognition import gemma
from bumblebee.cognition.router import ContextPackage
from bumblebee.models import Input
from bumblebee.utils.ollama_client import ChatCompletionResult, OllamaClient


class ReflexCognition:
    def __init__(self, entity: EntityConfig, client: OllamaClient) -> None:
        self.entity = entity
        self.client = client

    async def respond(
        self,
        inp: Input,
        system_prompt: str,
        recent_messages: list[dict[str, str]],
        context: ContextPackage,
    ) -> tuple[ChatCompletionResult, str]:
        h = self.entity.harness.cognition
        msgs: list[dict] = [{"role": "system", "content": system_prompt[:6000]}]
        for m in recent_messages[-4:]:
            msgs.append(m)
        msgs.append({"role": "user", "content": inp.text[:4000]})
        res = await self.client.chat_completion(
            self.entity.cognition.reflex_model,
            msgs,
            temperature=min(0.9, h.temperature),
            max_tokens=h.reflex_max_tokens,
            think=False,
        )
        mood = "neutral"
        low = (res.content or "").lower()
        if any(x in low for x in ("ha", "lol", "funny")):
            mood = "positive"
        elif any(x in low for x in ("sorry", "unfortunately", "can't")):
            mood = "slight_negative"
        return res, mood

    async def respond_stream(
        self,
        inp: Input,
        system_prompt: str,
        recent_messages: list[dict[str, str]],
        context: ContextPackage,
        on_delta: Callable[[str], Awaitable[None]],
    ) -> tuple[ChatCompletionResult, str]:
        h = self.entity.harness.cognition
        msgs: list[dict] = [{"role": "system", "content": system_prompt[:6000]}]
        for m in recent_messages[-4:]:
            msgs.append(m)
        msgs.append({"role": "user", "content": inp.text[:4000]})
        gen = await self.client.chat_completion(
            self.entity.cognition.reflex_model,
            msgs,
            temperature=min(0.9, h.temperature),
            max_tokens=h.reflex_max_tokens,
            think=False,
            stream=True,
        )
        buf = ""
        prev_visible = ""
        async for delta in gen:  # type: ignore[union-attr]
            buf += delta
            parsed = gemma.parse_assistant_output(buf)
            vis = parsed.visible_user_text
            if len(vis) > len(prev_visible):
                await on_delta(vis[len(prev_visible) :])
                prev_visible = vis
        final = gemma.parse_assistant_output(buf)
        visible = (final.visible_user_text or "").strip() or "…"
        res = ChatCompletionResult(
            content=visible,
            thinking=final.thinking,
            raw_assistant_text=buf,
        )
        mood = "neutral"
        low = visible.lower()
        if any(x in low for x in ("ha", "lol", "funny")):
            mood = "positive"
        elif any(x in low for x in ("sorry", "unfortunately", "can't")):
            mood = "slight_negative"
        return res, mood
