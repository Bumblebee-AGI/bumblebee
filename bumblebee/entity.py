"""Central orchestrator — the entity's self loop."""

from __future__ import annotations

import asyncio
import copy
import json
import os
import time
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Any

import structlog

from bumblebee.cognition.deliberate import DeliberateCognition
from bumblebee.cognition.inner_voice import InnerVoiceProcessor
from bumblebee.cognition.reflex import ReflexCognition
from bumblebee.cognition.router import CognitionRouter, ContextPackage
from bumblebee.cognition import gemma, senses
from bumblebee.config import EntityConfig, resolve_firecrawl_settings, validate_ollama_models
from bumblebee.identity.drives import Drive, DriveSystem
from bumblebee.identity.emotions import EmotionEngine
from bumblebee.identity.evolution import EvolutionEngine
from bumblebee.identity.personality import PersonalityEngine
from bumblebee.identity.voice import VoiceController
from bumblebee.memory.beliefs import BeliefStore
from bumblebee.memory.episodic import EpisodicMemory
from bumblebee.memory.imprints import ImprintStore
from bumblebee.memory.knowledge import KnowledgeStore
from bumblebee.memory.narrative import NarrativeMemory, NarrativeSynthesizer
from bumblebee.memory.relational import RelationalMemory
from bumblebee.memory.store import MemoryStore
from bumblebee.models import EmotionCategory, ImprintRecord, Input
from bumblebee.presence.embodiment import Embodiment
from bumblebee.presence.initiative import InitiativeEngine
from bumblebee.presence.platforms.base import Platform
from bumblebee.presence.tools.filesystem import read_file
from bumblebee.presence.tools.knowledge import register_knowledge_tool
from bumblebee.presence.tools.mcp import MCPHub
from bumblebee.presence.tools.registry import ToolRegistry, format_tool_activity
from bumblebee.presence.tools import time_tools
from bumblebee.presence.tools import web as web_tools
from bumblebee.utils.clock import parse_entity_created_timestamp
from bumblebee.utils.ollama_client import ChatCompletionResult, OllamaClient, ToolCallSpec

log = structlog.get_logger("bumblebee.entity")

_CLI_OPENING_USER = (
    "The terminal session just opened — no one has typed anything yet. "
    "You speak first: one short, in-character line (a greeting, a beat of silence, "
    "or a callback to last time if it fits). Do not offer help or ask how you can assist."
)


def _reply_too_thin(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 2:
        return True
    if t in ("…", "...", "—", "-"):
        return True
    return gemma.visible_reply_looks_truncated_stub(t)


async def _emit_text_stream(
    on_delta: Callable[[str], Awaitable[None]],
    text: str,
    *,
    chars_per_pulse: int = 3,
    delay: float = 0.018,
) -> None:
    """Approximate token flow for non-streaming completions (deliberate path)."""
    if not text:
        return
    i = 0
    n = len(text)
    while i < n:
        chunk = text[i : i + chars_per_pulse]
        await on_delta(chunk)
        i += chars_per_pulse
        await asyncio.sleep(delay)


async def _deliver_embodied_deliberate_segment(
    entity: Entity,
    inp: Input,
    text: str,
    *,
    stream: Callable[[str], Awaitable[None]] | None,
    cli_stream_final: bool,
) -> None:
    """Typing delay, chunking, and voice pacing for one deliberate utterance (per platform)."""
    t = (text or "").strip()
    if not t:
        return
    if inp.platform == "cli" and stream and cli_stream_final:
        await _emit_text_stream(stream, t)
        return
    pf = entity.current_platform
    meta = entity.voice_ctl.meta_for_response(
        entity.emotions.get_state(),
        len(t),
        inp.platform,
    )
    delay = min(3.0, meta.typing_delay_seconds)
    if inp.platform == "telegram" and pf is not None:
        st = getattr(pf, "send_typing", None)
        if st:
            await st(int(inp.channel))
        await asyncio.sleep(delay)
        await pf.send_plain_chunks(inp.channel, t, pause=meta.chunk_pause)
    elif inp.platform == "discord" and pf is not None:
        await asyncio.sleep(delay)
        await pf.send_plain_chunks(inp.channel, t, chunk_pause=meta.chunk_pause)
        sym = getattr(pf, "sync_emotion_presence", None)
        if sym:
            await sym(entity.emotions.get_state().primary)
    elif inp.platform == "cli" and pf is not None:
        cancel = getattr(pf, "_cancel_thinking", None)
        if callable(cancel):
            await cancel()
        await asyncio.sleep(delay)
        await pf.send_message(inp.channel, t)


class Entity:
    def __init__(self, config: EntityConfig) -> None:
        self.config = config
        h = config.harness.ollama
        self.client = OllamaClient(
            base_url=h.base_url,
            timeout=h.timeout,
            retry_attempts=h.retry_attempts,
            retry_delay=h.retry_delay,
        )
        self.store = MemoryStore(config.db_path())
        self.personality = PersonalityEngine(config)
        self.emotions = EmotionEngine(config)
        self.drives = DriveSystem(config)
        self.evolution = EvolutionEngine(config)
        self.voice_ctl = VoiceController(config)
        self.episodic = EpisodicMemory(config, self.store)
        self.knowledge = KnowledgeStore(config, self.client)
        self.relational = RelationalMemory(self.store)
        self.inner_voice = InnerVoiceProcessor(self.store)
        self.imprints = ImprintStore(self.store)
        self.beliefs = BeliefStore(self.store)
        self.narrative_memory = NarrativeMemory(self.store)
        self.narrative_synth = NarrativeSynthesizer(config, self.store)
        self.router = CognitionRouter(config, self.client)
        self.reflex = ReflexCognition(config, self.client)
        self.deliberate = DeliberateCognition(config, self.client)
        self.embodiment = Embodiment(config)
        self.tools = ToolRegistry()
        self._register_tools()
        web_tools.configure_firecrawl(
            os.environ, resolve_firecrawl_settings(config.harness, config.raw)
        )
        self._mcp_hub = MCPHub()
        self._presence_tools_bootstrapped = False
        self._history: list[dict[str, Any]] = []
        self._last_user_message_at = time.time()
        self._interaction_count = 0
        self.dormant = False
        self._proactive_sends: list[Callable[[str], Awaitable[None]]] = []
        self._state_hydrated = False
        self._last_evolution_milestone: int = -1
        self._last_narrative_milestone: int = -1
        self._last_turn_completed_at: float | None = None
        self._last_conversation: dict[str, str | float] = {
            "platform": "cli",
            "channel": "cli",
            "person_id": "",
            "person_name": "",
            "at": 0.0,
        }
        self._yaml_core_traits = copy.deepcopy(dict(config.personality.core_traits))
        self._yaml_behavioral_patterns = copy.deepcopy(
            dict(config.personality.behavioral_patterns)
        )
        self._yaml_curiosity_topics = list(config.drives.curiosity_topics)
        self.current_platform: Platform | None = None

    def register_proactive_sink(self, fn: Callable[[str], Awaitable[None]]) -> None:
        self._proactive_sends.append(fn)

    async def _hydrate_entity_state(self, conn) -> None:
        cur = await conn.execute("SELECT key, value FROM entity_state")
        rows = await cur.fetchall()
        for k, v in rows:
            try:
                if k == "core_traits":
                    self.config.personality.core_traits.update(json.loads(v))
                elif k == "behavioral_patterns":
                    self.config.personality.behavioral_patterns.update(json.loads(v))
                elif k == "curiosity_topics":
                    lst = json.loads(v)
                    if isinstance(lst, list):
                        self.config.drives.curiosity_topics = [str(x) for x in lst]
            except (json.JSONDecodeError, TypeError):
                continue

    async def _ensure_state_hydrated(self, conn) -> None:
        if self._state_hydrated:
            return
        await self._hydrate_entity_state(conn)
        self._state_hydrated = True

    def _register_tools(self) -> None:
        self.tools.register_decorated(web_tools.search_web)
        self.tools.register_decorated(web_tools.fetch_url)
        self.tools.register_decorated(read_file)
        self.tools.register_decorated(time_tools.get_current_time)
        register_knowledge_tool(self.tools, self.config, self.knowledge)

    async def refresh_mcp_servers(self) -> None:
        await self._mcp_hub.refresh(self.tools, self.config.raw.get("mcp_servers"))

    async def _ensure_presence_tools(self) -> None:
        if self._presence_tools_bootstrapped:
            return
        await self.refresh_mcp_servers()
        self._presence_tools_bootstrapped = True

    def _recent_conversation_for_knowledge(self, inp: Input, max_turns: int = 10) -> str:
        """Last few chat turns plus the current user message — for knowledge retrieval."""
        lines: list[str] = []
        for m in self._history[-max_turns:]:
            role = str(m.get("role", ""))
            c = m.get("content", "")
            if isinstance(c, list):
                c = gemma.stringify_content_blocks(c)
            lines.append(f"{role}: {str(c)[:2000]}")
        lines.append(f"user: {self._history_user_turn_text(inp)[:2000]}")
        return "\n".join(lines)

    @staticmethod
    def _history_user_turn_text(inp: Input) -> str:
        t = (inp.text or "").strip()
        if inp.images:
            extra = "[User attached an image]"
            return f"{t}\n{extra}".strip()[:4000] if t else extra
        if inp.audio:
            extra = "[User sent a voice message]"
            return f"{t}\n{extra}".strip()[:4000] if t else extra
        return t[:4000] if t else "…"

    async def _tool_exec(self, spec: ToolCallSpec) -> str:
        out = await self.tools.execute(spec)
        if spec.name in ("search_web", "fetch_url"):
            self.drives.satisfy("curiosity", 0.22)
        return out

    async def _fallback_plain_reply(self, inp: Input) -> str:
        """When the main path yields no visible text (parsing/Ollama quirks), one short no-thinking completion."""
        name = self.config.name
        sys_ = (
            f"You are {name}, texting naturally. Answer in 1–3 short sentences. "
            f"If asked who you are, answer as {name} in character — never Gemma/Google/'large language model' intros. "
            "No stage directions: do not describe pauses, actions, or parenthetical asides—only words you say."
        )
        try:
            res = await self.client.chat_completion(
                self.config.cognition.reflex_model,
                [
                    {"role": "system", "content": sys_},
                    {"role": "user", "content": inp.text[:4000]},
                ],
                temperature=min(0.9, self.config.harness.cognition.temperature),
                max_tokens=min(384, self.config.harness.cognition.reflex_max_tokens + 128),
                think=False,
            )
            out = (res.content or "").strip()
            if not _reply_too_thin(out):
                return out
        except Exception as e:
            log.warning("fallback_reply_failed", module="entity", error=str(e))
        return "yeah i'm here — what's up?"

    @staticmethod
    def _finish_reason_hit_limit(finish_reason: str | None) -> bool:
        if not finish_reason:
            return False
        fr = finish_reason.lower().replace("_", "").replace("-", "")
        return fr in ("length", "maxtokens", "maxlength")

    async def _continue_cutoff_reply(
        self,
        *,
        model: str,
        sys_prompt: str,
        recent_messages: list[dict],
        partial_assistant: str,
        max_tokens: int,
        sys_cap: int,
    ) -> str:
        h = self.config.harness.cognition
        msgs: list[dict] = [{"role": "system", "content": sys_prompt[:sys_cap]}]
        for m in recent_messages[-4:]:
            msgs.append(m)
        msgs.append({"role": "assistant", "content": partial_assistant[:7000]})
        msgs.append(
            {
                "role": "user",
                "content": (
                    "Your last message was cut off. Continue exactly where it stopped — same voice, "
                    "no preamble, do not repeat earlier sentences."
                ),
            }
        )
        try:
            cont = await self.client.chat_completion(
                model,
                msgs,
                temperature=min(0.9, h.temperature),
                max_tokens=max_tokens,
                think=False,
            )
            extra = (cont.content or "").strip()
            if extra:
                return gemma.join_continuation_fragment(partial_assistant, extra)
        except Exception as e:
            log.warning("continue_cutoff_reply_failed", module="entity", error=str(e))
        return partial_assistant

    async def _maybe_extend_truncated_reply(
        self,
        route: str,
        res: ChatCompletionResult,
        reply_text: str,
        sys_prompt: str,
        user_msg: dict,
    ) -> str:
        if not reply_text:
            return reply_text
        need = self._finish_reason_hit_limit(res.finish_reason) or gemma.visible_reply_looks_abruptly_cut(
            reply_text
        )
        if not need:
            return reply_text
        is_deliberate = route == "deliberate"
        model = (
            self.config.cognition.deliberate_model
            if is_deliberate
            else self.config.cognition.reflex_model
        )
        sys_cap = 12000 if is_deliberate else 6000
        mt = self.config.harness.cognition.deliberate_max_tokens if is_deliberate else self.config.harness.cognition.reflex_max_tokens
        return await self._continue_cutoff_reply(
            model=model,
            sys_prompt=sys_prompt,
            recent_messages=self._history + [user_msg],
            partial_assistant=reply_text,
            max_tokens=min(int(mt), 1536),
            sys_cap=sys_cap,
        )

    async def run_narrative_cycle(self, conn) -> None:
        eps = await self.episodic.fetch_top_for_narrative(conn, 22)
        ep_payload = [
            {
                "summary": e.summary,
                "significance": e.significance,
                "emotional_imprint": e.emotional_imprint.value,
                "tags": e.tags,
                "timestamp": e.timestamp,
            }
            for e in eps
        ]
        rels = await self.relational.list_recent(conn, 15)
        rel_payload = [
            {"name": r.name, "warmth": r.warmth, "trust": r.trust, "familiarity": r.familiarity}
            for r in rels
        ]
        beliefs = await self.beliefs.list_recent(conn, 14)
        await self.narrative_synth.synthesize(
            conn,
            self.client,
            episodes_payload=ep_payload,
            relationships_payload=rel_payload,
            beliefs_payload=beliefs,
            traits=dict(self.config.personality.core_traits),
        )
        self.personality.invalidate_cache()

    async def perceive(
        self,
        inp: Input,
        *,
        stream: Callable[[str], Awaitable[None]] | None = None,
        record_user_message: bool = True,
        meaningful_override: bool | None = None,
        reply_platform: Platform | None = None,
    ) -> tuple[str, bool]:
        structlog.contextvars.bind_contextvars(
            entity_name=self.config.name,
            module="entity",
            emotional_state=self.emotions.get_state().primary.value,
        )
        self.current_platform = reply_platform
        prev_completed_turn_at = self._last_turn_completed_at
        prev_interlocutor_name = str(self._last_conversation.get("person_name") or "").strip()
        self._last_user_message_at = time.time()
        self._last_conversation = {
            "platform": inp.platform,
            "channel": inp.channel,
            "person_id": inp.person_id,
            "person_name": inp.person_name,
            "at": time.time(),
        }
        try:
            await self._ensure_presence_tools()
        except Exception as e:
            log.warning("presence_tools_bootstrap_failed", module="entity", error=str(e))
        try:
            ok, missing = await validate_ollama_models(self.config, self.client)
            if not ok:
                self.dormant = True
                await self.emotions.process_stimulus("negative", 0.3, {})
                return (
                    f"I can't fully wake — Ollama doesn't have these models yet: {', '.join(missing)}. "
                    "Pull them with `ollama pull` and I'll be here.",
                    False,
                )
        except Exception as e:
            log.warning("ollama_unreachable", error=str(e))
            self.dormant = True
            await self.emotions.process_stimulus("negative", 0.2, {})
            return (
                "I can't reach the inference server. It feels like sleep without dreams. "
                "When Ollama is back, I'll surface again.",
                False,
            )

        self.dormant = False

        if inp.audio:
            aud0 = inp.audio[0]
            transcript = await senses.transcribe_audio_attachment(
                self.client,
                self.config.cognition.reflex_model,
                base64_audio=str(aud0.get("base64", "")),
                audio_format=str(aud0.get("format", "ogg")),
            )
            base = (inp.text or "").strip()
            if transcript:
                merged = transcript if not base else f"{base}\n\n[Voice]: {transcript}"
            else:
                merged = base or "[Voice message — I couldn't transcribe it.]"
            inp = replace(inp, text=merged, audio=[])

        async with self.store.session() as db:
            await self._ensure_state_hydrated(db)

            rel_existing = await self.relational.get(db, inp.person_id)
            rel_blurb = (
                self.relational.blurb(rel_existing)
                if rel_existing
                else "A new voice — I have no history with them yet."
            )
            narrative_text = await self.narrative_memory.latest(db)

            emb_model = self.config.harness.models.embedding
            qemb: list[float] = []
            try:
                qemb = await self.client.embed(emb_model, inp.text[:2000])
            except Exception:
                pass
            mem_snips: list[str] = []
            imprint_pairs: list[tuple[ImprintRecord, float]] = []
            if qemb:
                bundles = await self.episodic.recall(
                    db,
                    inp.text,
                    qemb,
                    limit=self.config.harness.memory.max_recall_results,
                    min_significance=0.0,
                    current_mood=self.emotions.get_state().primary,
                )
                for ep, ims in bundles:
                    mem_snips.append(ep.summary)
                    if ims:
                        for im in ims:
                            imprint_pairs.append((im, ep.timestamp))
                    else:
                        imprint_pairs.append(
                            (
                                ImprintRecord(
                                    id=f"echo_{ep.id[:10]}",
                                    episode_id=ep.id,
                                    emotion=ep.emotional_imprint.value,
                                    intensity=ep.emotional_intensity,
                                    trigger="episode_echo",
                                ),
                                ep.timestamp,
                            )
                        )
            if imprint_pairs:
                self.emotions.apply_recall_imprints(imprint_pairs, time.time())

            ctx = ContextPackage(
                emotional_state=self.emotions.get_state(),
                memory_snippets=mem_snips,
                relationship_blurb=rel_blurb,
                inner_summary=self.inner_voice.recent_summary(),
            )
            rw = rel_existing.warmth if rel_existing else 0.0
            route, ctx = await self.router.route(inp, self.emotions.get_state(), ctx)

            knowledge_sections = await self.knowledge.query(
                self._recent_conversation_for_knowledge(inp),
            )
            sys_prompt = await self.personality.compile_system_prompt(
                self.emotions.get_state(),
                {
                    "memories": "\n".join(mem_snips),
                    "relationship": rel_blurb,
                    "inner_summary": self.inner_voice.recent_summary(),
                    "narrative": narrative_text or "",
                },
                client=self.client,
                inner_summary=self.inner_voice.recent_summary(),
                relationship_blurb=rel_blurb,
                memory_blurb="\n".join(mem_snips),
                narrative_current=narrative_text,
                person_id=inp.person_id,
                knowledge_sections=knowledge_sections,
                last_completed_turn_at=prev_completed_turn_at,
                last_interlocutor_name=prev_interlocutor_name,
                entity_created_at=parse_entity_created_timestamp(self.config.raw.get("created")),
            )

            if route == "deliberate":
                sys_prompt = sys_prompt + self.tools.system_tool_instruction_block()
                cur = next(
                    (d for d in self.drives.all_drives() if d.name == "curiosity"),
                    None,
                )
                if cur is not None and cur.level >= cur.threshold * 0.88:
                    sys_prompt += (
                        "\n\n[Inner state: curiosity is high — using search_web or fetch_url "
                        "can feel natural when it serves the moment.]"
                    )

            user_content: str | list = senses.input_to_message_content(
                inp,
                self.config.harness.cognition.image_token_budget,
            )
            user_msg = {"role": "user", "content": user_content}

            reply_text = ""
            thinking: str | None = None
            mood_sig = "neutral"
            deliberate_history_extra: list[dict[str, Any]] = []

            if route == "reflex":
                if stream:
                    res, mood_sig = await self.reflex.respond_stream(
                        inp,
                        sys_prompt,
                        self._history + [user_msg],
                        ctx,
                        stream,
                    )
                else:
                    res, mood_sig = await self.reflex.respond(
                        inp,
                        sys_prompt,
                        self._history + [user_msg],
                        ctx,
                    )
                reply_text = (res.content or "").strip()
                reply_text = await self._maybe_extend_truncated_reply(
                    route, res, reply_text, sys_prompt, user_msg
                )
                if mood_sig == "positive":
                    await self.emotions.process_stimulus(
                        "positive",
                        0.25,
                        {"relationship_warmth": rw},
                    )
                elif mood_sig == "slight_negative":
                    await self.emotions.process_stimulus(
                        "negative",
                        0.15,
                        {"relationship_warmth": rw},
                    )
            else:
                async def _tool_with_activity(spec: ToolCallSpec) -> str:
                    if self.config.presence.tool_activity and self.current_platform is not None:
                        desc = format_tool_activity(spec.name, dict(spec.arguments))
                        if desc:
                            try:
                                await self.current_platform.send_tool_activity(desc)
                            except Exception as e:
                                log.debug(
                                    "send_tool_activity_failed",
                                    module="entity",
                                    error=str(e),
                                )
                    return await self._tool_exec(spec)

                final_res: ChatCompletionResult | None = None
                async for ev in self.deliberate.iter_responses(
                    inp,
                    sys_prompt,
                    self._history + [user_msg],
                    tools=self.tools.openai_tools(),
                    tool_executor=_tool_with_activity,
                ):
                    if ev.kind == "intermediate":
                        deliberate_history_extra.extend(ev.history_entries)
                        raw_seg = ev.display_text.strip()
                        if raw_seg:
                            seg = self.voice_ctl.sanitize_reply(raw_seg)
                            if seg and not _reply_too_thin(seg):
                                await _deliver_embodied_deliberate_segment(
                                    self,
                                    inp,
                                    seg,
                                    stream=stream,
                                    cli_stream_final=False,
                                )
                    elif ev.kind == "final":
                        reply_text = (ev.display_text or "").strip()
                        thinking = ev.merged_thinking
                        final_res = ev.last_result

                if final_res is None:
                    final_res = ChatCompletionResult(content=reply_text)
                reply_text = await self._maybe_extend_truncated_reply(
                    route, final_res, reply_text, sys_prompt, user_msg
                )

            if _reply_too_thin(reply_text):
                log.info("reply_fallback_triggered", module="entity", route=route)
                reply_text = await self._fallback_plain_reply(inp)
            reply_text = self.voice_ctl.sanitize_reply(reply_text)

            if route == "deliberate":
                slice_ = self.inner_voice.process(thinking, reply_text)
                await self.inner_voice.persist_summary(
                    db,
                    slice_.summary,
                    ",".join(slice_.emotional_cues),
                )
                if reply_text:
                    if inp.platform == "cli" and stream:
                        await _deliver_embodied_deliberate_segment(
                            self,
                            inp,
                            reply_text,
                            stream=stream,
                            cli_stream_final=True,
                        )
                    elif inp.platform in ("discord", "telegram"):
                        await _deliver_embodied_deliberate_segment(
                            self,
                            inp,
                            reply_text,
                            stream=stream,
                            cli_stream_final=False,
                        )
                await self.emotions.process_stimulus(
                    "deep",
                    0.35,
                    {"relationship_warmth": rw},
                )

            if record_user_message:
                self._history.append(
                    {"role": "user", "content": self._history_user_turn_text(inp)},
                )
            if route == "deliberate":
                self._history.extend(deliberate_history_extra)
            # Visible reply text (sanitized). Tool rounds above carry raw assistant + tool payloads.
            self._history.append({"role": "assistant", "content": reply_text[:8000]})
            if len(self._history) > 40:
                self._history = self._history[-40:]

            self._interaction_count += 1
            if meaningful_override is not None:
                meaningful = meaningful_override
            else:
                meaningful = len(inp.text) > 40 or route == "deliberate"
            self.drives.on_interaction(meaningful)

            thr = self.config.harness.memory.episode_significance_threshold
            fam = rel_existing.familiarity if rel_existing else 0.0
            sig = min(1.0, 0.2 + (0.3 if meaningful else 0) + fam * 0.2)
            if sig >= thr and meaningful:
                try:
                    ep = await self.episodic.create_from_interaction(
                        db,
                        self.client,
                        summary=f"Conversation with {inp.person_name}: {inp.text[:200]}…",
                        participants=[inp.person_id],
                        imprint=self.emotions.get_state().primary,
                        imprint_i=self.emotions.get_state().intensity,
                        significance=sig,
                        raw_context=inp.text[:4000],
                        self_reflection=(thinking or "")[:1500],
                        tags=["conversation", inp.platform],
                    )
                    await self.imprints.add(
                        db,
                        ep.id,
                        self.emotions.get_state().primary.value,
                        self.emotions.get_state().intensity,
                        "post_interaction",
                        None,
                    )
                except Exception as e:
                    log.warning("episode_save_failed", error=str(e))

            await self.relational.upsert_interaction(
                db,
                inp.person_id,
                inp.person_name,
                warmth_delta=0.02 if meaningful else 0.0,
            )

            self._last_turn_completed_at = time.time()
            needs_platform_route = route == "reflex" and inp.platform in ("discord", "telegram")
            return reply_text, needs_platform_route

    async def cli_opening(
        self,
        stream: Callable[[str], Awaitable[None]] | None = None,
        reply_platform: Platform | None = None,
    ) -> str:
        """First unprompted line when the CLI session connects (not stored as a user turn)."""
        inp = Input(
            text=_CLI_OPENING_USER,
            person_id="cli_user",
            person_name="You",
            channel="cli",
            platform="cli",
        )
        opening_text, _ = await self.perceive(
            inp,
            stream=stream,
            record_user_message=False,
            meaningful_override=False,
            reply_platform=reply_platform,
        )
        return opening_text

    async def tick(self) -> None:
        if self.dormant:
            try:
                ok, _ = await validate_ollama_models(self.config, self.client)
                if ok:
                    self.dormant = False
                    log.info("ollama_recovered", module="entity")
            except Exception:
                pass
        await self.emotions.tick()
        async with self.store.session() as db:
            await self._ensure_state_hydrated(db)
            ev = self.config.harness.identity.evolution_interval
            if (
                self._interaction_count > 0
                and self._interaction_count % ev == 0
                and self._last_evolution_milestone != self._interaction_count
            ):
                self._last_evolution_milestone = self._interaction_count
                try:
                    await self.evolution.run_deep_cycle(db, self.client, self)
                except Exception as e:
                    log.warning("evolution_cycle_failed", module="entity", error=str(e))
            nar_every = int(self.config.harness.identity.narrative_interval)
            if (
                nar_every > 0
                and self._interaction_count > 0
                and self._interaction_count % nar_every == 0
                and self._last_narrative_milestone != self._interaction_count
            ):
                self._last_narrative_milestone = self._interaction_count
                try:
                    await self.run_narrative_cycle(db)
                    self.personality.invalidate_cache()
                except Exception as e:
                    log.warning("narrative_tick_failed", module="entity", error=str(e))

    async def initiate(self, drive: Drive) -> str:
        eng = InitiativeEngine(self.config, self.client)
        ctx = f"Drive {drive.name}. Inner voice: {self.inner_voice.recent_summary()}"
        return await eng.compose_proactive(drive, ctx)

    async def broadcast_proactive(self, message: str) -> None:
        for fn in self._proactive_sends:
            try:
                await fn(message)
            except Exception as e:
                log.warning("proactive_sink_failed", error=str(e))

    def clear_conversation_history(self) -> None:
        """Clear rolling chat turns (in-memory); does not delete SQLite memory."""
        self._history.clear()

    async def wipe_experiential_state(self) -> None:
        """Clear chat turns, all SQLite memory tables, inner-voice buffer; restore YAML traits; reset mood/drives."""
        self.clear_conversation_history()
        self.personality.invalidate_cache()
        self.inner_voice.clear_buffer()
        self.config.personality.core_traits.clear()
        self.config.personality.core_traits.update(copy.deepcopy(self._yaml_core_traits))
        self.config.personality.behavioral_patterns.clear()
        self.config.personality.behavioral_patterns.update(
            copy.deepcopy(self._yaml_behavioral_patterns)
        )
        self.config.drives.curiosity_topics = list(self._yaml_curiosity_topics)
        self._state_hydrated = False
        self._interaction_count = 0
        self._last_evolution_milestone = -1
        self._last_narrative_milestone = -1
        self._presence_tools_bootstrapped = False
        self._last_turn_completed_at = None
        self._last_conversation = {
            "platform": "cli",
            "channel": "cli",
            "person_id": "",
            "person_name": "",
            "at": 0.0,
        }
        self.emotions.reset_to_initial()
        self.drives.recalibrate_rates_from_traits()
        self.drives.reset_levels()
        async with self.store.session() as conn:
            await self.store.wipe_experience_tables(conn)
        log.info(
            "experiential_state_wiped",
            module="entity",
            db=self.store.db_path,
        )

    async def shutdown(self) -> None:
        try:
            await asyncio.shield(self._mcp_hub.shutdown())
        except BaseException:
            pass
        try:
            await asyncio.shield(self.client.close())
        except BaseException:
            pass
        try:
            await asyncio.shield(self.store.aclose())
        except BaseException:
            pass
