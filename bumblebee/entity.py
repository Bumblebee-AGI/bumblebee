"""Central orchestrator — the entity's self loop."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable

import structlog

from bumblebee.cognition.deliberate import DeliberateCognition
from bumblebee.cognition.inner_voice import InnerVoiceProcessor
from bumblebee.cognition.reflex import ReflexCognition
from bumblebee.cognition.router import CognitionRouter, ContextPackage
from bumblebee.cognition import senses
from bumblebee.config import EntityConfig, validate_ollama_models
from bumblebee.identity.drives import Drive, DriveSystem
from bumblebee.identity.emotions import EmotionEngine
from bumblebee.identity.evolution import EvolutionEngine
from bumblebee.identity.personality import PersonalityEngine
from bumblebee.identity.voice import VoiceController
from bumblebee.memory.beliefs import BeliefStore
from bumblebee.memory.episodic import EpisodicMemory
from bumblebee.memory.imprints import ImprintStore
from bumblebee.memory.narrative import NarrativeMemory, NarrativeSynthesizer
from bumblebee.memory.relational import RelationalMemory
from bumblebee.memory.store import MemoryStore
from bumblebee.models import EmotionCategory, ImprintRecord, Input
from bumblebee.presence.embodiment import Embodiment
from bumblebee.presence.initiative import InitiativeEngine
from bumblebee.presence.tools.filesystem import read_file_safe
from bumblebee.presence.tools.registry import ToolRegistry
from bumblebee.presence.tools.web import fetch_url, search_duckduckgo_lite
from bumblebee.utils.ollama_client import OllamaClient, ToolCallSpec

log = structlog.get_logger("bumblebee.entity")

_CLI_OPENING_USER = (
    "The terminal session just opened — no one has typed anything yet. "
    "You speak first: one short, in-character line (a greeting, a beat of silence, "
    "or a callback to last time if it fits). Do not offer help or ask how you can assist."
)


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
        self._history: list[dict[str, str]] = []
        self._last_user_message_at = time.time()
        self._interaction_count = 0
        self.dormant = False
        self._proactive_sends: list[Callable[[str], Awaitable[None]]] = []
        self._state_hydrated = False
        self._last_evolution_milestone: int = -1

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
        async def search_web(query: str) -> str:
            return await search_duckduckgo_lite(query)

        async def read_local_file(path: str) -> str:
            return await read_file_safe(path)

        async def fetch_page(url: str) -> str:
            return await fetch_url(url)

        self.tools.register_fn(
            "search_web",
            "Search the web for current information (best-effort, no API key).",
            search_web,
        )
        self.tools.register_fn(
            "read_local_file",
            "Read a local file (bounded size).",
            read_local_file,
        )
        self.tools.register_fn(
            "fetch_page",
            "Fetch a URL and return text snippet.",
            fetch_page,
        )

    async def _tool_exec(self, spec: ToolCallSpec) -> str:
        return await self.tools.execute(spec)

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
    ) -> str:
        structlog.contextvars.bind_contextvars(
            entity_name=self.config.name,
            module="entity",
            emotional_state=self.emotions.get_state().primary.value,
        )
        self._last_user_message_at = time.time()
        try:
            ok, missing = await validate_ollama_models(self.config, self.client)
            if not ok:
                self.dormant = True
                await self.emotions.process_stimulus("negative", 0.3, {})
                return (
                    f"I can't fully wake — Ollama doesn't have these models yet: {', '.join(missing)}. "
                    "Pull them with `ollama pull` and I'll be here."
                )
        except Exception as e:
            log.warning("ollama_unreachable", error=str(e))
            self.dormant = True
            await self.emotions.process_stimulus("negative", 0.2, {})
            return (
                "I can't reach the inference server. It feels like sleep without dreams. "
                "When Ollama is back, I'll surface again."
            )

        self.dormant = False
        db = await self.store.connect()
        async with db:
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
            )

            user_content: str | list = senses.input_to_message_content(
                inp,
                self.config.harness.cognition.image_token_budget,
            )
            user_msg = {"role": "user", "content": user_content}

            reply_text = ""
            thinking: str | None = None

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
                reply_text = res.content or "…"
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
                res = await self.deliberate.respond(
                    inp,
                    sys_prompt,
                    self._history + [user_msg],
                    tools=self.tools.openai_tools(),
                    tool_executor=self._tool_exec,
                )
                reply_text = res.content or "…"
                thinking = res.thinking
                slice_ = self.inner_voice.process(thinking, reply_text)
                await self.inner_voice.persist_summary(
                    db,
                    slice_.summary,
                    ",".join(slice_.emotional_cues),
                )
                if stream and reply_text:
                    await _emit_text_stream(stream, reply_text)
                await self.emotions.process_stimulus(
                    "deep",
                    0.35,
                    {"relationship_warmth": rw},
                )

            if record_user_message:
                self._history.append({"role": "user", "content": inp.text[:4000]})
            hist_assistant = reply_text
            if thinking and route == "deliberate":
                hist_assistant = f"{thinking}\n\n{reply_text}"
            self._history.append({"role": "assistant", "content": hist_assistant[:8000]})
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

        return reply_text

    async def cli_opening(
        self,
        stream: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        """First unprompted line when the CLI session connects (not stored as a user turn)."""
        inp = Input(
            text=_CLI_OPENING_USER,
            person_id="cli_user",
            person_name="You",
            channel="cli",
            platform="cli",
        )
        return await self.perceive(
            inp,
            stream=stream,
            record_user_message=False,
            meaningful_override=False,
        )

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
        db = await self.store.connect()
        async with db:
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

    async def shutdown(self) -> None:
        await self.client.close()
