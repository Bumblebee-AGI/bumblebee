"""Central orchestrator — the entity's self loop."""

from __future__ import annotations

import asyncio
import copy
import json
import os
import re
import time
from pathlib import Path
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Any

import structlog

from bumblebee.cognition.deliberate import (
    AgentLoopState,
    DeliberateCognition,
    fit_system_prompt_to_budget,
)
from bumblebee.cognition.inner_voice import InnerVoiceProcessor
from bumblebee.cognition.router import CognitionRouter, ContextPackage
from bumblebee.cognition import gemma, senses
from bumblebee.cognition.history_compression import merge_rolling_summary
from bumblebee.config import EntityConfig, resolve_firecrawl_settings, validate_ollama_models
from bumblebee.inference import InferenceProvider, build_inference_provider
from bumblebee.identity.drives import Drive, DriveSystem
from bumblebee.identity.emotions import EmotionEngine
from bumblebee.identity.evolution import EvolutionEngine
from bumblebee.identity.personality import PersonalityEngine
from bumblebee.identity.voice import VoiceController
from bumblebee.memory.beliefs import BeliefStore
from bumblebee.memory.episodic import EpisodicMemory
from bumblebee.memory.imprints import ImprintStore
from bumblebee.memory.journal import Journal
from bumblebee.memory.knowledge import KnowledgeStore
from bumblebee.memory.narrative import NarrativeMemory, NarrativeSynthesizer
from bumblebee.memory.relational import RelationalMemory
from bumblebee.models import EmotionCategory, ImprintRecord, Input, is_group_like_chat, speaker_label_for_model
from bumblebee.presence.embodiment import Embodiment
from bumblebee.presence.initiative import InitiativeEngine
from bumblebee.presence.platforms.base import Platform
from bumblebee.presence.tools.filesystem import (
    append_file,
    list_directory,
    read_file,
    search_files,
    write_file,
)
from bumblebee.presence.tools import (
    browser as browser_tools,
    code as code_tools,
    imagegen as imagegen_tools,
    messaging as messaging_tools,
    news as news_tools,
    pdf as pdf_tools,
    reddit as reddit_tools,
    reminders as reminders_tools,
    shell as shell_tools,
    system_info as system_info_tools,
    voice as voice_tools,
    weather as weather_tools,
    wikipedia as wikipedia_tools,
    youtube as youtube_tools,
)
from bumblebee.presence.tools import automations as automation_tools
from bumblebee.presence.tools import discovery as discovery_tools
from bumblebee.presence.tools import journal_tools
from bumblebee.presence.tools.knowledge import register_knowledge_tool
from bumblebee.presence.tools.mcp import MCPHub
from bumblebee.presence.tools.registry import ToolRegistry, format_tool_activity
from bumblebee.presence.tools.runtime import (
    ToolRuntimeContext,
    reset_tool_runtime,
    set_tool_runtime,
)
from bumblebee.presence.tools import time_tools
from bumblebee.presence.tools import web as web_tools
from bumblebee.utils.clock import parse_entity_created_timestamp
from bumblebee.storage import (
    attachment_refs_episode_note,
    build_attachment_store,
    create_memory_store,
    load_stored_attachment,
    persist_incoming_attachments,
)
from bumblebee.utils.ollama_client import ChatCompletionResult, ToolCallSpec

log = structlog.get_logger("bumblebee.entity")

# entity_state key: JSON {"entries": {cache_key: route_dict, ...}} for cross-session messaging routes.
PERSON_ROUTES_STATE_KEY = "person_routes_v1"

_CLI_OPENING_USER = (
    "The terminal session just opened — no one has typed anything yet. "
    "You speak first: one short, in-character line (a greeting, a beat of silence, "
    "or a callback to last time if it fits). Do not offer help or ask how you can assist."
)

_FALLBACK_PLAIN_REPLY_FAILURE = (
    "something went wrong on my end and i couldn't finish that reply. "
    "try me again in a sec."
)


def _short_error_text(raw: str | None, *, limit: int = 180) -> str:
    txt = re.sub(r"\s+", " ", str(raw or "")).strip().strip(".")
    if not txt:
        return ""
    if len(txt) > limit:
        txt = txt[: limit - 1].rstrip() + "…"
    return txt


def format_user_visible_failure(
    error: Exception | str | None = None,
    *,
    tool_state: dict[str, Any] | None = None,
) -> str:
    """Short user-facing failure line based on the real cause, not a generic model-server hint."""
    tool_err = None
    if isinstance(tool_state, dict):
        raw = tool_state.get("last_tool_error")
        if isinstance(raw, dict):
            tool_err = raw
    if tool_err:
        tool_name = str(tool_err.get("tool") or "tool").replace("_", " ").strip()
        detail = _short_error_text(str(tool_err.get("error") or ""))
        low = detail.lower()
        if "locked to railway" in low or "railway container" in low:
            return f"i tried to use {tool_name}, but execution is locked to railway right now. {detail}"
        if "timeout" in low or "timed out" in low:
            return f"i tried to use {tool_name}, but it timed out. {detail}"
        if "no search results" in low or "empty" in low or "not found" in low:
            return f"i tried to use {tool_name}, but it came back empty. {detail}"
        if detail:
            return f"something went wrong while using {tool_name}. {detail}"
        return f"something went wrong while using {tool_name}."

    detail = _short_error_text(str(error or ""))
    low = detail.lower()
    if any(
        marker in low
        for marker in (
            "connection refused",
            "connect",
            "unreachable",
            "timed out",
            "timeout",
            "ollama",
            "gateway",
            "api connection",
            "clientconnectorerror",
            "getaddrinfo failed",
        )
    ):
        return "i lost the inference backend mid-reply. try me again in a sec."
    if detail:
        return f"something went wrong on my end and i couldn't finish that reply. {detail}"
    return _FALLBACK_PLAIN_REPLY_FAILURE


def _reply_too_thin(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 2:
        return True
    if t in ("…", "...", "—", "-"):
        return True
    return gemma.visible_reply_looks_truncated_stub(t)


def _intermediate_text_looks_like_tool_channel(text: str) -> bool:
    """True when the model leaked Gemma tool markup — must not be shown as chat."""
    t = text or ""
    return any(
        m in t
        for m in (
            gemma.TOOL_CALL_START,
            gemma.TOOL_RESPONSE_START,
            gemma.TOOL_DECL_START,
            gemma.TURN_START,
        )
    )


# After tool calls, models often emit a one-token ack; we already shipped the visible line pre-tool.
_PRO_FORMA_TOOL_FOLLOWUPS = frozenset(
    {
        "done",
        "done.",
        "ok",
        "ok.",
        "k",
        "k.",
        "yep",
        "yep.",
        "yeah",
        "yeah.",
        "there",
        "there.",
        "got it",
        "got it.",
        "sorted",
        "sorted.",
        "cool",
        "cool.",
        "all set",
        "all set.",
        "finished",
        "finished.",
        "i'm done",
        "i'm done.",
        "im done",
        "im done.",
        "i am done",
        "i am done.",
        "great",
        "great.",
        "nice",
        "nice.",
        "bet",
        "bet.",
        "alright",
        "aight",
    }
)


def _pro_forma_tool_followup(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return True
    if re.fullmatch(r"i['\u2019]?m\s+done\.?", t):
        return True
    if t in _PRO_FORMA_TOOL_FOLLOWUPS:
        return True
    base = t.rstrip(".!?")
    if len(t) <= 20 and " " not in t:
        return base in {
            "done",
            "ok",
            "k",
            "yep",
            "yeah",
            "there",
            "cool",
            "nice",
            "great",
            "bet",
            "alright",
            "aight",
        }
    return False


def _reply_looks_like_progress_only(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    starters = (
        "checking ",
        "checking the ",
        "looking ",
        "looking in ",
        "looking around",
        "searching ",
        "reading ",
        "opening ",
        "hang on",
        "hold on",
        "one sec",
        "give me a sec",
        "lemme ",
        "let me ",
    )
    if any(t.startswith(s) for s in starters):
        return True
    return bool(
        re.fullmatch(
            r"(i['\u2019]?m|im|i am)\s+(checking|looking|searching|reading|opening).{0,80}",
            t,
        )
    )


def _user_explicitly_requests_tool_grounding(text: str) -> bool:
    low = (text or "").lower()
    return any(
        phrase in low
        for phrase in (
            "use tools",
            "not memory",
            "don't guess",
            "do not guess",
            "exact path",
            "exact stdout",
            "exact error",
            "exact contents",
        )
    )


def _finish_reason_hint(res: ChatCompletionResult) -> bool:
    fr = str(res.finish_reason or "").lower().replace("_", "").replace("-", "")
    return fr in ("length", "maxtokens", "maxlength")


def _answer_ignores_tool_results(text: str, tool_state: dict[str, Any]) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return True
    previews = tool_state.get("tool_output_previews") if isinstance(tool_state, dict) else None
    if not isinstance(previews, list) or not previews:
        return False
    if "/" in t or "\\" in t:
        return False
    if any(ch.isdigit() for ch in t):
        return False
    interesting = 0
    for item in previews[-3:]:
        if not isinstance(item, dict):
            continue
        preview = str(item.get("preview") or "")
        tokens = [tok.strip(" .,:;()[]{}\"'") for tok in preview.split()]
        tokens = [tok.lower() for tok in tokens if len(tok) >= 4]
        hits = [tok for tok in tokens[:12] if tok in t]
        if hits:
            interesting += 1
    return interesting == 0


def _tool_state_summary(tool_state: dict[str, Any], *, limit: int = 3) -> str:
    previews = tool_state.get("tool_output_previews") if isinstance(tool_state, dict) else None
    if not isinstance(previews, list) or not previews:
        return ""
    lines: list[str] = []
    for item in previews[-limit:]:
        if not isinstance(item, dict):
            continue
        tool = str(item.get("tool") or "tool")
        ok = bool(item.get("ok"))
        preview = _short_error_text(str(item.get("preview") or ""), limit=220)
        lines.append(f"- {tool} ({'ok' if ok else 'error'}): {preview}")
    return "\n".join(lines)


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
        self.client: InferenceProvider = build_inference_provider(config.harness)
        self.store = create_memory_store(config)
        self.attachments = build_attachment_store(config)
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
        self._history_rolling_summary: str = ""
        self._last_user_message_at = time.time()
        self._interaction_count = 0
        self.dormant = False
        self._proactive_sends: list[Callable[[str], Awaitable[None]]] = []
        self._state_hydrated = False
        self._last_evolution_milestone: int = -1
        self._last_narrative_milestone: int = -1
        self._last_turn_completed_at: float | None = None
        self.automation_engine: object | None = None
        self.journal = Journal(Path(self.config.journal_path()))
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
        self._platforms: dict[str, Platform] = {}
        self._person_routes: dict[str, dict[str, str | float]] = {}

    def register_proactive_sink(self, fn: Callable[[str], Awaitable[None]]) -> None:
        self._proactive_sends.append(fn)

    def register_platform(self, name: str, platform: Platform) -> None:
        n = (name or "").strip().lower()
        if not n:
            return
        self._platforms[n] = platform

    async def send_message_to_platform(
        self,
        platform: str,
        target: str,
        message: str,
        *,
        as_voice: bool = False,
        voice_id: str = "",
    ) -> None:
        n = (platform or "").strip().lower()
        pf = self._platforms.get(n)
        if pf is None:
            raise RuntimeError(f"platform not connected: {platform}")
        msg = (message or "").strip()
        if not msg:
            raise RuntimeError("message is empty")
        if not as_voice:
            await pf.send_message(str(target), msg)
            return
        out_path = await voice_tools.synthesize_tts_to_file(self, msg, voice_id)
        try:
            send_audio = getattr(pf, "send_audio", None)
            if not callable(send_audio):
                raise RuntimeError(f"platform {platform!r} does not support voice delivery (send_audio)")
            ok = await send_audio(str(target), str(out_path))
            if not ok:
                raise RuntimeError("voice delivery failed (send_audio returned false)")
        finally:
            try:
                out_path.unlink(missing_ok=True)
            except OSError:
                pass

    async def send_dm_to_user(
        self,
        platform: str,
        user_id: str,
        message: str,
        *,
        as_voice: bool = False,
        voice_id: str = "",
    ) -> None:
        """Send a private DM using the user's platform id (Telegram: chat_id == user id; Discord: user snowflake)."""
        n = (platform or "").strip().lower()
        uid = (user_id or "").strip()
        msg = (message or "").strip()
        if not uid:
            raise RuntimeError("user_id is required")
        if not msg:
            raise RuntimeError("message is empty")
        pf = self._platforms.get(n)
        if pf is None:
            raise RuntimeError(f"platform not connected: {platform}")
        if not as_voice:
            if n == "telegram":
                await pf.send_message(uid, msg)
                return
            if n == "discord":
                dm = getattr(pf, "send_dm_to_user", None)
                if not callable(dm):
                    raise RuntimeError("discord platform does not support send_dm_to_user")
                await dm(uid, msg)
                return
            raise RuntimeError(f"send_dm_to_user is only supported for telegram and discord, not {platform!r}")

        out_path = await voice_tools.synthesize_tts_to_file(self, msg, voice_id)
        try:
            if n == "telegram":
                send_audio = getattr(pf, "send_audio", None)
                if not callable(send_audio):
                    raise RuntimeError("telegram does not support voice delivery (send_audio)")
                ok = await send_audio(uid, str(out_path))
                if not ok:
                    raise RuntimeError("telegram voice DM failed (send_audio returned false)")
                return
            if n == "discord":
                dm_audio = getattr(pf, "send_dm_audio", None)
                if not callable(dm_audio):
                    raise RuntimeError("discord platform does not support send_dm_audio")
                ok = await dm_audio(uid, str(out_path))
                if not ok:
                    raise RuntimeError("discord voice DM failed")
                return
            raise RuntimeError(f"voice DM is only supported for telegram and discord, not {platform!r}")
        finally:
            try:
                out_path.unlink(missing_ok=True)
            except OSError:
                pass

    def _remember_person_route(self, inp: Input) -> None:
        chat_type = str((inp.metadata or {}).get("chat_type") or "").strip().lower()
        route = {
            "platform": inp.platform,
            "channel": inp.channel,
            "person_id": inp.person_id,
            "person_name": inp.person_name,
            "chat_type": chat_type,
            "at": time.time(),
        }
        pid = (inp.person_id or "").strip()
        pname = (inp.person_name or "").strip().casefold()
        if pid:
            self._person_routes[pid] = route
        if pname:
            self._person_routes[f"name:{pname}"] = route

    def _apply_person_routes_state(self, raw: str) -> None:
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            log.warning("person_routes_state_invalid_json", module="entity")
            return
        entries = data.get("entries")
        if not isinstance(entries, dict):
            return
        for k, v in entries.items():
            if not isinstance(k, str) or not isinstance(v, dict):
                continue
            self._person_routes[k] = {
                "platform": str(v.get("platform") or ""),
                "channel": str(v.get("channel") or ""),
                "person_id": str(v.get("person_id") or ""),
                "person_name": str(v.get("person_name") or ""),
                "chat_type": str(v.get("chat_type") or ""),
                "at": float(v.get("at") or 0.0),
            }

    async def _persist_person_routes(self, conn) -> None:
        payload = json.dumps({"entries": self._person_routes}, ensure_ascii=False)
        await conn.execute(
            "INSERT OR REPLACE INTO entity_state (key, value) VALUES (?, ?)",
            (PERSON_ROUTES_STATE_KEY, payload),
        )
        await conn.commit()

    def list_known_person_routes(self, platform: str = "") -> list[dict[str, str | float]]:
        pf = (platform or "").strip().lower()
        out: list[dict[str, str | float]] = []
        seen: set[tuple[str, str, str, str]] = set()
        for r in self._person_routes.values():
            rp = str(r.get("platform") or "").strip().lower()
            if pf and rp != pf:
                continue
            key = (
                rp,
                str(r.get("channel") or ""),
                str(r.get("person_id") or ""),
                str(r.get("person_name") or ""),
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(dict(r))
        out.sort(key=lambda x: float(x.get("at") or 0.0), reverse=True)
        return out

    @staticmethod
    def _pick_best_route(
        candidates: list[dict[str, str | float]],
        *,
        platform: str,
        prefer_private: bool,
    ) -> dict[str, str | float] | None:
        cands = list(candidates)
        if platform:
            cands = [c for c in cands if str(c.get("platform") or "").strip().lower() == platform]
        if not cands:
            return None
        if prefer_private:
            priv = [
                c
                for c in cands
                if str(c.get("chat_type") or "").strip().lower() in ("private", "dm", "direct")
            ]
            if priv:
                cands = priv
        cands.sort(key=lambda x: float(x.get("at") or 0.0), reverse=True)
        return cands[0] if cands else None

    def resolve_person_route(
        self,
        target_person: str,
        *,
        platform: str = "",
        prefer_private: bool = True,
    ) -> dict[str, str | float] | None:
        t = (target_person or "").strip()
        if not t:
            return None
        pf = (platform or "").strip().lower()
        low = t.casefold()

        exact: list[dict[str, str | float]] = []
        by_id = self._person_routes.get(t)
        if by_id:
            exact.append(dict(by_id))
        by_name = self._person_routes.get(f"name:{low}")
        if by_name:
            exact.append(dict(by_name))
        picked = self._pick_best_route(exact, platform=pf, prefer_private=prefer_private)
        if picked:
            return picked

        fuzzy: list[dict[str, str | float]] = []
        for r in self.list_known_person_routes(pf):
            pid = str(r.get("person_id") or "").strip()
            pname = str(r.get("person_name") or "").strip().casefold()
            if pid == t or pname == low or low in pname:
                fuzzy.append(r)
        return self._pick_best_route(fuzzy, platform=pf, prefer_private=prefer_private)

    async def fetch_cli_header_counts(self) -> tuple[int, int, float | None]:
        async with self.store.session() as conn:
            ne = await self.store.count_episodes(conn)
            nr = await self.store.count_relationships(conn)
            mt = await self.store.min_episode_timestamp(conn)
        return ne, nr, mt

    async def fetch_cli_recent_summaries(self, limit: int = 5) -> list[str]:
        async with self.store.session() as conn:
            return await self.episodic.recent_summaries(conn, limit)

    async def read_stored_attachment(self, storage_ref: str) -> bytes | None:
        """Load bytes for a ``storage_ref`` produced by inbound attachment persistence (outbound / tools)."""
        return await load_stored_attachment(self.attachments, storage_ref)

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
                elif k == PERSON_ROUTES_STATE_KEY:
                    self._apply_person_routes_state(v if isinstance(v, str) else str(v))
            except (json.JSONDecodeError, TypeError):
                continue

    async def _ensure_state_hydrated(self, conn) -> None:
        if self._state_hydrated:
            return
        await self._hydrate_entity_state(conn)
        self._state_hydrated = True

    @staticmethod
    def _merge_tool_cfg(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
        out = {**base}
        for k, v in over.items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = Entity._merge_tool_cfg(out[k], v)
            else:
                out[k] = v
        return out

    def _effective_tools_cfg(self) -> dict[str, Any]:
        base = self.config.harness.tools if isinstance(self.config.harness.tools, dict) else {}
        over = self.config.raw.get("tools") if isinstance(self.config.raw, dict) else {}
        if not isinstance(over, dict):
            over = {}
        return self._merge_tool_cfg(base, over)

    def _tool_enabled(self, key: str, default: bool) -> bool:
        cfg = self._effective_tools_cfg().get(key)
        if isinstance(cfg, dict) and "enabled" in cfg:
            return bool(cfg.get("enabled"))
        return default

    def _register_tools(self) -> None:
        self.tools.register_decorated(web_tools.search_web)
        self.tools.register_decorated(web_tools.fetch_url)
        self.tools.register_decorated(read_file)
        self.tools.register_decorated(list_directory)
        self.tools.register_decorated(search_files)
        self.tools.register_decorated(write_file)
        self.tools.register_decorated(append_file)
        self.tools.register_decorated(time_tools.get_current_time)
        self.tools.register_decorated(discovery_tools.search_tools)
        self.tools.register_decorated(discovery_tools.describe_tool)
        if self._tool_enabled("youtube", True):
            self.tools.register_decorated(youtube_tools.get_youtube_transcript)
            self.tools.register_decorated(youtube_tools.search_youtube)
        if self._tool_enabled("reddit", True):
            self.tools.register_decorated(reddit_tools.read_reddit)
            self.tools.register_decorated(reddit_tools.read_reddit_post)
        if self._tool_enabled("wikipedia", True):
            self.tools.register_decorated(wikipedia_tools.read_wikipedia)
        if self._tool_enabled("weather", True):
            self.tools.register_decorated(weather_tools.get_weather)
        if self._tool_enabled("news", True):
            self.tools.register_decorated(news_tools.get_news)
        if self._tool_enabled("pdf", True):
            self.tools.register_decorated(pdf_tools.read_pdf)
        if self._tool_enabled("voice", True):
            self.tools.register_decorated(voice_tools.speak)
            self.tools.register_decorated(voice_tools.get_tts_voice)
            self.tools.register_decorated(voice_tools.list_tts_voices)
            self.tools.register_decorated(voice_tools.set_tts_voice)
        if self._tool_enabled("reminders", True):
            self.tools.register_decorated(reminders_tools.set_reminder)
            self.tools.register_decorated(reminders_tools.list_reminders)
            self.tools.register_decorated(reminders_tools.cancel_reminder)
        if self._tool_enabled("messaging", True):
            self.tools.register_decorated(messaging_tools.send_message_to)
            self.tools.register_decorated(messaging_tools.list_known_contacts)
            self.tools.register_decorated(messaging_tools.send_dm)
        if self._tool_enabled("automations", True):
            self.tools.register_decorated(automation_tools.create_automation)
            self.tools.register_decorated(automation_tools.list_automations)
            self.tools.register_decorated(automation_tools.edit_automation)
            self.tools.register_decorated(automation_tools.toggle_automation)
            self.tools.register_decorated(automation_tools.delete_automation)
            self.tools.register_decorated(automation_tools.run_automation_now)
        if self._tool_enabled("journal", True):
            self.tools.register_decorated(journal_tools.read_journal)
            self.tools.register_decorated(journal_tools.write_journal)
        if self._tool_enabled("system", True):
            self.tools.register_decorated(system_info_tools.get_system_info)
        if self._tool_enabled("shell", True):
            self.tools.register_decorated(shell_tools.run_command)
            self.tools.register_decorated(shell_tools.run_background)
            self.tools.register_decorated(shell_tools.check_process)
            self.tools.register_decorated(shell_tools.kill_process)
        if self._tool_enabled("code", True):
            self.tools.register_decorated(code_tools.execute_python)
            self.tools.register_decorated(code_tools.execute_javascript)
        if self._tool_enabled("browser", False):
            self.tools.register_decorated(browser_tools.browser_navigate)
            self.tools.register_decorated(browser_tools.browser_screenshot)
            self.tools.register_decorated(browser_tools.browser_click)
            self.tools.register_decorated(browser_tools.browser_type)
        if self._tool_enabled("imagegen", False):
            self.tools.register_decorated(imagegen_tools.generate_image)
        register_knowledge_tool(self.tools, self.config, self.knowledge)

    async def refresh_mcp_servers(self) -> None:
        await self._mcp_hub.refresh(self.tools, self.config.raw.get("mcp_servers"))

    async def _ensure_presence_tools(self) -> None:
        if self._presence_tools_bootstrapped:
            return
        await self.refresh_mcp_servers()
        self._presence_tools_bootstrapped = True

    @staticmethod
    def _message_content_str(content: Any) -> str:
        if isinstance(content, list):
            return gemma.stringify_content_blocks(content)
        return str(content or "")

    def _recent_conversation_for_knowledge(self, inp: Input, max_turns: int | None = None) -> str:
        """Last few chat turns plus the current user message — for knowledge retrieval."""
        if max_turns is None:
            max_turns = max(2, int(self.config.cognition.knowledge_recent_turns or 10))
        lines: list[str] = []
        hc = self.config.cognition.history_compression
        rs = (self._history_rolling_summary or "").strip()
        if hc.enabled and rs:
            lines.append(f"[earlier thread summary]\n{rs[:2500]}")
        for m in self._history[-max_turns:]:
            role = str(m.get("role", ""))
            c = self._message_content_str(m.get("content"))
            lines.append(f"{role}: {c[:2000]}")
        current_line = f"user: {self._history_user_turn_text(inp)[:2000]}".strip()
        if not lines or lines[-1].strip() != current_line:
            lines.append(current_line)
        return "\n".join(lines)

    def _messages_for_model(self, user_msg: dict[str, Any]) -> list[dict[str, Any]]:
        """Rolling chat + optional compressed prefix (same thread, not a new session)."""
        hc = self.config.cognition.history_compression
        s = (self._history_rolling_summary or "").strip()
        um = self._message_content_str(user_msg.get("content")).strip()
        hist = list(self._history)
        if hist and str(hist[-1].get("role", "")) == "user" and um:
            if self._message_content_str(hist[-1].get("content")).strip() == um:
                hist = hist[:-1]
        if not s or not hc.enabled:
            return hist + [user_msg]
        cap = min(int(hc.summary_max_chars), 8000)
        body = s[:cap]
        prefix = (
            "[Earlier conversation summary — same ongoing chat; continue naturally; "
            "do not reset with a generic greeting unless it genuinely fits]\n\n" + body
        )
        return [{"role": "user", "content": prefix}] + hist + [user_msg]

    async def _trim_history_with_compression(self) -> None:
        keep = max(8, int(self.config.cognition.rolling_history_max_messages or 40))
        if len(self._history) <= keep:
            return
        dropped = self._history[:-keep]
        self._history = self._history[-keep:]
        hc = self.config.cognition.history_compression
        if not hc.enabled or not dropped:
            return
        model = self.config.cognition.deliberate_model or self.config.harness.models.deliberate
        self._history_rolling_summary = await merge_rolling_summary(
            self.client,
            model,
            entity_name=self.config.name,
            prior_summary=self._history_rolling_summary,
            dropped_messages=dropped,
            per_msg_cap=hc.format_per_message_chars,
            max_merge_input_chars=hc.max_merge_input_chars,
            merge_max_tokens=hc.merge_max_tokens,
            summary_max_chars=hc.summary_max_chars,
            num_ctx=self.config.effective_ollama_num_ctx(),
        )

    def _history_user_turn_text(self, inp: Input) -> str:
        lim = max(600, int(self.config.cognition.history_message_char_limit or 4000))
        label = speaker_label_for_model(inp)

        def clip_body(body: str) -> str:
            if not label:
                return body[:lim] if body else "…"
            room = max(64, lim - len(label))
            if len(body) <= room:
                return label + body
            return label + body[: max(1, room - 1)] + "…"

        t = (inp.text or "").strip()
        if inp.images:
            extra = "[User attached an image]"
            core = f"{t}\n{extra}".strip() if t else extra
            return clip_body(core)
        if inp.audio:
            extra = "[User sent a voice message]"
            core = f"{t}\n{extra}".strip() if t else extra
            return clip_body(core)
        return clip_body(t) if t else (label + "…") if label else "…"

    async def _tool_exec(
        self,
        spec: ToolCallSpec,
        *,
        inp: Input | None = None,
        state: dict[str, Any] | None = None,
    ) -> str:
        tok = set_tool_runtime(
            ToolRuntimeContext(
                entity=self,
                inp=inp,
                platform=self.current_platform,
                state=state if state is not None else {},
            )
        )
        try:
            out = await self.tools.execute(spec)
        finally:
            reset_tool_runtime(tok)
        if state is not None:
            state["tool_calls"] = int(state.get("tool_calls") or 0) + 1
            state["last_tool_name"] = spec.name
            try:
                parsed = json.loads(out)
            except Exception:
                parsed = None
            ok = True
            if isinstance(parsed, dict):
                ok = not bool(parsed.get("error")) and parsed.get("ok", True) is not False
                err = str(parsed.get("error") or "").strip()
                if err:
                    state["last_tool_error"] = {"tool": spec.name, "error": err}
            if ok:
                state["tool_successes"] = int(state.get("tool_successes") or 0) + 1
                state.pop("last_tool_error", None)
            else:
                state["tool_failures"] = int(state.get("tool_failures") or 0) + 1
            previews = state.setdefault("tool_output_previews", [])
            if isinstance(previews, list):
                previews.append(
                    {
                        "tool": spec.name,
                        "ok": ok,
                        "preview": self._message_content_str(out)[:280],
                    }
                )
                if len(previews) > 8:
                    del previews[:-8]
        if spec.name in ("search_web", "fetch_url"):
            self.drives.satisfy("curiosity", 0.22)
        return out

    async def _loop_completion_gate(
        self,
        inp: Input,
        reply_text: str,
        res: ChatCompletionResult,
        loop_state: AgentLoopState,
        tool_state: dict[str, Any],
    ) -> tuple[bool, str | None]:
        """Decide whether the shared agent loop is done with this turn."""
        visible = (reply_text or "").strip()
        if not visible:
            if loop_state.tool_calls_seen > 0 or _finish_reason_hint(res):
                return False, (
                    "Same turn. You haven't answered yet. Either answer from the tool results, "
                    "call another tool, or state the exact failure."
                )
            return False, "Same turn. You haven't answered the user yet. Continue."
        if _reply_looks_like_progress_only(visible):
            return False, (
                "Same turn. That was progress chatter, not the final answer. Continue until you can "
                "answer from results or state the exact failure."
            )
        if _pro_forma_tool_followup(visible) and loop_state.tool_calls_seen > 0:
            return False, (
                "Same turn. A one-word acknowledgement is not enough. Answer the user from the tool "
                "results or explain the exact failure."
            )
        explicit_grounding = _user_explicitly_requests_tool_grounding(inp.text)
        if explicit_grounding and loop_state.tool_calls_seen <= 0:
            return False, (
                "Same turn. The user explicitly asked you to use tools. Call the relevant tool now, "
                "then answer from the result."
            )
        if loop_state.last_tool_failed:
            if _pro_forma_tool_followup(visible) or _reply_looks_like_progress_only(visible):
                return False, (
                    "Same turn. The last tool failed. Either retry with another tool or tell the "
                    "user exactly what failed."
                )
        needs_grounding_judgment = (
            explicit_grounding
            or loop_state.tool_calls_seen > 0
            or loop_state.last_tool_failed
        )
        if needs_grounding_judgment:
            judged_done, judged_reason = await self._judge_grounded_completion(
                inp,
                visible,
                res,
                loop_state,
                tool_state,
            )
            if not judged_done:
                return False, judged_reason
        return True, None

    async def _judge_grounded_completion(
        self,
        inp: Input,
        reply_text: str,
        res: ChatCompletionResult,
        loop_state: AgentLoopState,
        tool_state: dict[str, Any],
    ) -> tuple[bool, str | None]:
        """
        Lightweight model-side judgment for grounded completion.

        This is intentionally cheap and narrow: only used when tool grounding matters.
        """
        visible = (reply_text or "").strip()
        if not visible:
            return False, (
                "Same turn. You still need to answer the user from the current results or state the exact failure."
            )
        if _answer_ignores_tool_results(visible, tool_state) and loop_state.tool_calls_seen > 0:
            heuristic_reason = (
                "Same turn. You used tools, but your last message did not actually report the results. "
                "Answer from the tool outputs directly."
            )
        else:
            heuristic_reason = None
        tool_summary = _tool_state_summary(tool_state)
        user_request = (inp.text or "").strip()[:1200]
        candidate = visible[:1000]
        prompt = (
            "Reply with exactly one line starting with DONE or CONTINUE, then a colon, then a short reason.\n"
            "DONE = the candidate reply clearly answers the user's ask and is adequately grounded in the tool results.\n"
            "CONTINUE = the candidate reply is only progress chatter, ignores tool results, is too vague, "
            "or should keep going before the user sees it.\n"
            "Be strict. If the user asked for exactness or tools were used, the reply must reflect the actual tool outputs.\n\n"
            f"User request:\n{user_request}\n\n"
            f"Candidate reply:\n{candidate}\n\n"
            f"Tool rounds: {loop_state.tool_rounds}\n"
            f"Tool calls seen: {loop_state.tool_calls_seen}\n"
            f"Last tool failed: {loop_state.last_tool_failed}\n"
            f"Finish reason: {res.finish_reason or ''}\n\n"
            f"Recent tool outputs:\n{tool_summary or '- none'}"
        )
        try:
            judge = await self.client.chat_completion(
                self.config.cognition.reflex_model or self.config.cognition.deliberate_model,
                [
                    {
                        "role": "system",
                        "content": (
                            "You are a tiny completion judge for an agent loop. "
                            "Output exactly one line: DONE: <reason> or CONTINUE: <reason>."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=40,
                think=False,
                num_ctx=self.config.effective_ollama_num_ctx(),
            )
            verdict = (judge.content or "").strip()
            upper = verdict.upper()
            if upper.startswith("DONE:"):
                return True, None
            if upper.startswith("CONTINUE:"):
                reason = verdict.split(":", 1)[1].strip() or "Keep going until the answer is grounded."
                return False, f"Same turn. {reason}"
        except Exception as e:
            log.debug("grounded_completion_judge_failed", module="entity", error=str(e))
        if heuristic_reason:
            return False, heuristic_reason
        if _finish_reason_hint(res):
            return False, (
                "Same turn. The last step looks incomplete. Continue until you can answer clearly from the results."
            )
        return True, None

    def _fallback_user_turn_text(self, inp: Input) -> str:
        """User line for fallback completion; includes rolling summary when compression is on."""
        fb_line = speaker_label_for_model(inp) + (inp.text or "")[:4000]
        hc = self.config.cognition.history_compression
        s = (self._history_rolling_summary or "").strip()
        if not s or not hc.enabled:
            return fb_line
        cap = min(int(hc.summary_max_chars), 4000)
        body = s[:cap]
        prefix = (
            "[Earlier conversation summary — same ongoing chat; continue naturally; "
            "do not reset with a generic greeting unless it genuinely fits]\n\n" + body
        )
        return prefix + "\n\n---\n\nCurrent message:\n" + fb_line

    async def _fallback_plain_reply(
        self,
        inp: Input,
        *,
        tool_state: dict[str, Any] | None = None,
    ) -> str:
        """When the main path yields no visible text (parsing/Ollama quirks), one short no-thinking completion.

        Retries use ``harness.ollama.retry_attempts`` / ``retry_delay``; if all attempts fail, returns a
        user-visible error instead of silence or a generic greeting.
        """
        name = self.config.name
        sys_ = (
            f"You are {name}, texting naturally. Answer in 1–3 short sentences. "
            f"If asked who you are, answer as {name} in character — never Gemma/Google/'large language model' intros. "
            "No stage directions: do not describe pauses, actions, or parenthetical asides—only words you say."
        )
        fb_user = self._fallback_user_turn_text(inp)
        o = self.config.harness.ollama
        attempts = max(1, int(o.retry_attempts))
        delay = float(o.retry_delay)
        last_error: str | None = None
        for attempt in range(attempts):
            try:
                res = await self.client.chat_completion(
                    self.config.cognition.reflex_model,
                    [
                        {"role": "system", "content": sys_},
                        {"role": "user", "content": fb_user},
                    ],
                    temperature=min(0.9, self.config.harness.cognition.temperature),
                    max_tokens=min(384, self.config.harness.cognition.reflex_max_tokens + 128),
                    think=False,
                    num_ctx=self.config.effective_ollama_num_ctx(),
                )
                out = (res.content or "").strip()
                if not _reply_too_thin(out):
                    return out
                log.info(
                    "fallback_reply_thin",
                    module="entity",
                    attempt=attempt + 1,
                    attempts=attempts,
                    preview=out[:120] if out else "",
                )
            except Exception as e:
                last_error = str(e)
                log.warning(
                    "fallback_reply_failed",
                    module="entity",
                    attempt=attempt + 1,
                    attempts=attempts,
                    error=last_error,
                )
            if attempt + 1 < attempts:
                await asyncio.sleep(delay * (2**attempt))
        log.warning(
            "fallback_reply_exhausted",
            module="entity",
            attempts=attempts,
            last_error=last_error,
        )
        return format_user_visible_failure(last_error, tool_state=tool_state)

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
        msgs: list[dict] = [
            {"role": "system", "content": fit_system_prompt_to_budget(sys_prompt, sys_cap)},
        ]
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
                num_ctx=self.config.effective_ollama_num_ctx(),
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
            else (self.config.cognition.reflex_model or self.config.cognition.deliberate_model)
        )
        sys_cap = max(2000, int(self.config.cognition.system_prompt_char_limit or 12000))
        if is_deliberate:
            mt = int(self.config.cognition.deliberate_max_tokens or self.config.harness.cognition.deliberate_max_tokens)
        else:
            mt = self.config.harness.cognition.reflex_max_tokens
        return await self._continue_cutoff_reply(
            model=model,
            sys_prompt=sys_prompt,
            recent_messages=self._messages_for_model(user_msg),
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
        preserve_conversation_route: bool = False,
        routine_history: bool = True,
        skip_relational_upsert: bool = False,
        skip_drive_interaction: bool = False,
        skip_episode: bool = False,
        tool_names_log: list[str] | None = None,
    ) -> tuple[str, bool]:
        structlog.contextvars.bind_contextvars(
            entity_name=self.config.name,
            module="entity",
            emotional_state=self.emotions.get_state().primary.value,
        )
        self.current_platform = reply_platform
        prev_completed_turn_at = self._last_turn_completed_at
        prev_interlocutor_name = str(self._last_conversation.get("person_name") or "").strip()
        if not preserve_conversation_route:
            self._last_user_message_at = time.time()
            self._last_conversation = {
                "platform": inp.platform,
                "channel": inp.channel,
                "person_id": inp.person_id,
                "person_name": inp.person_name,
                "at": time.time(),
            }
            # In-memory routes for this process (e.g. turns that return before DB session).
            self._remember_person_route(inp)
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

        try:
            inp = await persist_incoming_attachments(self.attachments, inp)
        except Exception as e:
            log.warning("attachment_persist_batch_failed", module="entity", error=str(e))

        if inp.audio:
            aud0 = inp.audio[0]
            transcript = await senses.transcribe_audio_attachment(
                self.client,
                self.config.cognition.reflex_model,
                base64_audio=str(aud0.get("base64", "")),
                audio_format=str(aud0.get("format", "ogg")),
                num_ctx=self.config.effective_ollama_num_ctx(),
            )
            base = (inp.text or "").strip()
            if transcript:
                merged = transcript if not base else f"{base}\n\n[Voice]: {transcript}"
            else:
                merged = base or "[Voice message — I couldn't transcribe it.]"
            inp = replace(inp, text=merged, audio=[])

        async with self.store.session() as db:
            await self._ensure_state_hydrated(db)
            if not preserve_conversation_route:
                # Refresh timestamps after merge-load from entity_state.
                self._remember_person_route(inp)
                try:
                    await self._persist_person_routes(db)
                except Exception as e:
                    log.warning("persist_person_routes_failed", module="entity", error=str(e))

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

            if is_group_like_chat(inp):
                sys_prompt += (
                    "\n\n[Chat context: group channel — several people may post here. "
                    "User lines are prefixed with [display name · id …] (and #channel on Discord) "
                    "so you can tell speakers apart. Reply to the author of the latest message unless "
                    "they are clearly addressing everyone or someone else by name.]"
                )

            if route in ("reflex", "deliberate"):
                cap = max(2000, int(self.config.cognition.system_prompt_char_limit or 12000))
                full_tools = self.tools.system_tool_instruction_block()
                tools_block = (
                    self.tools.compact_system_tool_instruction()
                    if full_tools and len(sys_prompt) + len(full_tools) > cap
                    else full_tools
                )
                sys_prompt = sys_prompt + tools_block
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
                speaker_prefix=speaker_label_for_model(inp),
            )
            user_msg = {"role": "user", "content": user_content}

            reply_text = ""
            thinking: str | None = None
            mood_sig = "neutral"
            deliberate_history_extra: list[dict[str, Any]] = []
            intermediate_visible_to_chat = False
            last_intermediate_chat_segment = ""
            tool_state: dict[str, Any] = {}
            final_res: ChatCompletionResult | None = None

            async def _tool_with_activity(spec: ToolCallSpec) -> str:
                if tool_names_log is not None:
                    tool_names_log.append(spec.name)
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
                return await self._tool_exec(spec, inp=inp, state=tool_state)

            async def _final_checker(
                candidate_text: str,
                res: ChatCompletionResult,
                loop_state: AgentLoopState,
            ) -> tuple[bool, str | None]:
                return await self._loop_completion_gate(
                    inp,
                    candidate_text,
                    res,
                    loop_state,
                    tool_state,
                )

            inference_profile = "reflex" if route == "reflex" else "deliberate"
            async for ev in self.deliberate.iter_responses(
                inp,
                sys_prompt,
                self._messages_for_model(user_msg),
                tools=self.tools.openai_tools(),
                tool_executor=_tool_with_activity,
                inference_profile=inference_profile,
                final_checker=_final_checker,
            ):
                if ev.kind == "intermediate":
                    deliberate_history_extra.extend(ev.history_entries)
                    raw_seg = ev.display_text.strip()
                    if raw_seg and not _intermediate_text_looks_like_tool_channel(
                        raw_seg
                    ):
                        seg = self.voice_ctl.sanitize_reply(raw_seg)
                        if seg and not _reply_too_thin(seg):
                            await _deliver_embodied_deliberate_segment(
                                self,
                                inp,
                                seg,
                                stream=stream,
                                cli_stream_final=False,
                            )
                            if inp.platform in ("discord", "telegram"):
                                intermediate_visible_to_chat = True
                                last_intermediate_chat_segment = seg
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
                reply_text = await self._fallback_plain_reply(inp, tool_state=tool_state)
            reply_text = self.voice_ctl.sanitize_reply(reply_text)
            if _reply_too_thin(reply_text):
                # Sanitization can strip leaked markup/actions and leave empty output.
                log.info("reply_post_sanitize_fallback", module="entity", route=route)
                reply_text = self.voice_ctl.sanitize_reply(
                    await self._fallback_plain_reply(inp, tool_state=tool_state)
                )

            if route in ("reflex", "deliberate"):
                skip_final_chat = (
                    intermediate_visible_to_chat
                    and inp.platform in ("discord", "telegram")
                    and bool((reply_text or "").strip())
                    and (
                        _reply_too_thin(reply_text)
                        or _pro_forma_tool_followup(reply_text)
                    )
                )
                if skip_final_chat and last_intermediate_chat_segment.strip():
                    reply_text = last_intermediate_chat_segment.strip()
                slice_ = self.inner_voice.process(thinking or "", reply_text)
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
                    elif inp.platform in ("discord", "telegram") and not skip_final_chat:
                        await _deliver_embodied_deliberate_segment(
                            self,
                            inp,
                            reply_text,
                            stream=stream,
                            cli_stream_final=False,
                        )
                if route == "deliberate":
                    await self.emotions.process_stimulus(
                        "deep",
                        0.35,
                        {"relationship_warmth": rw},
                    )
                else:
                    low = (reply_text or "").lower()
                    if any(x in low for x in ("ha", "lol", "funny")):
                        await self.emotions.process_stimulus(
                            "positive",
                            0.25,
                            {"relationship_warmth": rw},
                        )
                    elif any(x in low for x in ("sorry", "unfortunately", "can't")):
                        await self.emotions.process_stimulus(
                            "negative",
                            0.15,
                            {"relationship_warmth": rw},
                        )

            if routine_history:
                if record_user_message:
                    self._history.append(
                        {"role": "user", "content": self._history_user_turn_text(inp)},
                    )
                self._history.extend(deliberate_history_extra)
                self._history.append({"role": "assistant", "content": reply_text[:8000]})
                await self._trim_history_with_compression()

            if not skip_drive_interaction:
                self._interaction_count += 1
                if meaningful_override is not None:
                    meaningful = meaningful_override
                else:
                    meaningful = len(inp.text) > 40 or bool(deliberate_history_extra)
                self.drives.on_interaction(meaningful)
            else:
                if meaningful_override is not None:
                    meaningful = meaningful_override
                else:
                    meaningful = len(inp.text) > 40 or bool(deliberate_history_extra)

            thr = self.config.harness.memory.episode_significance_threshold
            fam = rel_existing.familiarity if rel_existing else 0.0
            sig = min(1.0, 0.2 + (0.3 if meaningful else 0) + fam * 0.2)
            if not skip_episode and sig >= thr and meaningful:
                try:
                    _refs = inp.metadata.get("attachment_storage_refs") or []
                    _note = attachment_refs_episode_note(_refs)
                    _base_budget = max(0, 4000 - len(_note))
                    _raw = (inp.text[:_base_budget] + _note)[:4000]
                    ep = await self.episodic.create_from_interaction(
                        db,
                        self.client,
                        summary=f"Conversation with {inp.person_name}: {inp.text[:200]}…",
                        participants=[inp.person_id],
                        imprint=self.emotions.get_state().primary,
                        imprint_i=self.emotions.get_state().intensity,
                        significance=sig,
                        raw_context=_raw,
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

            if not skip_relational_upsert:
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
        self._history_rolling_summary = ""

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
