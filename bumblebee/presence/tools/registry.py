"""Gemma/OpenAI-style tool registration, schema generation, and execution."""

from __future__ import annotations

import inspect
import json
from collections import Counter
from pathlib import Path
from typing import Any, Awaitable, Callable
from urllib.parse import urlparse

from bumblebee.cognition import gemma
from bumblebee.utils.ollama_client import ToolCallSpec

# Used to split system prompt from the tools appendix (see DeliberateCognition._build_messages).
TOOL_SYSTEM_PROMPT_PREFIX = "\n\n[Tools available to you"
TOOL_SYSTEM_PROMPT_PREFIX_COMPACT = "\n\n[Tools — callable via"


def extract_domain(url: str) -> str:
    s = (url or "").strip()
    if not s:
        return ""
    if "://" not in s:
        s = f"https://{s}"
    try:
        parsed = urlparse(s)
        host = parsed.hostname or ""
    except ValueError:
        return ""
    if not host:
        return ""
    if host.lower().startswith("www."):
        host = host[4:]
    return host


def format_tool_activity(tool_name: str, args: dict[str, Any]) -> str | None:
    if tool_name == "search_web":
        return f'🔍 looking up "{args.get("query", "something")}"...'
    if tool_name == "fetch_url":
        domain = extract_domain(str(args.get("url", "") or ""))
        return f"🌐 reading something on {domain}..." if domain else "🌐 reading that page..."
    if tool_name == "read_file":
        filename = Path(str(args.get("path", "") or "")).name
        return f"📄 checking {filename}..."
    if tool_name == "get_current_time":
        return "🕐 checking the time..."
    if tool_name == "get_youtube_transcript":
        return "▶️ pulling transcript..."
    if tool_name == "search_youtube":
        return f'📺 searching youtube for "{args.get("query", "something")}"...'
    if tool_name in ("read_reddit", "read_reddit_post"):
        sub = str(args.get("subreddit", "") or "").strip()
        return f"🔶 browsing r/{sub}..." if sub else "🔶 reading reddit..."
    if tool_name == "read_wikipedia":
        return f'📚 reading about {args.get("topic", "something")}...'
    if tool_name == "get_weather":
        return f'🌤️ checking weather in {args.get("location", "somewhere")}...'
    if tool_name == "get_news":
        topic = str(args.get("topic", "") or "").strip()
        return f"📰 checking news about {topic}..." if topic else "📰 checking the news..."
    if tool_name == "read_pdf":
        return "📑 reading pdf..."
    if tool_name == "speak":
        return "🗣️ recording a voice message..."
    if tool_name == "get_tts_voice":
        return "🗣️ checking current voice..."
    if tool_name == "list_tts_voices":
        return "🗣️ browsing available voices..."
    if tool_name == "set_tts_voice":
        return "🗣️ switching voice..."
    if tool_name == "set_reminder":
        return "⏰ setting a reminder..."
    if tool_name == "list_reminders":
        return None
    if tool_name == "cancel_reminder":
        return "❌ canceling reminder..."
    if tool_name == "send_message_to":
        platform = str(args.get("platform", "somewhere") or "somewhere")
        return f"💬 sending a message on {platform}..."
    if tool_name == "send_dm":
        if args.get("list_targets"):
            return "💬 listing DM targets..."
        return "💬 sending a direct message..."
    if tool_name == "get_system_info":
        return "💻 checking system stats..."
    if tool_name == "run_command":
        cmd = str(args.get("command", "something") or "something")
        short = cmd[:40] + "..." if len(cmd) > 40 else cmd
        return f"⚡ running: {short}"
    if tool_name == "run_background":
        return "🔄 starting background process..."
    if tool_name == "check_process":
        return "📊 checking process..."
    if tool_name == "kill_process":
        return "🛑 stopping process..."
    if tool_name == "list_directory":
        return "📁 looking around..."
    if tool_name == "search_files":
        return f'🔎 searching for {args.get("pattern", "files")}...'
    if tool_name == "write_file":
        name = Path(str(args.get("path", "file") or "file")).name
        return f"✏️ writing {name}..."
    if tool_name == "append_file":
        name = Path(str(args.get("path", "file") or "file")).name
        return f"📎 appending to {name}..."
    if tool_name == "execute_python":
        return "🐍 running python..."
    if tool_name == "execute_javascript":
        return "📜 running javascript..."
    if tool_name == "browser_navigate":
        domain = extract_domain(str(args.get("url", "") or ""))
        return f"🌐 opening {domain}..." if domain else "🌐 opening page..."
    if tool_name == "browser_screenshot":
        return "📸 taking a screenshot..."
    if tool_name == "browser_click":
        return "👆 clicking..."
    if tool_name == "browser_type":
        return "⌨️ typing..."
    if tool_name == "generate_image":
        return "🎨 creating an image..."
    if tool_name == "update_knowledge":
        action = str(args.get("action", "") or "").lower()
        section = str(args.get("section", "") or "something")
        emoji = {"add": "📌", "update": "✏️", "remove": "🗑️"}.get(action, "📝")
        verb = {"add": "adding", "update": "updating", "remove": "removing"}.get(
            action, "editing"
        )
        return f"{emoji} {verb} knowledge: {section}..."
    if tool_name == "search_tools":
        q = str(args.get("query", "") or "").strip()
        return f'🧰 searching tools for "{q}"...' if q else "🧰 listing available tools..."
    if tool_name == "describe_tool":
        n = str(args.get("tool_name", "") or "").strip()
        return f"🧰 inspecting tool {n}..." if n else "🧰 inspecting tool..."
    if tool_name == "create_automation":
        return f'⏰ setting up routine: {args.get("name", "something")}...'
    if tool_name == "list_automations":
        return None
    if tool_name == "edit_automation":
        return "✏️ updating routine..."
    if tool_name == "toggle_automation":
        enabled = args.get("enabled", True)
        return f'{"✅" if enabled else "⏸️"} {"enabling" if enabled else "pausing"} routine...'
    if tool_name == "delete_automation":
        return "🗑️ removing routine..."
    if tool_name == "run_automation_now":
        return "▶️ running routine now..."
    if tool_name == "write_journal":
        return "📓 writing in journal..."
    if tool_name == "read_journal":
        return None
    if tool_name.startswith("mcp_"):
        parts = tool_name.split("_", 2)
        server = (parts[1] if len(parts) >= 2 else "mcp").lower()
        emoji_map = {
            "github": "🐙",
            "spotify": "🎵",
            "slack": "💬",
            "notion": "📝",
            "calendar": "📅",
            "gmail": "📧",
            "maps": "📍",
            "weather": "🌤️",
        }
        emoji = emoji_map.get(server, "🔌")
        return f"{emoji} asking {server}..."
    return None


class ToolFn:
    def __init__(self, name: str, description: str, fn: Callable[..., Awaitable[str]]):
        self.name = name
        self.description = description
        self.fn = fn
        self.schema = self._build_schema(fn)

    def _build_schema(self, fn: Callable[..., Any]) -> dict[str, Any]:
        sig = inspect.signature(fn)
        props: dict[str, Any] = {}
        required: list[str] = []
        for pname, p in sig.parameters.items():
            if pname == "self":
                continue
            ann = p.annotation if p.annotation != inspect.Parameter.empty else str
            json_type = "string"
            if ann is int:
                json_type = "integer"
            elif ann is float:
                json_type = "number"
            elif ann is bool:
                json_type = "boolean"
            props[pname] = {"type": json_type}
            if p.default == inspect.Parameter.empty:
                required.append(pname)
        return {
            "type": "object",
            "properties": props,
            "required": required,
        }

    def openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.schema,
            },
        }


def tool(name: str, description: str) -> Callable[[Callable[..., Awaitable[str]]], Callable[..., Awaitable[str]]]:
    """Decorator: define an async tool; register it on a registry via ``registry.register_decorated``."""

    def deco(fn: Callable[..., Awaitable[str]]) -> Callable[..., Awaitable[str]]:
        setattr(fn, "_bumblebee_tool_name", name)
        setattr(fn, "_bumblebee_tool_description", description)
        return fn

    return deco


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolFn] = {}
        self._call_counts: Counter[str] = Counter()

    def register(self, t: ToolFn) -> None:
        self._tools[t.name] = t

    def register_fn(
        self,
        name: str,
        description: str,
        fn: Callable[..., Awaitable[str]],
        *,
        parameters_schema: dict[str, Any] | None = None,
    ) -> None:
        t = ToolFn(name, description, fn)
        if parameters_schema and isinstance(parameters_schema, dict):
            t.schema = parameters_schema
        self.register(t)

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def register_decorated(self, fn: Callable[..., Awaitable[str]]) -> None:
        n = getattr(fn, "_bumblebee_tool_name", None)
        d = getattr(fn, "_bumblebee_tool_description", None)
        if not n or not d:
            raise ValueError("Function missing @tool metadata")
        self.register_fn(str(n), str(d), fn)

    def openai_tools(self) -> list[dict[str, Any]]:
        return [t.openai_tool() for t in self._tools.values()]

    def list_tools(self) -> list[tuple[str, str]]:
        """Return (name, description) for currently-registered tools."""
        return sorted(
            [(t.name, t.description) for t in self._tools.values()],
            key=lambda row: row[0],
        )

    def tool_discovery_detail(self, name: str) -> dict[str, Any] | None:
        """OpenAI-style function payload (name, description, parameters) or ``None``."""
        t = self._tools.get(name)
        if t is None:
            return None
        wrapped = t.openai_tool()
        fn = wrapped.get("function")
        return dict(fn) if isinstance(fn, dict) else None

    def tool_discovery_summaries(self) -> list[dict[str, Any]]:
        """Compact rows for search/listing (name, description, parameter names)."""
        rows: list[dict[str, Any]] = []
        for t in sorted(self._tools.values(), key=lambda x: x.name):
            props = t.schema.get("properties") if isinstance(t.schema, dict) else None
            keys = list(props.keys()) if isinstance(props, dict) else []
            req = t.schema.get("required") if isinstance(t.schema, dict) else None
            rows.append(
                {
                    "name": t.name,
                    "description": t.description,
                    "parameters": keys,
                    "required": list(req) if isinstance(req, list) else [],
                }
            )
        return rows

    def resolve_tool_name(self, raw: str) -> str | None:
        """Exact or case-insensitive match to a registered tool name."""
        s = (raw or "").strip()
        if not s:
            return None
        if s in self._tools:
            return s
        low = s.casefold()
        for name in self._tools:
            if name.casefold() == low:
                return name
        return None

    def suggest_tool_names(self, substring: str, limit: int = 8) -> list[str]:
        """Registered names containing substring (case-insensitive)."""
        sub = (substring or "").strip().casefold()
        if not sub:
            return []
        lim = max(1, min(int(limit or 8), 32))
        return [n for n in sorted(self._tools.keys()) if sub in n.casefold()][:lim]

    def gemma_tool_declarations(self) -> str:
        return gemma.format_tool_declarations_block(self.openai_tools())

    def system_tool_instruction_block(self) -> str:
        """Append to system prompt: native declarations + short usage cue."""
        decl = self.gemma_tool_declarations()
        if not decl:
            return ""
        names = ", ".join(sorted(self._tools.keys()))
        return (
            "\n\n[Tools available to you — use when curiosity or the moment calls for it, "
            "not as a chore for the user]\n"
            f"You may call: {names}.\n"
            "When you invoke a tool, wait for its result before finishing your reply.\n\n"
            f"{decl}"
        )

    def compact_system_tool_instruction(self) -> str:
        """
        Short tools appendix when the full Gemma declaration block would overflow
        ``system_prompt_char_limit``. OpenAI-compatible backends already receive
        full schemas via the ``tools`` request field; this keeps behavioral cues
        and the name list in the system string.
        """
        if not self._tools:
            return ""
        names = ", ".join(sorted(self._tools.keys()))
        return (
            "\n\n[Tools — callable via the tool channel; full parameter schemas are in the API]\n"
            f"You may call: {names}.\n"
            "If they ask what you can do or what you're capable of, call search_tools with an "
            "empty query first, then answer from that in your voice — casual, not a feature list "
            "unless they want detail.\n"
            "When you invoke a tool, wait for its result before finishing your reply.\n"
        )

    def usage_snapshot(self) -> dict[str, int]:
        return dict(self._call_counts)

    async def execute(self, spec: ToolCallSpec) -> str:
        fn = self._tools.get(spec.name)
        if not fn:
            return json.dumps({"error": f"unknown tool {spec.name}"})
        try:
            kwargs = dict(spec.arguments)
            out = await fn.fn(**kwargs)
            self._call_counts[spec.name] += 1
            return out
        except TypeError:
            try:
                out = await fn.fn()
                self._call_counts[spec.name] += 1
                return out
            except Exception as e:
                return json.dumps({"error": str(e)})
        except Exception as e:
            return json.dumps({"error": str(e)})


def wrap_simple(coro_fn: Callable[..., Awaitable[str]]) -> ToolFn:
    return ToolFn(coro_fn.__name__, coro_fn.__doc__ or "", coro_fn)
