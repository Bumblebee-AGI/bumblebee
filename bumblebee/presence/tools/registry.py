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
    if tool_name in ("think", "say", "end_turn", "wait"):
        return None
    if tool_name == "search_web":
        return f'🔍 looking up "{args.get("query", "something")}"...'
    if tool_name == "fetch_url":
        domain = extract_domain(str(args.get("url", "") or ""))
        return f"🌐 reading something on {domain}..." if domain else "🌐 reading that page..."
    if tool_name == "read_file":
        filename = Path(str(args.get("path", "") or "")).name
        sl = int(args.get("start_line") or 0)
        el = int(args.get("end_line") or 0)
        if sl > 0 and el > 0 and el != sl:
            return f"📄 reading {filename} lines {sl}–{el}..."
        if sl > 0:
            return f"📄 reading {filename} line {sl}..."
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
        if args.get("as_voice"):
            return f"🗣️ sending a voice message on {platform}..."
        return f"💬 sending a message on {platform}..."
    if tool_name == "send_dm":
        if args.get("list_targets"):
            return "💬 listing DM targets..."
        if args.get("as_voice"):
            return "🗣️ sending a voice DM..."
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
    if tool_name == "send_file":
        name = Path(str(args.get("path", "file") or "file")).name
        return f"📎 sending {name}..."
    if tool_name == "append_file":
        name = Path(str(args.get("path", "file") or "file")).name
        return f"📎 appending to {name}..."
    if tool_name == "get_execution_context":
        return "🧭 checking execution context..."
    if tool_name == "list_checkpoints":
        return "🧷 listing checkpoints..."
    if tool_name == "rollback_checkpoint":
        return "↩️ rolling back checkpoint..."
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
    if tool_name == "desktop_session_status":
        return "🖥️ checking the remote desktop..."
    if tool_name == "desktop_session_view":
        return "🖼️ refreshing the remote desktop view..."
    if tool_name == "desktop_session_type":
        return "⌨️ typing into the remote desktop..."
    if tool_name == "desktop_session_keypress":
        return "⌨️ sending keys to the remote desktop..."
    if tool_name == "desktop_session_click":
        return "🖱️ clicking in the remote desktop..."
    if tool_name == "desktop_session_open_url":
        return "🌐 opening a page on the remote desktop..."
    if tool_name == "desktop_session_stop":
        return "🛑 stopping the remote desktop session..."
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
    if tool_name == "list_skills":
        return "🧠 checking procedural memory..."
    if tool_name == "read_skill":
        return "🧠 reading a skill..."
    if tool_name == "update_skill":
        return "🧠 updating procedural memory..."
    if tool_name == "create_project":
        return "🧵 starting a long-horizon project..."
    if tool_name == "list_projects":
        return "🧵 checking ongoing projects..."
    if tool_name == "update_project":
        return "🧵 updating a project..."
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
        """Append to system prompt: behavioral cues for tool use.

        Tool schemas are passed via the API ``tools`` parameter, NOT duplicated
        in the system prompt.  Gemma-native ``<|tool>`` declarations are omitted
        to avoid the model echoing text-format tool syntax instead of using the
        structured API channel.
        """
        if not self._tools:
            return ""
        names = ", ".join(sorted(self._tools.keys()))
        return (
            "\n\n[Tools available to you — use when curiosity or the moment calls for it, "
            "not as a chore for the user]\n"
            f"You may call: {names}.\n"
            "Treat these as part of your body and memory: execution/context tools are how you inspect "
            "and act in your environment, skills are procedural memory, and projects are long-running "
            "threads you can carry across sessions.\n"
            "If they ask about the workspace, repo contents, files on disk, current machine, or "
            "other host state, do not guess from generic software-project priors. Use the relevant "
            "tool first (usually list_directory, read_file, search_files, or get_system_info), then "
            "answer from that result. If you cannot verify, say so plainly.\n"
            "When you invoke a tool, wait for its result before finishing your reply.\n\n"
            "[Agency — how you communicate and control your process]\n"
            "Use the say tool to send each message the user should see. One call per thought, "
            "short, like texting. If you have two things to share, call say twice. "
            "This is true for conversation AND for sharing tool results. "
            "After your last message, call end_turn.\n"
            "CRITICAL: use the tool calling API to invoke say, think, end_turn, and wait. "
            "Do NOT write their names or arguments as text in your response. "
            "Do NOT write XML tags like <t>, <thought>, <mood>, <say>, or <end_turn>. "
            "These are function calls, not text formatting.\n\n"
            "Your four agency tools:\n"
            "- say: send a message right now. The user sees it immediately. Use for all visible output.\n"
            "- think: private reasoning nobody sees. Use before complex decisions.\n"
            "- end_turn: you're done. Optionally save mood and a parting thought.\n"
            "- wait: pause 1-15s before your next action.\n\n"
            "[Research — how you investigate topics]\n"
            "When researching or answering multi-part questions that need web lookups:\n"
            "- One search at a time. Get results, read them, think about what you found.\n"
            "- Use think() between searches to digest findings and plan the next query.\n"
            "- Share what you've learned so far with say() before moving to the next part.\n"
            "- If results mention a promising source, fetch_url it before moving on.\n"
            "- Let earlier findings shape later searches — build understanding step by step.\n"
            "- Never batch multiple search_web calls in one response. Sequence them across rounds.\n\n"
            "You control when your turn ends. Silence is valid.\n"
            "If a tool fails, don't retry it. Tell the user or try something else.\n"
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
            "Execution/context tools are part of your body, skills are procedural memory, and projects "
            "are long-running commitments you can keep alive across sessions.\n"
            "If they ask what you can do or what you're capable of, call search_tools with an "
            "empty query first, then answer from that in your voice — casual, not a feature list "
            "unless they want detail.\n"
            "If they ask about workspace contents, repo files, or what machine / host you're on, "
            "ground the answer with list_directory, read_file, search_files, or get_system_info "
            "instead of guessing from stereotypes.\n"
            "When you invoke a tool, wait for its result before finishing your reply.\n\n"
            "[Agency — say is how you talk]\n"
            "Use the say tool for every message the user should see. "
            "One call per thought, short, like texting. Then call end_turn. "
            "think: private reasoning. wait: pause. "
            "ALWAYS use the tool calling API — never write tool names as text.\n\n"
            "[Research]\n"
            "One search at a time. Read results, think(), share via say(), then search the next thing. "
            "Never batch multiple search_web calls in one response. Let each result shape the next query.\n"
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
