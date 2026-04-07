"""Shared execution client for dangerous tools (RPC first, guarded local fallback).

Hybrid (`hybrid_railway`) is meant to run the **body** on Railway. Local subprocess/file
fallback is only allowed there (``RAILWAY_ENVIRONMENT``) or when ``tools.execution.allow_local``
is true — so a home workstation with hybrid env vars does not silently become the execution host.

Read-only workspace tools (``read_file``, ``list_directory``, ``search_files``) use the same
``ExecutionRPCClient`` as writes: **RPC first** when ``BUMBLEBEE_EXECUTION_RPC_URL`` /
``tools.execution.base_url`` is set, otherwise local execution under ``tools.execution.workspace_dir``
when ``local_body_host_permitted`` is true. Paths are resolved under the configured workspace root
(like ``write_file``), not arbitrary absolute paths outside it.

``read_pdf`` and ``get_system_info`` still use direct local OS access with the hybrid guard only.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp
import structlog

from bumblebee.presence.tools.runtime import require_tool_runtime

log = structlog.get_logger("bumblebee.presence.tools.execution")

_CLIENTS: dict[str, "ExecutionRPCClient"] = {}
_BG_PROCS: dict[str, dict[str, Any]] = {}


def _rpc_error_suggests_unknown_action(res: dict[str, Any]) -> bool:
    """True when the remote execution service does not implement this action (older hands build)."""
    err = str(res.get("error") or "").lower()
    return "unknown action" in err or "unknown_action" in err


def should_fallback_rpc_to_local(res: dict[str, Any]) -> bool:
    """
    True when the HTTP execution RPC failed in a way that suggests the URL is wrong or down,
    so in-container / local execution is a reasonable fallback (e.g. Railway worker with a
    leftover BUMBLEBEE_EXECUTION_RPC_URL pointing at a home tunnel that is off).
    """
    err = str(res.get("error") or "").lower()
    if "rpc http 401" in err or "rpc http 403" in err:
        return False
    if "rpc http 404" in err or "rpc http 405" in err or "rpc http 422" in err:
        return False
    if "rpc http 500" in err or "rpc http 501" in err:
        return False
    if "rpc http 502" in err or "rpc http 503" in err or "rpc http 504" in err:
        return True
    transport_markers = (
        "cannot connect",
        "connection refused",
        "name or service not known",
        "getaddrinfo failed",
        "temporary failure in name resolution",
        "timed out",
        "timeout",
        "sslcertverificationerror",
        "certificate verify failed",
        "newconnectionerror",
        "clientconnectorerror",
        "connection reset",
        "nodename nor servname",
    )
    return any(m in err for m in transport_markers)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {**base}
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _effective_tools_config(entity: Any) -> dict[str, Any]:
    base = entity.config.harness.tools if isinstance(entity.config.harness.tools, dict) else {}
    over = entity.config.raw.get("tools") if isinstance(entity.config.raw, dict) else {}
    if not isinstance(over, dict):
        over = {}
    return _deep_merge(base, over)


def require_railway_execution(entity: Any) -> bool:
    """
    True when this worker must never execute shell/fs/code on an off-Railway host.

    This still allows in-container execution on Railway itself, and still allows
    HTTP execution RPC to another host when ``BUMBLEBEE_EXECUTION_RPC_URL`` /
    ``tools.execution.base_url`` is configured.
    """
    tools_cfg = _effective_tools_config(entity)
    exec_cfg = tools_cfg.get("execution") if isinstance(tools_cfg.get("execution"), dict) else {}
    if not isinstance(exec_cfg, dict):
        exec_cfg = {}
    raw = (
        (os.environ.get("BUMBLEBEE_EXECUTION_REQUIRE_RAILWAY") or "").strip()
        or str(exec_cfg.get("require_railway") or "").strip()
    )
    return raw.lower() in ("1", "true", "yes", "on")


def _configured_workspace_root(entity: Any) -> Path:
    """Workspace root for local execution fallback / in-container tools."""
    tools_cfg = _effective_tools_config(entity)
    exec_cfg = tools_cfg.get("execution") if isinstance(tools_cfg.get("execution"), dict) else {}
    if not isinstance(exec_cfg, dict):
        exec_cfg = {}
    workspace_dir = (
        (os.environ.get("BUMBLEBEE_EXECUTION_WORKSPACE_DIR") or "").strip()
        or str(exec_cfg.get("workspace_dir") or "").strip()
    )
    return Path(workspace_dir).expanduser().resolve() if workspace_dir else Path.cwd().resolve()


def _tail_text(path: Path, max_bytes: int = 12000) -> str:
    if not path.exists():
        return ""
    try:
        data = path.read_bytes()
    except OSError:
        return ""
    if len(data) > max_bytes:
        data = data[-max_bytes:]
    return data.decode("utf-8", errors="replace")


@dataclass
class ExecutionRPCClient:
    base_url: str
    rpc_path: str
    token: str
    timeout: float
    allow_local_backend: bool
    workspace_root: Path

    async def call(self, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        if self.base_url:
            res = await self._call_http(action, payload)
            if res.get("ok"):
                return res
            if self.allow_local_backend and (
                should_fallback_rpc_to_local(res) or _rpc_error_suggests_unknown_action(res)
            ):
                log.warning(
                    "execution_rpc_unreachable_fallback_local",
                    action=action,
                    error=str(res.get("error") or "")[:300],
                )
                return await asyncio.to_thread(self._call_local, action, payload)
            return res
        if self.allow_local_backend:
            return await asyncio.to_thread(self._call_local, action, payload)
        return {
            "ok": False,
            "error": (
                "Execution backend unavailable. Configure tools.execution.base_url "
                "(or BUMBLEBEE_EXECUTION_RPC_URL), or enable tools.execution.allow_local."
            ),
        }

    async def _call_http(self, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        base = self.base_url.rstrip("/")
        path = self.rpc_path.strip() or "/rpc"
        if not path.startswith("/"):
            path = "/" + path
        url = f"{base}{path}"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        body = {"action": action, "payload": payload or {}}
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=max(5.0, self.timeout)),
            ) as session:
                async with session.post(url, json=body, headers=headers) as resp:
                    txt = await resp.text()
                    try:
                        data = json.loads(txt) if txt else {}
                    except json.JSONDecodeError:
                        data = {"ok": False, "error": txt[:600]}
                    if resp.status >= 400 and "error" not in data:
                        data["error"] = f"rpc http {resp.status}"
                    if "ok" not in data:
                        data["ok"] = resp.status < 400
                    return data
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def _resolve_workspace_path(self, path: str) -> Path:
        p = Path(path).expanduser()
        if p.is_absolute():
            resolved = p.resolve()
        else:
            resolved = (self.workspace_root / p).resolve()
        root = self.workspace_root.resolve()
        if resolved != root and root not in resolved.parents:
            raise RuntimeError(f"path escapes workspace: {resolved}")
        return resolved

    def _call_local(self, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            if action == "run_command":
                cmd = str(payload.get("command") or "").strip()
                timeout = int(payload.get("timeout") or self.timeout)
                if not cmd:
                    return {"ok": False, "error": "empty command"}
                p = subprocess.run(
                    cmd,
                    shell=True,
                    cwd=str(self.workspace_root),
                    capture_output=True,
                    text=True,
                    timeout=max(1, timeout),
                )
                return {
                    "ok": True,
                    "exit_code": int(p.returncode),
                    "stdout": (p.stdout or "")[:20000],
                    "stderr": (p.stderr or "")[:20000],
                }

            if action == "run_background":
                cmd = str(payload.get("command") or "").strip()
                if not cmd:
                    return {"ok": False, "error": "empty command"}
                bg_dir = self.workspace_root / ".bumblebee" / "background"
                bg_dir.mkdir(parents=True, exist_ok=True)
                proc_id = f"bg_{uuid.uuid4().hex[:10]}"
                out_path = bg_dir / f"{proc_id}.out.log"
                err_path = bg_dir / f"{proc_id}.err.log"
                with out_path.open("wb") as out_f, err_path.open("wb") as err_f:
                    proc = subprocess.Popen(  # noqa: S602
                        cmd,
                        shell=True,
                        cwd=str(self.workspace_root),
                        stdout=out_f,
                        stderr=err_f,
                    )
                _BG_PROCS[proc_id] = {
                    "proc": proc,
                    "out": out_path,
                    "err": err_path,
                    "command": cmd,
                    "started_at": time.time(),
                }
                return {"ok": True, "process_id": proc_id, "pid": int(proc.pid)}

            if action == "check_process":
                proc_id = str(payload.get("process_id") or "").strip()
                item = _BG_PROCS.get(proc_id)
                if not item:
                    return {"ok": False, "error": f"unknown process_id: {proc_id}"}
                proc = item["proc"]
                code = proc.poll()
                return {
                    "ok": True,
                    "process_id": proc_id,
                    "running": code is None,
                    "exit_code": None if code is None else int(code),
                    "stdout_tail": _tail_text(item["out"]),
                    "stderr_tail": _tail_text(item["err"]),
                    "command": item["command"],
                }

            if action == "kill_process":
                proc_id = str(payload.get("process_id") or "").strip()
                item = _BG_PROCS.get(proc_id)
                if not item:
                    return {"ok": False, "error": f"unknown process_id: {proc_id}"}
                proc = item["proc"]
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=4)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                return {
                    "ok": True,
                    "process_id": proc_id,
                    "exit_code": None if proc.poll() is None else int(proc.poll()),
                }

            if action == "list_directory":
                raw_path = str(payload.get("path") or ".").strip() or "."
                target = self._resolve_workspace_path(raw_path)
                if not target.exists():
                    return {"ok": False, "error": f"path not found: {target}"}
                if not target.is_dir():
                    return {"ok": False, "error": f"not a directory: {target}"}
                try:
                    out: list[dict[str, str | int]] = []
                    for child in sorted(target.iterdir(), key=lambda c: (not c.is_dir(), c.name.lower()))[:500]:
                        kind = "dir" if child.is_dir() else "file"
                        size = child.stat().st_size if child.is_file() else 0
                        out.append({"name": child.name, "kind": kind, "size": int(size)})
                    return {"ok": True, "path": str(target), "entries": out}
                except OSError as e:
                    return {"ok": False, "error": str(e)}

            if action == "read_file":
                max_bytes = int(payload.get("max_bytes") or 48000)
                max_bytes = max(1024, min(max_bytes, 256_000))
                raw_path = str(payload.get("path") or ".").strip() or "."
                target = self._resolve_workspace_path(raw_path)
                if not target.exists():
                    return {"ok": False, "error": f"path not found: {target}"}
                if not target.is_file():
                    return {"ok": False, "error": f"not a file: {target}"}
                try:
                    def _int_field(key: str) -> int:
                        v = payload.get(key)
                        if v is None or v == "":
                            return 0
                        try:
                            return int(v)
                        except (TypeError, ValueError):
                            return 0

                    start_line = _int_field("start_line")
                    end_line = _int_field("end_line")
                    if start_line > 0:
                        if end_line > 0 and end_line < start_line:
                            return {"ok": False, "error": "end_line must be >= start_line"}
                        end_eff = end_line if end_line >= start_line else start_line
                        max_span = 400
                        if end_eff - start_line + 1 > max_span:
                            return {
                                "ok": False,
                                "error": f"line range too wide (max {max_span} lines per call)",
                            }
                        lines_out: list[tuple[int, str]] = []
                        with target.open("r", encoding="utf-8", errors="replace") as f:
                            for i, line in enumerate(f, start=1):
                                if i > end_eff:
                                    break
                                if i >= start_line:
                                    lines_out.append((i, line.rstrip("\r\n")))
                        if not lines_out:
                            return {
                                "ok": False,
                                "error": f"no lines in range {start_line}-{end_eff} (file may be shorter)",
                            }
                        header = f"--- lines {start_line}-{lines_out[-1][0]} of {target.name} (1-based) ---\n"
                        body = "\n".join(f"{n:6d}|{text}" for n, text in lines_out)
                        return {
                            "ok": True,
                            "path": str(target),
                            "content": header + body,
                            "line_mode": True,
                            "start_line": start_line,
                            "end_line": lines_out[-1][0],
                        }
                    data = target.read_bytes()[:max_bytes]
                    return {
                        "ok": True,
                        "path": str(target),
                        "content": data.decode("utf-8", errors="replace"),
                    }
                except OSError as e:
                    return {"ok": False, "error": str(e)}

            if action == "search_files":
                directory = str(payload.get("directory") or ".").strip() or "."
                pattern = str(payload.get("pattern") or "").strip() or "*"
                d = self._resolve_workspace_path(directory)
                if not d.exists() or not d.is_dir():
                    return {"ok": False, "error": f"directory not found: {d}"}
                try:
                    matches = [str(p) for p in d.rglob(pattern) if p.is_file()][:1000]
                    return {"ok": True, "directory": str(d), "pattern": pattern, "matches": matches}
                except OSError as e:
                    return {"ok": False, "error": str(e)}

            if action == "write_file":
                target = self._resolve_workspace_path(str(payload.get("path") or ""))
                content = str(payload.get("content") or "")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
                return {"ok": True, "path": str(target), "bytes": len(content.encode("utf-8"))}

            if action == "append_file":
                target = self._resolve_workspace_path(str(payload.get("path") or ""))
                content = str(payload.get("content") or "")
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("a", encoding="utf-8") as f:
                    f.write(content)
                return {"ok": True, "path": str(target), "bytes": len(content.encode("utf-8"))}

            if action == "execute_python":
                code = str(payload.get("code") or "")
                timeout = int(payload.get("timeout") or self.timeout)
                if not code.strip():
                    return {"ok": False, "error": "empty code"}
                tmp = Path(tempfile.gettempdir()) / f"bb_py_{uuid.uuid4().hex[:8]}.py"
                tmp.write_text(code, encoding="utf-8")
                try:
                    p = subprocess.run(
                        [sys.executable, str(tmp)],
                        cwd=str(self.workspace_root),
                        capture_output=True,
                        text=True,
                        timeout=max(1, timeout),
                    )
                    return {
                        "ok": True,
                        "exit_code": int(p.returncode),
                        "stdout": (p.stdout or "")[:20000],
                        "stderr": (p.stderr or "")[:20000],
                    }
                finally:
                    try:
                        tmp.unlink(missing_ok=True)
                    except OSError:
                        pass

            if action == "execute_javascript":
                code = str(payload.get("code") or "")
                timeout = int(payload.get("timeout") or self.timeout)
                if not code.strip():
                    return {"ok": False, "error": "empty code"}
                node = shutil.which("node")
                if not node:
                    return {"ok": False, "error": "node not available in execution environment"}
                tmp = Path(tempfile.gettempdir()) / f"bb_js_{uuid.uuid4().hex[:8]}.js"
                tmp.write_text(code, encoding="utf-8")
                try:
                    p = subprocess.run(
                        [node, str(tmp)],
                        cwd=str(self.workspace_root),
                        capture_output=True,
                        text=True,
                        timeout=max(1, timeout),
                    )
                    return {
                        "ok": True,
                        "exit_code": int(p.returncode),
                        "stdout": (p.stdout or "")[:20000],
                        "stderr": (p.stderr or "")[:20000],
                    }
                finally:
                    try:
                        tmp.unlink(missing_ok=True)
                    except OSError:
                        pass

            if action in (
                "browser_navigate",
                "browser_screenshot",
                "browser_click",
                "browser_type",
                "generate_image",
            ):
                return {
                    "ok": False,
                    "error": (
                        f"{action} requires an RPC execution backend with browser/image capabilities. "
                        "Set tools.execution.base_url."
                    ),
                }

            return {"ok": False, "error": f"unknown action: {action}"}
        except subprocess.TimeoutExpired as e:
            return {"ok": False, "error": f"timeout: {e}"}
        except Exception as e:
            log.warning("execution_local_failed", action=action, error=str(e))
            return {"ok": False, "error": str(e)}


HYBRID_OFF_RAILWAY_TOOL_BLOCK = (
    "Tool disabled: hybrid_railway on a host without RAILWAY_ENVIRONMENT (this Python process is not "
    "the Railway container). Run the worker on Railway, or set tools.execution.allow_local in entity YAML "
    "for local debugging. For file tools without touching this PC, set BUMBLEBEE_EXECUTION_RPC_URL (or "
    "tools.execution.base_url) to a hands host that implements the execution RPC. If you use Telegram and "
    "only wanted the cloud bot, stop any local `bumblebee run` that uses the same TELEGRAM_TOKEN."
)

RAILWAY_REQUIRED_TOOL_BLOCK = (
    "Tool disabled: execution is locked to Railway. This Python process is not running inside the "
    "Railway container, so it may not use local disk/shell/code on this machine. Run the worker on "
    "Railway, or set BUMBLEBEE_EXECUTION_RPC_URL (or tools.execution.base_url) to a remote execution "
    "host. To relax this for local debugging, unset BUMBLEBEE_EXECUTION_REQUIRE_RAILWAY (or set "
    "tools.execution.require_railway: false)."
)


def execution_rpc_url_configured(entity: Any) -> bool:
    """True when an HTTP execution backend URL is configured (env or entity tools.execution)."""
    tools_cfg = _effective_tools_config(entity)
    exec_cfg = tools_cfg.get("execution") if isinstance(tools_cfg.get("execution"), dict) else {}
    if not isinstance(exec_cfg, dict):
        exec_cfg = {}
    base = (
        (os.environ.get("BUMBLEBEE_EXECUTION_RPC_URL") or "").strip()
        or str(exec_cfg.get("base_url") or "").strip()
    )
    return bool(base.rstrip("/"))


def local_tool_block_message(entity: Any) -> str:
    """Why local disk/shell/code is blocked for this process."""
    if require_railway_execution(entity) and not (
        (os.environ.get("RAILWAY_ENVIRONMENT") or "").strip()
    ):
        return RAILWAY_REQUIRED_TOOL_BLOCK
    return HYBRID_OFF_RAILWAY_TOOL_BLOCK


def read_only_workspace_fs_allowed(entity: Any) -> bool:
    """True when read-only workspace file tools may run (local/Railway body or RPC URL set)."""
    return local_body_host_permitted(entity) or execution_rpc_url_configured(entity)


def local_body_host_permitted(entity: Any) -> bool:
    """True if this process may use local disk/shell for tools (not RPC-only hybrid on a dev PC)."""
    tools_cfg = _effective_tools_config(entity)
    exec_cfg = tools_cfg.get("execution") if isinstance(tools_cfg.get("execution"), dict) else {}
    if not isinstance(exec_cfg, dict):
        exec_cfg = {}
    mode = (entity.config.harness.deployment.mode or "local").strip().lower()
    allow_local = bool(exec_cfg.get("allow_local", False))
    on_railway = bool((os.environ.get("RAILWAY_ENVIRONMENT") or "").strip())
    if require_railway_execution(entity) and not on_railway:
        return False
    return (
        allow_local
        or mode == "local"
        or (mode == "hybrid_railway" and on_railway)
    )


def get_execution_client() -> ExecutionRPCClient:
    ctx = require_tool_runtime()
    entity = ctx.entity
    key = f"{entity.config.name}:{id(entity)}"
    cached = _CLIENTS.get(key)
    if cached is not None:
        return cached

    tools_cfg = _effective_tools_config(entity)
    exec_cfg = tools_cfg.get("execution") if isinstance(tools_cfg.get("execution"), dict) else {}
    if not isinstance(exec_cfg, dict):
        exec_cfg = {}
    base_url = (
        (os.environ.get("BUMBLEBEE_EXECUTION_RPC_URL") or "").strip()
        or str(exec_cfg.get("base_url") or "").strip()
    ).rstrip("/")
    token_env = str(exec_cfg.get("token_env") or "BUMBLEBEE_EXECUTION_RPC_TOKEN")
    token = (os.environ.get(token_env) or "").strip()
    timeout = float(exec_cfg.get("timeout") or 45.0)
    rpc_path = str(exec_cfg.get("rpc_path") or "/rpc")
    workspace_root = _configured_workspace_root(entity)
    allow_local_backend = local_body_host_permitted(entity)

    client = ExecutionRPCClient(
        base_url=base_url,
        rpc_path=rpc_path,
        token=token,
        timeout=timeout,
        allow_local_backend=allow_local_backend,
        workspace_root=workspace_root,
    )
    _CLIENTS[key] = client
    return client
