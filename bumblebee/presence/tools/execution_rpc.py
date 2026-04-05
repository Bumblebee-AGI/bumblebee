"""Shared execution client for dangerous tools (RPC first, guarded local fallback)."""

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
            return await self._call_http(action, payload)
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
    workspace_dir = str(exec_cfg.get("workspace_dir") or "").strip()
    workspace_root = (
        Path(workspace_dir).expanduser().resolve()
        if workspace_dir
        else Path.cwd().resolve()
    )
    mode = (entity.config.harness.deployment.mode or "local").strip().lower()
    allow_local = bool(exec_cfg.get("allow_local", False))
    allow_local_backend = (mode == "hybrid_railway") or allow_local

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
