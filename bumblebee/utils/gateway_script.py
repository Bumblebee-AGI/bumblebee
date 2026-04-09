"""Locate and run ``scripts/gateway.ps1`` (Windows home stack: Ollama + gateway + cloudflared)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import click


def gateway_script_path() -> Path:
    candidates = [
        Path.cwd() / "scripts" / "gateway.ps1",
        Path(__file__).resolve().parent.parent.parent / "scripts" / "gateway.ps1",
    ]
    for p in candidates:
        if p.is_file():
            return p
    checked = ", ".join(str(p) for p in candidates)
    raise click.ClickException(
        f"Could not locate scripts/gateway.ps1. Checked: {checked}. Run from the repo root."
    )


def gateway_script_available() -> bool:
    try:
        gateway_script_path()
    except click.ClickException:
        return False
    return os.name == "nt"


def run_gateway_script(action: str, extra_args: list[str] | None = None) -> int:
    """Run gateway.ps1. Returns process exit code. Raises ClickException on missing script."""
    if os.name != "nt":
        raise click.ClickException("Gateway helper commands are currently supported on Windows only.")
    script = gateway_script_path()
    cmd = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(script),
        action,
    ]
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.call(cmd)
