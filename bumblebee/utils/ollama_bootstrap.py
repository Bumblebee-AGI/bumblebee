"""Ensure local Ollama is running; optional explicit model pull for first-time setup."""

from __future__ import annotations

import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable

import click

from bumblebee.config import EntityConfig


def _is_local_ollama(base_url: str) -> bool:
    u = base_url.lower().rstrip("/")
    return "localhost" in u or "127.0.0.1" in u or "0.0.0.0" in u


def _tags_reachable(base_url: str, timeout_s: float = 2.0) -> bool:
    url = f"{base_url.rstrip('/')}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            code = resp.getcode()
            return 200 <= code < 300
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


def _wait_ready(base_url: str, *, timeout_s: float = 120.0, interval_s: float = 0.5) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if _tags_reachable(base_url):
            return True
        time.sleep(interval_s)
    return False


def _spawn_ollama_serve() -> None:
    kwargs: dict = {
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "stdin": subprocess.DEVNULL,
    }
    if sys.platform == "win32":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    else:
        kwargs["start_new_session"] = True
    subprocess.Popen(["ollama", "serve"], **kwargs)


def _required_model_names(ec: EntityConfig) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for m in (
        ec.cognition.reflex_model,
        ec.cognition.deliberate_model,
        ec.harness.models.deliberate,
        ec.harness.models.embedding,
    ):
        if m and m not in seen:
            seen.add(m)
            out.append(m)
    return out


def pull_required_models(ec: EntityConfig, info: Callable[[str], None] = print) -> None:
    """Run ``ollama pull`` for each model in config (use when setting up a new machine)."""
    _pull_models_impl(ec, info)


def _pull_models_impl(ec: EntityConfig, info: Callable[[str], None]) -> None:
    models = _required_model_names(ec)
    info(f"Pulling models: {', '.join(models)}")
    for m in models:
        info(f"  → pull {m}")
        r = subprocess.run(["ollama", "pull", m], check=False)
        if r.returncode != 0:
            raise click.ClickException(
                f"`ollama pull {m}` failed (exit {r.returncode}). Fix the error above and retry."
            )
    info("Pull complete.")


def bootstrap_stack(
    ec: EntityConfig,
    info: Callable[[str], None] = print,
    *,
    pull_models: bool = False,
) -> None:
    """
    If Ollama base URL looks local and the API is down, start `ollama serve` and wait.
    Does not pull models unless ``pull_models=True`` (Bumblebee assumes models already exist).
    """
    base = ec.harness.ollama.base_url.rstrip("/")

    if _is_local_ollama(base):
        if _tags_reachable(base):
            info("Ollama is already running.")
        else:
            info("Ollama not reachable — starting `ollama serve` in the background…")
            try:
                _spawn_ollama_serve()
            except FileNotFoundError as e:
                raise click.ClickException(
                    "Could not run `ollama serve`: is Ollama installed and on your PATH?"
                ) from e
            if not _wait_ready(base):
                raise click.ClickException(
                    "Ollama did not become ready in time. Check that port 11434 is free and "
                    "`ollama` works in a terminal."
                )
            info("Ollama is up.")
        if pull_models:
            _pull_models_impl(ec, info)
        return

    info(f"Using remote Ollama at {base} — not starting a local server.")
    if not _tags_reachable(base, timeout_s=5.0):
        raise click.ClickException(
            f"Cannot reach Ollama at {base}. Start it or fix harness.ollama.base_url."
        )
    if pull_models:
        info("Skipping `ollama pull` (remote server — install models on that host).")
