"""CLI entry: create, run, talk, status, evolve, recall, export, import."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
from importlib import metadata
from pathlib import Path

import click

from bumblebee.config import (
    load_entity_config,
    load_harness_config,
    project_configs_dir,
    validate_entity_env,
)
from bumblebee.entity import Entity
from bumblebee.genesis import creator
from bumblebee.presence.daemon import PresenceDaemon
from bumblebee.presence.platforms.cli import CLIPlatform
from bumblebee.presence.platforms.discord_platform import DiscordPlatform, token_from_config
from bumblebee.presence.platforms.telegram_platform import TelegramPlatform
from bumblebee.utils.log import setup_logging
from bumblebee.utils.ollama_bootstrap import bootstrap_stack


def _async(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _app_version() -> str:
    try:
        return metadata.version("bumblebee")
    except metadata.PackageNotFoundError:
        return "0.1.0"


def _apply_dotenv_file(path: Path) -> None:
    """Set os.environ keys from a single .env file.

    Non-empty existing values are kept; missing or blank entries are filled from the file.
    """
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip().removeprefix("\ufeff")
        val = val.strip().strip('"').strip("'")
        if not key:
            continue
        # Fill when unset or empty so a stale Windows env entry can't block `.env`.
        if (os.environ.get(key) or "").strip():
            continue
        os.environ[key] = val


def _load_repo_dotenv() -> None:
    """Load `.env` from cwd first, then package parent (repo root when running from source).

    A normal (non-editable) install puts code under site-packages; ``parent.parent`` then
    is not your project folder, so tokens in the repo ``.env`` are missed unless we also
    read ``Path.cwd() / ".env"``.
    """
    _apply_dotenv_file(Path.cwd() / ".env")
    pkg_root = Path(__file__).resolve().parent.parent
    _apply_dotenv_file(pkg_root / ".env")


async def _wait_until_stopped() -> None:
    """Block until SIGINT (Ctrl+C). Works on Windows where Event().wait() alone may not unblock."""
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()

    def _schedule_stop(*_args: object) -> None:
        try:
            loop.call_soon_threadsafe(stop.set)
        except RuntimeError:
            pass

    prev: object | None = None
    used_asyncio_handler = False
    try:
        loop.add_signal_handler(signal.SIGINT, _schedule_stop)
        used_asyncio_handler = True
    except (NotImplementedError, RuntimeError):
        prev = signal.signal(signal.SIGINT, _schedule_stop)
    try:
        await stop.wait()
    finally:
        if used_asyncio_handler:
            try:
                loop.remove_signal_handler(signal.SIGINT)
            except (NotImplementedError, RuntimeError):
                pass
        elif prev is not None:
            signal.signal(signal.SIGINT, prev)


def _prepare_ollama_cli(entity_name: str, with_ollama: bool, pull_models: bool) -> None:
    if not with_ollama and not pull_models:
        return
    harness = load_harness_config()
    ec = load_entity_config(entity_name, harness)
    if with_ollama:
        bootstrap_stack(ec, info=click.echo, pull_models=pull_models)
    else:
        from bumblebee.utils.ollama_bootstrap import pull_required_models

        pull_required_models(ec, click.echo)


@click.group()
def cli() -> None:
    # Runs before every subcommand; keep in sync with ``main()`` for entry points that call ``cli`` only.
    _load_repo_dotenv()


@cli.command("create")
def cmd_create() -> None:
    """Launch interactive entity wizard."""
    creator.run_wizard()


@cli.command("talk")
@click.argument("entity_name")
@click.option(
    "--ollama",
    "with_ollama",
    is_flag=True,
    help="Start local Ollama if needed (does not pull models; use --pull-models to fetch).",
)
@click.option(
    "--pull-models",
    is_flag=True,
    help="Run ollama pull for configured models (optional; combine with --ollama).",
)
def cmd_talk(entity_name: str, with_ollama: bool, pull_models: bool) -> None:
    """CLI-only conversation (no daemon, no other platforms)."""
    _prepare_ollama_cli(entity_name, with_ollama, pull_models)
    try:
        asyncio.run(_talk(entity_name))
    except KeyboardInterrupt:
        click.echo("\nStopped.", err=True)


async def _talk(entity_name: str) -> None:
    harness = load_harness_config()
    ec = load_entity_config(entity_name, harness)
    setup_logging(
        ec.name,
        ec.harness.logging.level,
        ec.log_path(),
        ec.harness.logging.format,
    )
    ent = Entity(ec)
    cli_p = CLIPlatform(ent, app_version=_app_version(), immersive=True)
    await cli_p.connect()

    async def on_inp(inp):
        return await ent.perceive(inp, stream=cli_p.stream_delta)

    await cli_p.on_message(on_inp)
    try:
        await cli_p.run_repl()
    finally:
        await ent.shutdown()


@cli.command("run")
@click.argument("entity_name")
@click.option(
    "--ollama",
    "with_ollama",
    is_flag=True,
    help="Start local Ollama if needed (does not pull models; use --pull-models to fetch).",
)
@click.option(
    "--pull-models",
    is_flag=True,
    help="Run ollama pull for configured models (optional; combine with --ollama).",
)
def cmd_run(entity_name: str, with_ollama: bool, pull_models: bool) -> None:
    """Start entity with daemon and configured platforms."""
    _prepare_ollama_cli(entity_name, with_ollama, pull_models)
    try:
        asyncio.run(_run(entity_name))
    except KeyboardInterrupt:
        click.echo("\nStopped.", err=True)


async def _run(entity_name: str) -> None:
    harness = load_harness_config()
    ec = load_entity_config(entity_name, harness)
    for w in validate_entity_env(ec):
        click.echo(f"Warning: {w}", err=True)
    setup_logging(
        ec.name,
        ec.harness.logging.level,
        ec.log_path(),
        ec.harness.logging.format,
    )
    ent = Entity(ec)
    ent.register_proactive_sink(lambda m: click.echo(f"\n[{ec.name}]: {m}\n"))

    daemon = PresenceDaemon(ent)

    platforms: list = []
    discord_p: DiscordPlatform | None = None
    telegram_p: TelegramPlatform | None = None

    for pl in ec.presence.platforms:
        t = (pl.get("type") or "").lower()
        if t == "discord":
            tok = token_from_config(pl.get("token_env", "DISCORD_TOKEN"))
            if not tok:
                click.echo("Discord token missing; skipping Discord.", err=True)
                continue
            chans = pl.get("channels") or []
            discord_p = DiscordPlatform(tok, [str(c) for c in chans])
            platforms.append(discord_p)
        elif t == "telegram":
            tok = token_from_config(pl.get("token_env", "TELEGRAM_TOKEN"))
            if not tok:
                click.echo("Telegram token missing; skipping Telegram.", err=True)
                continue
            raw_allow = pl.get("allowed_user_ids")
            allow_set: set[int] | None = None
            if isinstance(raw_allow, list) and len(raw_allow) > 0:
                allow_set = {int(x) for x in raw_allow}
            telegram_p = TelegramPlatform(
                tok,
                entity=ent,
                app_version=_app_version(),
                allowed_user_ids=allow_set,
            )
            platforms.append(telegram_p)

    cli_p = CLIPlatform(ent, app_version=_app_version(), immersive=False)
    has_cli = any((p.get("type") or "").lower() == "cli" for p in ec.presence.platforms)

    async def route_reply(inp, text: str):
        meta = ent.voice_ctl.meta_for_response(
            ent.emotions.get_state(),
            len(text),
            inp.platform,
        )
        delay = min(3.0, meta.typing_delay_seconds)
        if inp.platform == "telegram" and telegram_p:
            await telegram_p.send_typing(int(inp.channel))
            await asyncio.sleep(delay)
        else:
            await asyncio.sleep(delay)
        if inp.platform == "discord" and discord_p:
            await discord_p.send_message(inp.channel, text)
        elif inp.platform == "telegram" and telegram_p:
            await telegram_p.send_message(inp.channel, text)

    async def _telegram_typing_worker(chat_id: int, stop: asyncio.Event) -> None:
        while not stop.is_set():
            await telegram_p.send_typing(chat_id)
            try:
                await asyncio.wait_for(stop.wait(), timeout=4.5)
            except asyncio.TimeoutError:
                continue

    async def on_inp(inp):
        stream = cli_p.stream_delta if inp.platform == "cli" else None
        stop_typing = asyncio.Event()
        typing_task: asyncio.Task[None] | None = None
        if inp.platform == "telegram" and telegram_p:
            typing_task = asyncio.create_task(
                _telegram_typing_worker(int(inp.channel), stop_typing)
            )
        try:
            reply = await ent.perceive(inp, stream=stream)
        finally:
            stop_typing.set()
            if typing_task is not None:
                typing_task.cancel()
                try:
                    await typing_task
                except asyncio.CancelledError:
                    pass
        if inp.platform == "cli":
            return reply
        try:
            await route_reply(inp, reply)
        except Exception as e:
            click.echo(f"Error: {e}", err=True)
        return ""

    for p in platforms:
        await p.on_message(on_inp)
        await p.connect()

    # After messaging platforms are live — avoids aiosqlite + PTB fighting the loop at startup.
    await daemon.start()

    if has_cli:
        await cli_p.connect()
        await cli_p.on_message(on_inp)
        try:
            await cli_p.run_repl()
        finally:
            await daemon.stop()
            for p in platforms:
                await p.disconnect()
            await ent.shutdown()
    else:
        click.echo(f"{ec.name} running (no CLI). Ctrl+C to stop.")
        try:
            await _wait_until_stopped()
        except asyncio.CancelledError:
            pass
        finally:
            await daemon.stop()
            for p in platforms:
                await p.disconnect()
            await ent.shutdown()


@cli.command("status")
@click.argument("entity_name")
def cmd_status(entity_name: str) -> None:
    """Show emotional state, drives, paths."""
    harness = load_harness_config()
    ec = load_entity_config(entity_name, harness)
    setup_logging(ec.name, ec.harness.logging.level, None, "console")
    ent = Entity(ec)

    async def _show():
        s = ent.emotions.get_state()
        click.echo(f"Entity: {ec.name}")
        click.echo(f"Emotion: {s.primary.value} ({s.intensity:.2f})")
        click.echo("Drives:")
        for d in ent.drives.all_drives():
            click.echo(f"  {d.name}: {d.level:.2f} (threshold {d.threshold})")
        click.echo(f"DB: {ec.db_path()}")
        click.echo(f"Dormant: {ent.dormant}")
        await ent.shutdown()

    asyncio.run(_show())


@cli.command("evolve")
@click.argument("entity_name")
def cmd_evolve(entity_name: str) -> None:
    """Force one trait evolution cycle."""
    harness = load_harness_config()
    ec = load_entity_config(entity_name, harness)
    setup_logging(ec.name, ec.harness.logging.level, None, "console")
    ent = Entity(ec)

    async def _go():
        c = await ent.evolution.run_cycle(
            {"anxiety_trend": 0.15, "deep_conversations": 5}
        )
        click.echo(json.dumps(c, indent=2))
        await ent.shutdown()

    asyncio.run(_go())


@cli.command("recall")
@click.argument("entity_name")
@click.argument("query")
def cmd_recall(entity_name: str, query: str) -> None:
    """Search episodic memories by semantic similarity."""
    harness = load_harness_config()
    ec = load_entity_config(entity_name, harness)
    ent = Entity(ec)

    async def _recall():
        db = await ent.store.connect()
        try:
            qe = await ent.client.embed(ec.harness.models.embedding, query)
            if not qe:
                click.echo("Could not embed query (Ollama?).")
                await ent.shutdown()
                return
            pairs = await ent.store.search_episodes_by_embedding(db, qe, limit=8)
            for eid, score in pairs:
                click.echo(f"{score:.3f}  {eid}")
        finally:
            await db.close()
        await ent.shutdown()

    asyncio.run(_recall())


@cli.command("export")
@click.argument("entity_name")
@click.argument("dest_dir", type=click.Path())
def cmd_export(entity_name: str, dest_dir: str) -> None:
    """Copy entity YAML + SQLite DB to a folder."""
    harness = load_harness_config()
    ec = load_entity_config(entity_name, harness)
    src_yaml = project_configs_dir() / "entities" / f"{entity_name}.yaml"
    d = Path(dest_dir)
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_yaml, d / f"{entity_name}.yaml")
    db = Path(ec.db_path())
    if db.exists():
        shutil.copy2(db, d / "memory.db")
    click.echo(f"Exported to {d}")


@cli.command("import")
@click.argument("bundle_dir", type=click.Path(exists=True))
def cmd_import(bundle_dir: str) -> None:
    """Import entity YAML + memory.db into configs and ~/.bumblebee."""
    b = Path(bundle_dir)
    yamls = list(b.glob("*.yaml"))
    if not yamls:
        raise click.ClickException("No YAML in bundle")
    y = yamls[0]
    name = y.stem
    ent_dir = project_configs_dir() / "entities"
    ent_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(y, ent_dir / f"{name}.yaml")
    mem = b / "memory.db"
    if mem.exists():
        harness = load_harness_config()
        ec = load_entity_config(name, harness)
        dest = Path(ec.db_path())
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(mem, dest)
    click.echo(f"Imported entity {name}")


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
