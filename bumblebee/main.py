"""CLI entry: setup, create, run, stop, talk, worker, api, status, evolve, knowledge, journal, recall, wipe, soma-reset, export, import, update, gateway (on/off/status/restart/setup)."""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import shutil
import signal
import subprocess
from importlib import metadata
from pathlib import Path

import click

from bumblebee.config import (
    load_entity_config,
    load_harness_config,
    project_configs_dir,
    validate_entity_env,
)
from bumblebee.entity import Entity, format_user_visible_failure
from bumblebee.genesis import creator
from bumblebee.health import check_inference
from bumblebee.models import Input
from bumblebee.presence.daemon import PresenceDaemon
from bumblebee.presence.platforms.base import Platform
from bumblebee.presence.platforms.cli import CLIPlatform
from bumblebee.presence.platforms.discord_platform import DiscordPlatform, token_from_config
from bumblebee.presence.platforms.telegram_platform import TelegramPlatform, merge_telegram_operator_user_ids
from bumblebee.utils.log import setup_logging
from bumblebee.utils.ollama_bootstrap import bootstrap_stack, shutdown_spawned_ollama
from bumblebee.utils.gateway_script import (
    gateway_script_available,
    run_gateway_script as _run_gateway_script_subprocess,
)
from bumblebee.utils.repo_dotenv import load_repo_dotenv
from bumblebee.utils.self_update import perform_self_update


def _async(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


KNOWLEDGE_STARTER_TEMPLATE = """## [locked] about yourself
write anything you want the entity to know about itself.
(locked sections can't be edited by the entity — only by you)

## [locked] about your creator
who made you? what do you know about them?

## things you care about
topics, opinions, facts — whatever matters.
(the entity can edit unlocked sections like this one)
"""


def _app_version() -> str:
    try:
        return metadata.version("bumblebee")
    except metadata.PackageNotFoundError:
        return "0.1.0"


def _load_repo_dotenv() -> None:
    load_repo_dotenv(anchor=Path(__file__))


async def _wait_until_stopped() -> None:
    """Block until SIGINT (Ctrl+C) or SIGTERM where available."""
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()

    def _schedule_stop(*_args: object) -> None:
        try:
            loop.call_soon_threadsafe(stop.set)
        except RuntimeError:
            pass

    prev_sigint: object | None = None
    prev_sigterm: object | None = None
    used_asyncio_sigint = False
    try:
        loop.add_signal_handler(signal.SIGINT, _schedule_stop)
        used_asyncio_sigint = True
    except (NotImplementedError, RuntimeError):
        prev_sigint = signal.signal(signal.SIGINT, _schedule_stop)
    if hasattr(signal, "SIGTERM"):
        try:
            loop.add_signal_handler(signal.SIGTERM, _schedule_stop)
        except (NotImplementedError, RuntimeError):
            try:
                prev_sigterm = signal.signal(signal.SIGTERM, _schedule_stop)
            except ValueError:
                pass
    try:
        await stop.wait()
    finally:
        if used_asyncio_sigint:
            try:
                loop.remove_signal_handler(signal.SIGINT)
            except (NotImplementedError, RuntimeError):
                pass
        elif prev_sigint is not None:
            signal.signal(signal.SIGINT, prev_sigint)
        if hasattr(signal, "SIGTERM"):
            try:
                loop.remove_signal_handler(signal.SIGTERM)
            except (NotImplementedError, RuntimeError, ValueError):
                pass
            if prev_sigterm is not None:
                try:
                    signal.signal(signal.SIGTERM, prev_sigterm)
                except ValueError:
                    pass


async def _graceful_run_shutdown(daemon: PresenceDaemon, platforms: list, ent: Entity) -> None:
    """Best-effort cleanup when the loop is cancelling (Ctrl+C); shield so close() can finish."""
    try:
        await asyncio.shield(daemon.stop())
    except BaseException:
        pass
    for p in platforms:
        try:
            await asyncio.shield(p.disconnect())
        except BaseException:
            pass
    try:
        await asyncio.shield(ent.shutdown())
    except BaseException:
        pass
    shutdown_spawned_ollama()


class _OneShotCLIPlatform(Platform):
    """Minimal CLI platform for single-turn prompts."""

    def __init__(self, *, person_id: str = "cli_user", person_name: str = "You") -> None:
        self._person_id = person_id
        self._person_name = person_name
        self._cb = None

    async def connect(self) -> None:
        return None

    async def send_message(self, channel: str, content: str) -> None:
        click.echo(content)

    async def send_tool_activity(self, description: str) -> None:
        return None

    async def on_message(self, callback) -> None:
        self._cb = callback

    async def set_presence(self, status: str) -> None:
        return None

    async def disconnect(self) -> None:
        return None

    def get_person_id(self, message: object) -> str:
        return self._person_id

    def get_person_name(self, message: object) -> str:
        return self._person_name


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


def _run_gateway_script(action: str, extra_args: list[str] | None = None) -> None:
    rc = _run_gateway_script_subprocess(action, extra_args)
    if rc != 0:
        raise click.ClickException(f"gateway {action} failed with exit code {rc}.")


@click.group()
def cli() -> None:
    # Runs before every subcommand; keep in sync with ``main()`` for entry points that call ``cli`` only.
    _load_repo_dotenv()


@cli.command("setup")
@click.option(
    "--profile",
    type=click.Choice(["ask", "hybrid", "local"], case_sensitive=False),
    default="ask",
    show_default=True,
    help="hybrid = home Ollama + gateway + Cloudflare tunnel + Railway (default when asking); local = single machine.",
)
@click.option(
    "--mode",
    "legacy_mode",
    type=click.Choice(["ask", "quick", "full"], case_sensitive=False),
    default=None,
    hidden=True,
    help="Deprecated: use --profile. quick→local, full→hybrid.",
)
def cmd_setup(profile: str, legacy_mode: str | None) -> None:
    """Interactive setup wizard (.env, gateway stack, Railway, entity)."""
    from bumblebee.setup_wizard import run_setup_wizard

    resolved = profile
    if legacy_mode == "quick":
        resolved = "local"
    elif legacy_mode == "full":
        resolved = "hybrid"
    elif legacy_mode == "ask":
        resolved = "ask"
    run_setup_wizard(profile=resolved)


@cli.command("create")
def cmd_create() -> None:
    """Launch interactive entity wizard."""
    creator.run_wizard()


@cli.group("gateway")
def cmd_gateway() -> None:
    """Inference gateway + Cloudflare tunnel: setup wizard; on/off/status/restart (Windows script)."""


@cmd_gateway.command("setup")
@click.option("--tunnel-name", default="bumblebee-inference", show_default=True)
@click.option("--cloudflared-config", default="", help="Path to cloudflared config.yml (default: ~/.cloudflared/config.yml).")
@click.option("--gateway-host", default="127.0.0.1", show_default=True)
@click.option("--gateway-port", default=8010, show_default=True, type=int)
def cmd_gateway_setup(
    tunnel_name: str,
    cloudflared_config: str,
    gateway_host: str,
    gateway_port: int,
) -> None:
    """Interactive wizard: bearer token, tunnel config, .env, then optional gateway on (Windows)."""
    from bumblebee.gateway_setup import run_gateway_setup_wizard

    run_gateway_setup_wizard(
        tunnel_name=tunnel_name,
        cloudflared_config=cloudflared_config,
        gateway_host=gateway_host,
        gateway_port=gateway_port,
    )


@cmd_gateway.command("on")
@click.option("--tunnel-name", default="bumblebee-inference", show_default=True)
@click.option("--cloudflared-config", default="", help="Path to cloudflared config.yml.")
@click.option("--tunnel-url", default="", help="Optional tunnel URL override for probes.")
@click.option("--gateway-host", default="127.0.0.1", show_default=True)
@click.option("--gateway-port", default=8010, show_default=True, type=int)
def cmd_gateway_on(
    tunnel_name: str,
    cloudflared_config: str,
    tunnel_url: str,
    gateway_host: str,
    gateway_port: int,
) -> None:
    """Start/validate ollama + inference gateway + cloudflared tunnel."""
    args = [
        "-TunnelName",
        tunnel_name,
        "-GatewayHost",
        gateway_host,
        "-GatewayPort",
        str(gateway_port),
    ]
    if cloudflared_config.strip():
        args += ["-CloudflaredConfig", cloudflared_config.strip()]
    if tunnel_url.strip():
        args += ["-TunnelUrl", tunnel_url.strip()]
    _run_gateway_script("on", args)


@cmd_gateway.command("off")
@click.option("--tunnel-name", default="bumblebee-inference", show_default=True)
@click.option("--leave-ollama-running", is_flag=True)
def cmd_gateway_off(tunnel_name: str, leave_ollama_running: bool) -> None:
    """Stop cloudflared + gateway (+ ollama unless --leave-ollama-running)."""
    args = ["-TunnelName", tunnel_name]
    if leave_ollama_running:
        args.append("-LeaveOllamaRunning")
    _run_gateway_script("off", args)


@cmd_gateway.command("status")
@click.option("--tunnel-name", default="bumblebee-inference", show_default=True)
@click.option("--cloudflared-config", default="", help="Path to cloudflared config.yml.")
@click.option("--tunnel-url", default="", help="Optional tunnel URL override for probes.")
@click.option("--gateway-host", default="127.0.0.1", show_default=True)
@click.option("--gateway-port", default=8010, show_default=True, type=int)
def cmd_gateway_status(
    tunnel_name: str,
    cloudflared_config: str,
    tunnel_url: str,
    gateway_host: str,
    gateway_port: int,
) -> None:
    """Show gateway process + health status."""
    args = [
        "-TunnelName",
        tunnel_name,
        "-GatewayHost",
        gateway_host,
        "-GatewayPort",
        str(gateway_port),
    ]
    if cloudflared_config.strip():
        args += ["-CloudflaredConfig", cloudflared_config.strip()]
    if tunnel_url.strip():
        args += ["-TunnelUrl", tunnel_url.strip()]
    _run_gateway_script("status", args)


@cmd_gateway.command("restart")
@click.option("--tunnel-name", default="bumblebee-inference", show_default=True)
@click.option("--cloudflared-config", default="", help="Path to cloudflared config.yml.")
@click.option("--tunnel-url", default="", help="Optional tunnel URL override for probes.")
@click.option("--gateway-host", default="127.0.0.1", show_default=True)
@click.option("--gateway-port", default=8010, show_default=True, type=int)
@click.option("--leave-ollama-running", is_flag=True)
def cmd_gateway_restart(
    tunnel_name: str,
    cloudflared_config: str,
    tunnel_url: str,
    gateway_host: str,
    gateway_port: int,
    leave_ollama_running: bool,
) -> None:
    """Stop then start the gateway stack (same as ``off`` then ``on``)."""
    args = [
        "-TunnelName",
        tunnel_name,
        "-GatewayHost",
        gateway_host,
        "-GatewayPort",
        str(gateway_port),
    ]
    if cloudflared_config.strip():
        args += ["-CloudflaredConfig", cloudflared_config.strip()]
    if tunnel_url.strip():
        args += ["-TunnelUrl", tunnel_url.strip()]
    if leave_ollama_running:
        args.append("-LeaveOllamaRunning")
    _run_gateway_script("restart", args)


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
    finally:
        shutdown_spawned_ollama()


@cli.command("ask")
@click.argument("entity_name")
@click.argument("message_parts", nargs=-1, required=True)
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
def cmd_ask(
    entity_name: str,
    message_parts: tuple[str, ...],
    with_ollama: bool,
    pull_models: bool,
) -> None:
    """Single-turn CLI prompt (one reply, no daemon)."""
    message = " ".join(message_parts).strip()
    if not message:
        raise click.ClickException("Message cannot be empty.")
    _prepare_ollama_cli(entity_name, with_ollama, pull_models)
    try:
        asyncio.run(_ask_once(entity_name, message))
    except KeyboardInterrupt:
        click.echo("\nStopped.", err=True)
    finally:
        shutdown_spawned_ollama()


async def _ask_once(entity_name: str, message: str) -> None:
    harness = load_harness_config()
    ec = load_entity_config(entity_name, harness)
    setup_logging(
        ec.name,
        ec.harness.logging.level,
        ec.log_path(),
        ec.harness.logging.format,
        immersive=True,
    )
    ent = Entity(ec)
    cli_p = _OneShotCLIPlatform()
    await cli_p.connect()
    ent.register_platform("cli", cli_p)
    inp = Input(
        text=message,
        person_id=cli_p.get_person_id(None),
        person_name=cli_p.get_person_name(None),
        channel="cli",
        platform="cli",
    )
    try:
        reply, _ = await ent.perceive(inp, reply_platform=cli_p)
        click.echo(reply)
    finally:
        try:
            await asyncio.shield(cli_p.disconnect())
        except BaseException:
            pass
        try:
            await asyncio.shield(ent.shutdown())
        except BaseException:
            pass


async def _talk(entity_name: str) -> None:
    harness = load_harness_config()
    ec = load_entity_config(entity_name, harness)
    setup_logging(
        ec.name,
        ec.harness.logging.level,
        ec.log_path(),
        ec.harness.logging.format,
        immersive=True,
    )
    ent = Entity(ec)
    cli_p = CLIPlatform(ent, app_version=_app_version(), immersive=True)
    await cli_p.connect()
    ent.register_platform("cli", cli_p)

    async def on_inp(inp):
        try:
            reply, _ = await ent.perceive(inp, stream=cli_p.stream_delta, reply_platform=cli_p)
            return reply
        except Exception as e:
            return format_user_visible_failure(e)

    await cli_p.on_message(on_inp)
    try:
        await cli_p.run_repl()
    finally:
        try:
            await asyncio.shield(ent.shutdown())
        except BaseException:
            pass


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
        asyncio.run(_run(entity_name, worker_mode=False))
    except KeyboardInterrupt:
        click.echo("\nStopped.", err=True)
    finally:
        shutdown_spawned_ollama()


@cli.command("stop")
@click.option(
    "--skip-gateway",
    is_flag=True,
    help="Do not run scripts/gateway.ps1 off (Windows home stack: cloudflared + gateway + Ollama).",
)
@click.option("--tunnel-name", default="bumblebee-inference", show_default=True)
@click.option(
    "--leave-ollama-running",
    is_flag=True,
    help="Do not terminate Ollama OS processes; forward to gateway.ps1 off when applicable.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="List processes that would be stopped; do not kill or run gateway off.",
)
def cmd_stop(
    skip_gateway: bool,
    tunnel_name: str,
    leave_ollama_running: bool,
    dry_run: bool,
) -> None:
    """Stop local Bumblebee processes, optional Windows gateway stack, and all Ollama OS processes."""
    from bumblebee.utils.stack_stop import stop_all_ollama_processes, stop_local_bumblebee_processes

    stop_local_bumblebee_processes(dry_run=dry_run, log=click.echo)
    if dry_run:
        if skip_gateway:
            click.echo("Dry run: would skip gateway off (--skip-gateway).")
        elif gateway_script_available():
            click.echo(
                f"Dry run: would run gateway off (tunnel={tunnel_name!r}"
                f"{', leave Ollama running (script only)' if leave_ollama_running else ''})."
            )
        else:
            click.echo("Dry run: gateway.ps1 not available — would skip gateway off.")
        if leave_ollama_running:
            click.echo("Dry run: would leave Ollama OS processes running (--leave-ollama-running).")
        else:
            stop_all_ollama_processes(dry_run=True, log=click.echo)
        return

    if not skip_gateway and gateway_script_available():
        args = ["-TunnelName", tunnel_name]
        if leave_ollama_running:
            args.append("-LeaveOllamaRunning")
        rc = _run_gateway_script_subprocess("off", args)
        if rc != 0:
            click.echo(
                f"gateway off exited with code {rc} (nothing running or script reported an error).",
                err=True,
            )
        else:
            click.echo("Home gateway stack stopped (gateway.ps1 off).")
    elif not skip_gateway:
        click.echo(
            "Gateway script not available (Windows + scripts/gateway.ps1 only). "
            "Skipping gateway off; stopping Bumblebee + Ollama processes only.",
            err=True,
        )

    if skip_gateway:
        click.echo("Left gateway script untouched (--skip-gateway).")

    if leave_ollama_running:
        click.echo("Left Ollama running (--leave-ollama-running).")
    else:
        stop_all_ollama_processes(dry_run=False, log=click.echo)


async def _run(entity_name: str, *, worker_mode: bool = False) -> None:
    harness = load_harness_config()
    ec = load_entity_config(entity_name, harness)
    for w in validate_entity_env(ec):
        click.echo(f"Warning: {w}", err=True)
    _worker_quiet = (
        "bumblebee.presence.telegram",
        "bumblebee.presence.discord",
        "bumblebee.inference_gateway",
        "telegram",
        "httpx",
        "httpcore",
    ) if worker_mode else ()
    setup_logging(
        ec.name,
        ec.harness.logging.level,
        ec.log_path(),
        ec.harness.logging.format,
        console_quiet=_worker_quiet,
    )
    ent = Entity(ec)
    try:
        inf = await check_inference(ec, ent.client)
        if not inf.get("ok"):
            click.echo(f"Warning: inference not healthy: {inf}", err=True)
    except Exception as e:
        click.echo(f"Warning: inference check failed: {e}", err=True)

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
            proactive_raw = pl.get("proactive_channel_id")
            proactive_cid: int | None = None
            if proactive_raw is not None:
                try:
                    proactive_cid = int(proactive_raw)
                except (TypeError, ValueError):
                    proactive_cid = None
            discord_p = DiscordPlatform(
                tok,
                [str(c) for c in chans],
                entity=ent,
                proactive_channel_id=proactive_cid,
            )
            platforms.append(discord_p)
            ent.register_platform("discord", discord_p)
        elif t == "telegram":
            tok = token_from_config(pl.get("token_env", "TELEGRAM_TOKEN"))
            if not tok:
                click.echo("Telegram token missing; skipping Telegram.", err=True)
                continue
            raw_allow = pl.get("allowed_user_ids")
            allow_users: set[int] | None = None
            if isinstance(raw_allow, list) and len(raw_allow) > 0:
                allow_users = {int(x) for x in raw_allow}
            raw_chats = pl.get("allowed_chat_ids")
            allow_chats: set[int] | None = None
            if isinstance(raw_chats, list) and len(raw_chats) > 0:
                allow_chats = {int(x) for x in raw_chats}
            op_users = merge_telegram_operator_user_ids(pl.get("operator_user_ids"))
            raw_cu = pl.get("concurrent_updates", 1)
            try:
                tg_concurrent_updates = max(1, min(256, int(raw_cu)))
            except (TypeError, ValueError):
                tg_concurrent_updates = 1
            raw_pt = pl.get("poll_timeout", 20.0)
            try:
                tg_poll_timeout = max(1.0, min(30.0, float(raw_pt)))
            except (TypeError, ValueError):
                tg_poll_timeout = 20.0
            raw_pi = pl.get("poll_interval", 0.35)
            try:
                tg_poll_interval = max(0.0, min(2.0, float(raw_pi)))
            except (TypeError, ValueError):
                tg_poll_interval = 0.35
            telegram_p = TelegramPlatform(
                tok,
                entity=ent,
                app_version=_app_version(),
                allowed_user_ids=allow_users,
                allowed_chat_ids=allow_chats,
                operator_user_ids=op_users,
                concurrent_updates=tg_concurrent_updates,
                poll_timeout=tg_poll_timeout,
                poll_interval=tg_poll_interval,
            )
            platforms.append(telegram_p)
            ent.register_platform("telegram", telegram_p)

    cli_p = CLIPlatform(ent, app_version=_app_version(), immersive=False)
    ent.register_platform("cli", cli_p)
    has_cli = (
        any((p.get("type") or "").lower() == "cli" for p in ec.presence.platforms)
        and not worker_mode
    )

    async def proactive_fanout(message: str) -> None:
        sent = False
        if discord_p:
            try:
                await discord_p.send_proactive_default(message)
                sent = True
            except Exception:
                pass
        if telegram_p:
            try:
                await telegram_p.send_proactive_default(message)
                sent = True
            except Exception:
                pass
        if not sent:
            click.echo(f"\n[{ec.name}]: {message}\n")

    ent.register_proactive_sink(proactive_fanout)

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
            await discord_p.send_plain_chunks(inp.channel, text, chunk_pause=meta.chunk_pause)
            await discord_p.sync_emotion_presence(ent.emotions.get_state().primary)
        elif inp.platform == "telegram" and telegram_p:
            await telegram_p.send_plain_chunks(inp.channel, text, pause=meta.chunk_pause)

    async def _telegram_typing_worker(chat_id: int, stop: asyncio.Event) -> None:
        while not stop.is_set():
            await telegram_p.send_typing(chat_id)
            try:
                await asyncio.wait_for(stop.wait(), timeout=4.5)
            except asyncio.TimeoutError:
                continue

    async def on_inp(inp):
        stream = cli_p.stream_delta if inp.platform == "cli" else None
        reply_pf = None
        if inp.platform == "cli":
            reply_pf = cli_p
        elif inp.platform == "discord":
            reply_pf = discord_p
        elif inp.platform == "telegram":
            reply_pf = telegram_p
        stop_typing = asyncio.Event()
        typing_task: asyncio.Task[None] | None = None
        if inp.platform == "telegram" and telegram_p:
            typing_task = asyncio.create_task(
                _telegram_typing_worker(int(inp.channel), stop_typing)
            )
        try:
            try:
                reply, needs_platform_route = await ent.perceive(
                    inp, stream=stream, reply_platform=reply_pf
                )
            except Exception as e:
                reply = format_user_visible_failure(e)
                needs_platform_route = True
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
            if needs_platform_route and reply:
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
            await _graceful_run_shutdown(daemon, platforms + [cli_p], ent)
    else:
        click.echo(f"{ec.name} running (no CLI). Ctrl+C to stop.")
        try:
            await _wait_until_stopped()
        except asyncio.CancelledError:
            pass
        finally:
            await _graceful_run_shutdown(daemon, platforms, ent)


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
        du = ec.database_url()
        click.echo(f"DB: {du if du else ec.db_path()}")
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


@cli.command("knowledge")
@click.argument("entity_name")
def cmd_knowledge(entity_name: str) -> None:
    """Create (if missing) and open the entity's knowledge.md in $EDITOR (default notepad / nano)."""
    harness = load_harness_config()
    ec = load_entity_config(entity_name, harness)
    kpath = Path(ec.knowledge_path()).expanduser()
    kpath.parent.mkdir(parents=True, exist_ok=True)
    if not kpath.is_file():
        kpath.write_text(KNOWLEDGE_STARTER_TEMPLATE, encoding="utf-8", newline="\n")
    editor = (os.environ.get("EDITOR") or "").strip()
    if not editor:
        if os.name == "nt":
            subprocess.run(["notepad", str(kpath)], check=False)
        else:
            nano = shutil.which("nano") or "nano"
            subprocess.run([nano, str(kpath)], check=False)
        return
    try:
        parts = shlex.split(editor, posix=os.name != "nt") + [str(kpath)]
    except ValueError as e:
        raise click.ClickException(f"Could not parse EDITOR: {e}") from e
    subprocess.run(parts, check=False)


@cli.command("journal")
@click.argument("entity_name")
def cmd_journal(entity_name: str) -> None:
    """Open the entity's journal.md in $EDITOR (append-only reflections from routines)."""
    harness = load_harness_config()
    ec = load_entity_config(entity_name, harness)
    jpath = Path(ec.journal_path()).expanduser()
    jpath.parent.mkdir(parents=True, exist_ok=True)
    if not jpath.is_file():
        jpath.write_text("# journal\n\n", encoding="utf-8", newline="\n")
    editor = (os.environ.get("EDITOR") or "").strip()
    if not editor:
        if os.name == "nt":
            subprocess.run(["notepad", str(jpath)], check=False)
        else:
            nano = shutil.which("nano") or "nano"
            subprocess.run([nano, str(jpath)], check=False)
        return
    try:
        parts = shlex.split(editor, posix=os.name != "nt") + [str(jpath)]
    except ValueError as e:
        raise click.ClickException(f"Could not parse EDITOR: {e}") from e
    subprocess.run(parts, check=False)


@cli.command("recall")
@click.argument("entity_name")
@click.argument("query")
def cmd_recall(entity_name: str, query: str) -> None:
    """Search episodic memories by semantic similarity."""
    harness = load_harness_config()
    ec = load_entity_config(entity_name, harness)
    ent = Entity(ec)

    async def _recall():
        try:
            async with ent.store.session() as db:
                qe = await ent.client.embed(ec.harness.models.embedding, query)
                if not qe:
                    click.echo("Could not embed query (inference backend unavailable?).")
                    return
                pairs = await ent.store.search_episodes_by_embedding(db, qe, limit=8)
                for eid, score in pairs:
                    click.echo(f"{score:.3f}  {eid}")
        finally:
            try:
                await asyncio.shield(ent.shutdown())
            except BaseException:
                pass

    asyncio.run(_recall())


@cli.command("worker")
@click.argument("entity_name")
def cmd_worker(entity_name: str) -> None:
    """Run platforms + daemon without CLI (use on Railway as bumblebee-worker)."""
    if not (entity_name or "").strip():
        raise click.ClickException(
            "Missing entity name (got empty string). For Railway: set BUMBLEBEE_ENTITY "
            "(e.g. canary) on the worker service."
        )
    try:
        asyncio.run(_run(entity_name, worker_mode=True))
    except KeyboardInterrupt:
        click.echo("\nStopped.", err=True)


@cli.command("api")
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", default=8080, show_default=True, type=int)
def cmd_api(host: str, port: int) -> None:
    """HTTP health service (Railway bumblebee-api). Requires: pip install 'bumblebee[api]'"""
    try:
        import uvicorn
    except ImportError as e:
        raise click.ClickException(
            "Missing uvicorn/fastapi. Install: pip install 'bumblebee[api]'"
        ) from e
    from bumblebee.apps.api import create_api_app

    uvicorn.run(create_api_app(), host=host, port=port, log_level="info")


@cli.command("wipe")
@click.argument("entity_name")
@click.option(
    "--yes",
    "skip_confirm",
    is_flag=True,
    help="Skip confirmation (non-interactive).",
)
def cmd_wipe(entity_name: str, skip_confirm: bool) -> None:
    """Delete in-memory chat turns and all SQLite memory (episodes, people, beliefs, narrative, etc.)."""
    harness = load_harness_config()
    ec = load_entity_config(entity_name, harness)
    db_path = ec.db_path()
    if not skip_confirm:
        click.echo(
            f"This clears conversation context and wipes the memory database for {ec.name!r}:\n  {db_path}",
            err=True,
        )
        if not click.confirm("Proceed?"):
            raise click.Abort()
    setup_logging(ec.name, ec.harness.logging.level, None, "console")
    ent = Entity(ec)

    async def _wipe():
        await ent.wipe_experiential_state()
        try:
            await asyncio.shield(ent.shutdown())
        except BaseException:
            pass

    asyncio.run(_wipe())
    click.echo(f"Wiped {ec.name} (DB + rolling chat + inner voice; YAML traits restored from file).")


@cli.command("soma-reset")
@click.argument("entity_name")
@click.option(
    "--yes",
    "skip_confirm",
    is_flag=True,
    help="Skip confirmation (non-interactive / Railway).",
)
def cmd_soma_reset(entity_name: str, skip_confirm: bool) -> None:
    """Reset soma bars to YAML initials, clear GEN noise + affects; write DB + soma-state.json + body.md.

    Hybrid / Railway: run once against the same DATABASE_URL and workspace as the worker
    (e.g. ``railway run --service bumblebee-worker -- bumblebee soma-reset YOUR_ENTITY --yes``),
    then restart the worker so the running process reloads tonic state.
    """
    harness = load_harness_config()
    ec = load_entity_config(entity_name, harness)
    if not skip_confirm:
        click.echo(
            f"Reset soma (bars, noise, affects) for {ec.name!r} to harness YAML baseline "
            f"and persist to DB + {ec.soma_dir()}.",
            err=True,
        )
        if not click.confirm("Proceed?"):
            raise click.Abort()
    setup_logging(ec.name, ec.harness.logging.level, None, "console")
    ent = Entity(ec)

    async def _reset():
        await ent.reset_soma_baseline()
        try:
            await asyncio.shield(ent.shutdown())
        except BaseException:
            pass

    asyncio.run(_reset())
    click.echo(f"Soma baseline reset for {ec.name}. Restart the worker if it is already running.")


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


@cli.command("update")
@click.option(
    "--no-pip",
    "skip_pip",
    is_flag=True,
    help="After a git fast-forward, skip ``pip install -e .`` (git checkouts only).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Print what would run without fetching, merging, or installing.",
)
def cmd_update(skip_pip: bool, dry_run: bool) -> None:
    """Update this install from the public GitHub repository (git or pip)."""

    def _log(msg: str) -> None:
        click.echo(msg)

    res = perform_self_update(
        pip_reinstall=not skip_pip,
        dry_run=dry_run,
        log=_log,
    )
    if res.get("ok"):
        if dry_run:
            click.echo("Dry run finished.")
        elif res.get("method") == "git":
            for m in res.get("messages") or ():
                click.echo(m)
            pip_r = res.get("pip")
            if isinstance(pip_r, dict) and pip_r.get("ok") and (pip_r.get("output") or "").strip():
                click.echo((pip_r.get("output") or "").strip())
            click.echo("Update complete. Restart running workers or daemons to load new code.")
        else:
            out = (res.get("output") or "").strip()
            if out:
                click.echo(out)
            click.echo("Update complete. Restart running workers or daemons to load new code.")
        return

    err = str(res.get("error") or "update failed")
    raise click.ClickException(err)


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
