"""Interactive gateway + Cloudflare tunnel setup: ``bumblebee gateway setup``.

Walks through **both** together: shared bearer token + tunnel ingress to the local inference
gateway (Ollama stays behind the gateway). Windows can finish with ``bumblebee gateway on``.
"""

from __future__ import annotations

import os
import secrets
import shutil
from pathlib import Path

import click

from bumblebee.config import project_configs_dir
from bumblebee.utils.cloudflared_config import default_cloudflared_config_path, tunnel_https_url_from_config
from bumblebee.utils.dotenv_merge import merge_dotenv_keys
from bumblebee.utils.gateway_script import gateway_script_available, run_gateway_script


def _repo_root() -> Path:
    return project_configs_dir().parent


def _dotenv_path() -> Path:
    return _repo_root() / ".env"


def _read_dotenv_value(path: Path, key: str) -> str:
    if not path.is_file():
        return ""
    for raw in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, rest = line.partition("=")
        if k.strip() == key:
            return rest.strip().strip('"').strip("'")
    return ""


def _resolve_gateway_token(env_path: Path) -> str:
    for k in ("INFERENCE_GATEWAY_TOKEN", "BUMBLEBEE_INFERENCE_GATEWAY_TOKEN"):
        v = _read_dotenv_value(env_path, k)
        if v:
            return v
    return ""


def _prompt_secret(prompt: str) -> str:
    v = click.prompt(prompt, hide_input=True, confirmation_prompt=False)
    return (v or "").strip()


def _prompt_line(prompt: str, *, default: str | None = None) -> str:
    if default is None:
        v = click.prompt(prompt, show_default=False)
    else:
        v = click.prompt(prompt, default=default, show_default=True)
    return (v or "").strip()


def _cloudflared_executable() -> str | None:
    w = shutil.which("cloudflared")
    if w:
        return w
    if os.name == "nt":
        local = os.environ.get("LOCALAPPDATA", "")
        if local:
            link = Path(local) / "Microsoft" / "WinGet" / "Links" / "cloudflared.exe"
            if link.is_file():
                return str(link)
    return None


def _print_config_template(*, tunnel_name: str, gateway_host: str, gateway_port: int) -> None:
    svc = f"http://{gateway_host}:{gateway_port}"
    click.echo("")
    click.echo(
        "After `cloudflared tunnel create "
        + tunnel_name
        + "` you get a tunnel UUID and a credentials JSON path.\n"
        "Put something like this in your config (replace placeholders):\n"
    )
    click.echo(
        click.style("# ~/.cloudflared/config.yml (example)", fg="yellow")
        + "\n"
        "tunnel: <TUNNEL_UUID_FROM_CREATE>\n"
        "credentials-file: <PATH_TO_UUID.json>\n"
        "\n"
        "ingress:\n"
        "  - hostname: <your-public-hostname>\n"
        f"    service: {svc}\n"
        "  - service: http_status:404\n"
    )
    click.echo(
        "The tunnel hostname must forward to the gateway only — "
        + click.style("not", bold=True)
        + " a shared reverse proxy. See docs/architecture/inference-boundary.md.\n"
    )


def run_gateway_setup_wizard(
    *,
    tunnel_name: str,
    cloudflared_config: str,
    gateway_host: str,
    gateway_port: int,
) -> None:
    root = _repo_root()
    env_path = _dotenv_path()
    cf_path = Path(cloudflared_config.strip()) if cloudflared_config.strip() else default_cloudflared_config_path()

    click.echo("")
    click.echo(click.style("Bumblebee gateway + tunnel setup", fg="cyan", bold=True))
    click.echo(
        "You will wire two things together:\n"
        "  • A bearer token — used by the inference gateway and by `scripts/gateway.ps1` health checks.\n"
        "  • Cloudflare Tunnel — public HTTPS → this machine’s gateway only "
        f"(`http://{gateway_host}:{gateway_port}`).\n"
    )
    click.echo(f"Repo: {root}")
    click.echo(f".env: {env_path}")
    click.echo(f"cloudflared config: {cf_path}\n")

    if not env_path.is_file():
        if click.confirm(".env not found — create it?", default=True):
            env_path.write_text(
                "# Bumblebee — created by `bumblebee gateway setup`\n",
                encoding="utf-8",
            )
            click.echo(f"Created {env_path}")

    click.echo(click.style("— Step 1 — Bearer token (gateway + tunnel probes)", fg="green"))
    click.echo(
        "Set this once in `.env`. The same value must be used anywhere that calls your tunnel "
        "(e.g. Railway `BUMBLEBEE_INFERENCE_GATEWAY_TOKEN`).\n"
    )
    tok = _resolve_gateway_token(env_path)
    if tok:
        click.echo("Found an existing gateway token in .env.")
        if not click.confirm("Keep it?", default=True):
            tok = ""
    if not tok:
        if click.confirm("Generate a new random token?", default=True):
            tok = secrets.token_urlsafe(32)
        else:
            tok = _prompt_secret("Paste INFERENCE_GATEWAY_TOKEN")
    if not tok:
        raise click.ClickException("A gateway token is required.")

    click.echo(click.style("— Step 2 — Cloudflare Tunnel (while the token above is in mind)", fg="green"))
    click.echo(
        "On this machine:\n"
        "  1) Install cloudflared (e.g. Windows: winget install Cloudflare.cloudflared).\n"
        "  2) cloudflared tunnel login\n"
        f"  3) cloudflared tunnel create {tunnel_name}\n"
        "  4) Route DNS to the tunnel (dashboard or `cloudflared tunnel route dns …`).\n"
        "  5) Edit config so ingress sends your public hostname to the gateway URL only.\n"
    )
    exe = _cloudflared_executable()
    if exe:
        click.echo(f"cloudflared found: {exe}")
    else:
        click.echo(
            click.style("cloudflared not found on PATH.", fg="yellow")
            + " Install it, then re-run this wizard or continue manually.\n"
        )

    if cf_path.is_file():
        click.echo(f"Config exists: {cf_path}")
        detected = tunnel_https_url_from_config(cf_path)
        if detected:
            click.echo(f"Detected tunnel URL: {detected}")
    else:
        click.echo(click.style(f"Config missing: {cf_path}", fg="yellow"))
        _print_config_template(
            tunnel_name=tunnel_name,
            gateway_host=gateway_host,
            gateway_port=gateway_port,
        )
        click.echo(
            "When the file exists, `bumblebee gateway on` runs:\n"
            f'  cloudflared tunnel run {tunnel_name}\n'
        )

    if click.confirm("Show the config template again?", default=False):
        _print_config_template(
            tunnel_name=tunnel_name,
            gateway_host=gateway_host,
            gateway_port=gateway_port,
        )

    click.echo(click.style("— Step 3 — Local gateway bind + Ollama URL", fg="green"))
    click.echo(
        f"The inference gateway should listen on {gateway_host}:{gateway_port} "
        "(default). Tunnel ingress must target that URL.\n"
        "Install extras if needed: pip install 'bumblebee[gateway]'\n"
    )
    ollama = _prompt_line("Ollama base URL for the gateway (OLLAMA_HOST)", default="http://127.0.0.1:11434")
    ollama = ollama.rstrip("/")

    click.echo(click.style("— Step 4 — Write .env", fg="green"))
    detected = tunnel_https_url_from_config(cf_path) if cf_path.is_file() else ""
    if detected:
        click.echo(f"From config: {detected}")
        if click.confirm("Use as BUMBLEBEE_INFERENCE_BASE_URL?", default=True):
            base_url = detected
        else:
            base_url = _prompt_line("Public tunnel URL (https://…, no path)").rstrip("/")
    else:
        base_url = _prompt_line(
            "Public tunnel URL (https://…, no path) — set after DNS/config is ready",
            default="",
        ).rstrip("/")

    updates: dict[str, str] = {
        "INFERENCE_GATEWAY_TOKEN": tok,
        "BUMBLEBEE_INFERENCE_GATEWAY_TOKEN": tok,
        "OLLAMA_HOST": ollama,
    }
    if base_url:
        updates["BUMBLEBEE_INFERENCE_BASE_URL"] = base_url

    if click.confirm(
        "Also set hybrid-oriented keys BUMBLEBEE_DEPLOYMENT_MODE=hybrid_railway "
        "and BUMBLEBEE_INFERENCE_PROVIDER=remote_gateway?",
        default=False,
    ):
        updates["BUMBLEBEE_DEPLOYMENT_MODE"] = "hybrid_railway"
        updates["BUMBLEBEE_INFERENCE_PROVIDER"] = "remote_gateway"

    merge_dotenv_keys(env_path, updates)
    click.echo(f"Updated {env_path} ({len(updates)} key(s))")

    click.echo(click.style("— Step 5 — Start the stack", fg="green"))
    if gateway_script_available():
        click.echo(
            "Windows: `bumblebee gateway on` starts Ollama (if needed), the inference gateway, "
            "and cloudflared using your .env token.\n"
        )
        if click.confirm("Run it now?", default=True):
            args = [
                "-TunnelName",
                tunnel_name,
                "-GatewayHost",
                gateway_host,
                "-GatewayPort",
                str(gateway_port),
            ]
            if str(cf_path) != str(default_cloudflared_config_path()):
                args += ["-CloudflaredConfig", str(cf_path)]
            rc = run_gateway_script("on", args)
            if rc != 0:
                click.echo(
                    "Startup failed — fix cloudflared config or install Ollama, then: bumblebee gateway on",
                    err=True,
                )
        elif click.confirm("Show gateway status only?", default=False):
            args = [
                "-TunnelName",
                tunnel_name,
                "-GatewayHost",
                gateway_host,
                "-GatewayPort",
                str(gateway_port),
            ]
            if str(cf_path) != str(default_cloudflared_config_path()):
                args += ["-CloudflaredConfig", str(cf_path)]
            run_gateway_script("status", args)
    else:
        click.echo(
            "`scripts/gateway.ps1` not found (run from repo root on Windows for one-command startup).\n"
            "Otherwise, in separate terminals:\n"
            "  ollama serve\n"
            f"  INFERENCE_GATEWAY_TOKEN=<token> python -m bumblebee.inference_gateway\n"
            f"  cloudflared tunnel run {tunnel_name}   # with --config pointing at your config.yml\n"
        )

    click.echo("\n--- Done ---\n")
    click.echo("Verify locally:")
    click.echo(
        f'  curl -sS -H "Authorization: Bearer <token>" http://{gateway_host}:{gateway_port}/health\n'
    )
    if base_url:
        click.echo("Verify through the tunnel (after cloudflared is up):")
        click.echo(f'  curl -sS -H "Authorization: Bearer <token>" {base_url}/health\n')
    click.echo("Docs: docs/deployment/local-inference-node.md, docs/deployment/hybrid-railway.md\n")
