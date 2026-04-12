"""Interactive harness setup: ``bumblebee setup``.

Default path: **hybrid** — Ollama + inference gateway + Cloudflare tunnel at home, body on Railway.
Can run ``gateway.ps1`` / ``gateway.sh`` (home stack), ``npm run ollama:reset``, automated ``cloudflared`` tunnel + DNS,
and ``railway`` CLI when available.
"""

from __future__ import annotations

import secrets
import shutil
import subprocess
from pathlib import Path

import click

from bumblebee.config import project_configs_dir
from bumblebee.genesis import creator
from bumblebee.utils.cloudflared_config import default_cloudflared_config_path
from bumblebee.utils.gateway_health_probe import print_gateway_health_block
from bumblebee.utils.setup_preflight import collect_preflight, format_preflight_text
from bumblebee.utils.tunnel_url_prompt import prompt_public_gateway_base_url
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


def _list_entity_names() -> list[str]:
    ent_dir = project_configs_dir() / "entities"
    if not ent_dir.is_dir():
        return []
    names: list[str] = []
    for p in sorted(ent_dir.glob("*.yaml")):
        if p.name.startswith("."):
            continue
        if p.name.endswith(".example.yaml"):
            continue
        names.append(p.stem)
    return names


def _railway_bin() -> str | None:
    return shutil.which("railway")


def _railway_ok() -> bool:
    b = _railway_bin()
    if not b:
        return False
    r = subprocess.run(
        [b, "whoami"],
        cwd=_repo_root(),
        capture_output=True,
        text=True,
        timeout=60,
    )
    return r.returncode == 0


def _railway_linked() -> bool:
    b = _railway_bin()
    if not b:
        return False
    r = subprocess.run(
        [b, "status"],
        cwd=_repo_root(),
        capture_output=True,
        text=True,
        timeout=30,
    )
    if r.returncode != 0:
        return False
    return "Project:" in (r.stdout or "")


def _ensure_railway_cli_ready() -> bool:
    """Ensure logged in and linked when the user wants Railway automation."""
    b = _railway_bin()
    if not b:
        return False
    if not _railway_ok():
        click.echo("Railway CLI needs a login.")
        if click.confirm("Run `railway login` now?", default=True):
            subprocess.call([b, "login"], cwd=_repo_root())
        if not _railway_ok():
            click.echo("Still not logged in — run `railway login` from the repo root and retry.", err=True)
            return False
    if not _railway_linked():
        click.echo(
            "This folder must be linked to your Railway project (the one with bumblebee-worker / bumblebee-api)."
        )
        if click.confirm("Run `railway link` now?", default=True):
            subprocess.call([b, "link"], cwd=_repo_root())
        if not _railway_linked():
            click.echo(
                "Still not linked — from repo root: `railway link` and pick the project with your services.",
                err=True,
            )
            return False
    return True


DEFAULT_GATEWAY_HOST = "127.0.0.1"
DEFAULT_GATEWAY_PORT = 8010


def _railway_set(service: str, pairs: list[str]) -> bool:
    b = _railway_bin()
    if not b:
        return False
    r = subprocess.run(
        [b, "variable", "set", "-s", service, *pairs],
        cwd=_repo_root(),
        capture_output=True,
        text=True,
        timeout=120,
    )
    if r.returncode != 0:
        click.echo((r.stderr or r.stdout or "railway variable set failed").strip(), err=True)
    return r.returncode == 0


def _railway_set_stdin(service: str, key: str, value: str) -> bool:
    b = _railway_bin()
    if not b:
        return False
    r = subprocess.run(
        [b, "variable", "set", "-s", service, key, "--stdin"],
        input=value,
        text=True,
        cwd=_repo_root(),
        capture_output=True,
        timeout=120,
    )
    if r.returncode != 0:
        click.echo((r.stderr or r.stdout or "railway variable set --stdin failed").strip(), err=True)
    return r.returncode == 0


def _run_npm(script: str) -> int:
    npm = shutil.which("npm")
    if not npm:
        return 127
    return subprocess.call([npm, "run", script], cwd=_repo_root())


def _print_railway_cheat_sheet(entity: str, base_url: str, pg_service: str) -> None:
    click.echo("\nRailway (copy-paste if CLI steps were skipped):\n")
    db_ref = "${{" + pg_service + ".DATABASE_URL}}"
    click.echo(
        f"  railway variable set -s bumblebee-worker DATABASE_URL={db_ref}\n"
        f"  railway variable set -s bumblebee-api DATABASE_URL={db_ref}\n"
    )
    common = (
        f"BUMBLEBEE_DEPLOYMENT_MODE=hybrid_railway BUMBLEBEE_INFERENCE_PROVIDER=remote_gateway "
        f"BUMBLEBEE_INFERENCE_BASE_URL={base_url} BUMBLEBEE_ENTITY={entity} "
    )
    click.echo(f"  railway variable set -s bumblebee-worker {common}BUMBLEBEE_RAILWAY_ROLE=worker")
    click.echo("  printf '%s' '<token>' | railway variable set -s bumblebee-worker BUMBLEBEE_INFERENCE_GATEWAY_TOKEN --stdin")
    click.echo(f"  railway variable set -s bumblebee-api {common}BUMBLEBEE_RAILWAY_ROLE=api")
    click.echo("  printf '%s' '<token>' | railway variable set -s bumblebee-api BUMBLEBEE_INFERENCE_GATEWAY_TOKEN --stdin")
    click.echo("\n  railway up -s bumblebee-worker\n  railway up -s bumblebee-api\n")
    click.echo("(Replace <token> with the same secret as INFERENCE_GATEWAY_TOKEN in .env; base URL is tunnel root, no path.)\n")


def _entity_section() -> str | None:
    """Returns entity name if picked or created; ``None`` if skipped or still none."""
    click.echo("\n--- Entity ---\n")
    names = _list_entity_names()
    if not names:
        click.echo("No entities in configs/entities/ yet.")
        if click.confirm("Run the entity creator wizard now?", default=True):
            return creator.run_wizard()
        click.echo("Later: bumblebee create")
        return None
    click.echo("Entities: " + ", ".join(names))
    choice = click.prompt(
        "[c]reate new  /  [p]ick existing  /  [s]kip",
        type=click.Choice(["c", "p", "s"], case_sensitive=False),
        default="p",
    )
    if choice == "c":
        return creator.run_wizard()
    if choice == "p":
        pick = click.prompt(
            "Entity name",
            type=click.Choice(names, case_sensitive=False),
        )
        click.echo(f"\nTry: bumblebee talk {pick}")
        if click.confirm("Show command with --ollama?", default=False):
            click.echo(f"  bumblebee talk {pick} --ollama")
        return pick
    return None


def _resolve_entity_name_for_railway(chosen: str | None) -> str:
    if chosen and chosen.strip():
        return chosen.strip()
    names = _list_entity_names()
    if len(names) == 1:
        return names[0]
    if len(names) > 1:
        return click.prompt(
            "Entity name for Railway (BUMBLEBEE_ENTITY)",
            type=click.Choice(names, case_sensitive=False),
            default=names[0],
        )
    typed = _prompt_line("Entity name for Railway (must match configs/entities/<name>.yaml — Enter to skip)")
    return typed.strip()


def _run_local_flow(env_path: Path) -> None:
    click.echo("\n--- Local (single machine) ---\n")
    updates: dict[str, str] = {}
    ollama = _prompt_line("Ollama URL (Enter to skip — uses default.yaml)")
    if ollama:
        updates["OLLAMA_HOST"] = ollama.rstrip("/")
    if click.confirm("Set TELEGRAM_TOKEN?", default=False):
        t = _prompt_secret("TELEGRAM_TOKEN")
        if t:
            updates["TELEGRAM_TOKEN"] = t
    if click.confirm("Set DISCORD_TOKEN?", default=False):
        t = _prompt_secret("DISCORD_TOKEN")
        if t:
            updates["DISCORD_TOKEN"] = t
    updates["BUMBLEBEE_DEPLOYMENT_MODE"] = "local"
    updates["BUMBLEBEE_INFERENCE_PROVIDER"] = "local"
    merge_dotenv_keys(env_path, updates)
    click.echo(f"\nUpdated {env_path} ({len(updates)} key(s))")


def _run_hybrid_env_and_home(
    env_path: Path,
    *,
    gateway_host: str,
    gateway_port: int,
    skip_tunnel_bootstrap: bool,
) -> tuple[str, str]:
    click.echo("\n--- Hybrid (home brain + Railway body) ---\n")
    click.echo(
        "What you are setting up:\n"
        "  • Home: Ollama + the inference gateway + a Cloudflare Tunnel (your GPU stays here).\n"
        "  • Cloud: Railway worker + API + Postgres (always-on presence and storage).\n"
        "The tunnel must expose only the gateway URL — not a generic reverse proxy.\n"
    )

    updates: dict[str, str] = {}

    tok = _resolve_gateway_token(env_path)
    if not tok:
        if click.confirm("Generate a new shared gateway token (home + Railway)?", default=True):
            tok = secrets.token_urlsafe(32)
        else:
            tok = _prompt_secret("Paste INFERENCE_GATEWAY_TOKEN (same value Railway will use)")
    if not tok:
        raise click.ClickException("A gateway token is required for the hybrid stack.")
    updates["INFERENCE_GATEWAY_TOKEN"] = tok
    updates["BUMBLEBEE_INFERENCE_GATEWAY_TOKEN"] = tok

    cf_path = default_cloudflared_config_path()
    base_url = prompt_public_gateway_base_url(
        cf_path=cf_path,
        gateway_host=gateway_host,
        gateway_port=gateway_port,
        skip_tunnel_bootstrap=skip_tunnel_bootstrap,
        required=True,
    )
    updates["BUMBLEBEE_DEPLOYMENT_MODE"] = "hybrid_railway"
    updates["BUMBLEBEE_INFERENCE_PROVIDER"] = "remote_gateway"
    updates["BUMBLEBEE_INFERENCE_BASE_URL"] = base_url

    ollama = _prompt_line("Ollama URL for home stack", default="http://127.0.0.1:11434")
    if ollama:
        updates["OLLAMA_HOST"] = ollama.rstrip("/")

    if click.confirm("Optional: set FIRECRAWL_API_KEY in .env?", default=False):
        fc = _prompt_secret("FIRECRAWL_API_KEY")
        if fc:
            updates["FIRECRAWL_API_KEY"] = fc

    if click.confirm("Optional: DATABASE_URL in .env (local worker test against Postgres)?", default=False):
        db = _prompt_line("DATABASE_URL")
        if db:
            updates["DATABASE_URL"] = db

    if click.confirm("Optional: S3 attachment vars in .env (hybrid durable blobs)?", default=False):
        s3: dict[str, str] = {}
        for key, label in (
            ("BUMBLEBEE_S3_ENDPOINT_URL", "BUMBLEBEE_S3_ENDPOINT_URL"),
            ("BUMBLEBEE_S3_BUCKET", "BUMBLEBEE_S3_BUCKET"),
            ("BUMBLEBEE_S3_ACCESS_KEY", "BUMBLEBEE_S3_ACCESS_KEY"),
            ("BUMBLEBEE_S3_SECRET_KEY", "BUMBLEBEE_S3_SECRET_KEY"),
        ):
            v = _prompt_secret(label)
            if v:
                s3[key] = v
        updates.update(s3)
        if s3:
            updates["BUMBLEBEE_ATTACHMENTS_BACKEND"] = "object_s3_compat"

    merge_dotenv_keys(env_path, updates)
    click.echo(f"\nUpdated {env_path} ({len(updates)} key(s))")

    if click.confirm("Run npm run ollama:reset (model pulls / Ollama defaults on Windows)?", default=False):
        rc = _run_npm("ollama:reset")
        if rc != 0:
            click.echo(f"npm run ollama:reset exited {rc} (install Node/npm or run scripts/ollama-reset.ps1).", err=True)

    gateway_rc: int | None = None
    if gateway_script_available():
        if click.confirm("Start home stack now (ollama + inference gateway + cloudflared)?", default=True):
            gateway_rc = run_gateway_script("on")
            if gateway_rc != 0:
                click.echo(
                    "gateway on failed — fix cloudflared config (~/.cloudflared/config.yml) and run: bumblebee gateway on",
                    err=True,
                )
        elif click.confirm("Show gateway status only?", default=False):
            run_gateway_script("status")
    else:
        click.echo(
            "\nHome stack: run `bumblebee gateway on` after cloudflared is configured "
            "(requires `scripts/gateway.ps1` on Windows or `scripts/gateway.sh` on macOS/Linux).\n"
            "Else: ollama serve, then INFERENCE_GATEWAY_TOKEN=… python -m bumblebee.inference_gateway, "
            "then cloudflared tunnel run …\n"
        )

    if click.confirm(
        "Run gateway health checks (GET /health with your token — local"
        + (" + tunnel" if base_url else "")
        + ")?",
        default=(gateway_rc is None or gateway_rc == 0),
    ):
        print_gateway_health_block(
            gateway_host=gateway_host,
            gateway_port=gateway_port,
            public_base_url=base_url,
            token=tok,
        )

    return base_url, tok


def _railway_volume_exists(service: str, mount: str) -> bool | None:
    """Check whether *service* has a volume at *mount*. Returns ``None`` when the check cannot run."""
    b = _railway_bin()
    if not b:
        return None
    r = subprocess.run(
        [b, "volume", "list"],
        cwd=_repo_root(),
        capture_output=True,
        text=True,
        timeout=60,
    )
    if r.returncode != 0:
        return None
    return mount in (r.stdout or "")


def _railway_variable_get(service: str, key: str) -> str | None:
    """Read a single Railway variable. Returns ``None`` if the CLI fails."""
    b = _railway_bin()
    if not b:
        return None
    r = subprocess.run(
        [b, "variable", "list", "-s", service],
        cwd=_repo_root(),
        capture_output=True,
        text=True,
        timeout=60,
    )
    if r.returncode != 0:
        return None
    for line in (r.stdout or "").splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            if k.strip() == key:
                return v.strip()
    return ""


def _run_hybrid_volume_setup() -> None:
    """Ensure the bumblebee-worker has a persistent volume at /app/data."""
    click.echo("\n--- Persistent volume (worker data that survives redeploys) ---\n")
    click.echo(
        "The worker needs a Railway volume mounted at /app/data for knowledge.md,\n"
        "journal.md, soma state, and any files the entity creates. Without it,\n"
        "those files are lost every time the container redeploys.\n"
    )

    has_volume = _railway_volume_exists("bumblebee-worker", "/app/data")
    if has_volume is None:
        click.echo("Could not check volumes (Railway CLI unavailable or not linked).")
        click.echo("Manual step: railway volume add --mount-path /app/data -s bumblebee-worker\n")
    elif has_volume:
        click.echo("Volume at /app/data detected — good.\n")
    else:
        click.echo("No volume at /app/data found on bumblebee-worker.")
        if click.confirm("Create one now?", default=True):
            b = _railway_bin()
            if b:
                r = subprocess.run(
                    [b, "volume", "add", "--mount-path", "/app/data", "-s", "bumblebee-worker"],
                    cwd=_repo_root(),
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if r.returncode == 0:
                    click.echo("Volume created at /app/data.")
                else:
                    click.echo(
                        (r.stderr or r.stdout or "railway volume add failed").strip(),
                        err=True,
                    )
                    click.echo("Create manually: railway volume add --mount-path /app/data -s bumblebee-worker")
        else:
            click.echo("Skipped. Create later: railway volume add --mount-path /app/data -s bumblebee-worker\n")

    ws_val = _railway_variable_get("bumblebee-worker", "BUMBLEBEE_EXECUTION_WORKSPACE_DIR")
    if ws_val is None:
        click.echo("Could not check variables. Ensure BUMBLEBEE_EXECUTION_WORKSPACE_DIR=/app/data is set.\n")
    elif ws_val == "/app/data":
        click.echo("BUMBLEBEE_EXECUTION_WORKSPACE_DIR already set to /app/data.\n")
    else:
        click.echo(
            "BUMBLEBEE_EXECUTION_WORKSPACE_DIR tells the harness where to store knowledge,\n"
            "journal, soma state, and workspace files. It should point at the volume mount.\n"
        )
        if click.confirm("Set BUMBLEBEE_EXECUTION_WORKSPACE_DIR=/app/data on bumblebee-worker?", default=True):
            _railway_set("bumblebee-worker", ["BUMBLEBEE_EXECUTION_WORKSPACE_DIR=/app/data"])
            click.echo("Set.\n")
        else:
            click.echo("Skipped. Set later: railway variable set -s bumblebee-worker BUMBLEBEE_EXECUTION_WORKSPACE_DIR=/app/data\n")

    click.echo("See docs/hybrid-railway-persistence.md for the full persistence guide.\n")


def _run_hybrid_railway(entity_choice: str | None, base_url: str, tok: str) -> None:
    entity = _resolve_entity_name_for_railway(entity_choice)
    if not entity:
        click.echo("Skipping Railway CLI — set BUMBLEBEE_ENTITY after you add an entity YAML.")
        click.echo("Inference: tunnel must end at the gateway only; see .env.example and bumblebee/inference_gateway/.")
        return

    pg_service = _prompt_line("Railway Postgres plugin service name (for DATABASE_URL reference)", default="Postgres")

    railway_ok = False
    if _railway_bin():
        if _ensure_railway_cli_ready():
            railway_ok = True
            if click.confirm("Apply hybrid variables on Railway (bumblebee-worker + bumblebee-api)?", default=True):
                db_tok = "${{" + pg_service + ".DATABASE_URL}}"
                _railway_set("bumblebee-worker", [f"DATABASE_URL={db_tok}"])
                _railway_set("bumblebee-api", [f"DATABASE_URL={db_tok}"])
                wvars = [
                    "BUMBLEBEE_DEPLOYMENT_MODE=hybrid_railway",
                    "BUMBLEBEE_INFERENCE_PROVIDER=remote_gateway",
                    f"BUMBLEBEE_INFERENCE_BASE_URL={base_url}",
                    f"BUMBLEBEE_ENTITY={entity}",
                    "BUMBLEBEE_RAILWAY_ROLE=worker",
                ]
                avs = [
                    "BUMBLEBEE_DEPLOYMENT_MODE=hybrid_railway",
                    "BUMBLEBEE_INFERENCE_PROVIDER=remote_gateway",
                    f"BUMBLEBEE_INFERENCE_BASE_URL={base_url}",
                    f"BUMBLEBEE_ENTITY={entity}",
                    "BUMBLEBEE_RAILWAY_ROLE=api",
                ]
                _railway_set("bumblebee-worker", wvars)
                _railway_set("bumblebee-api", avs)
                _railway_set_stdin("bumblebee-worker", "BUMBLEBEE_INFERENCE_GATEWAY_TOKEN", tok)
                _railway_set_stdin("bumblebee-api", "BUMBLEBEE_INFERENCE_GATEWAY_TOKEN", tok)
                if click.confirm("Set TELEGRAM_TOKEN on bumblebee-worker (from stdin)?", default=False):
                    tg = _prompt_secret("TELEGRAM_TOKEN")
                    if tg:
                        _railway_set_stdin("bumblebee-worker", "TELEGRAM_TOKEN", tg)
                if click.confirm("Set DISCORD_TOKEN on bumblebee-worker (from stdin)?", default=False):
                    dc = _prompt_secret("DISCORD_TOKEN")
                    if dc:
                        _railway_set_stdin("bumblebee-worker", "DISCORD_TOKEN", dc)
    else:
        click.echo("Railway CLI not on PATH — install it, then use the cheat sheet below.")

    if railway_ok:
        _run_hybrid_volume_setup()

    if railway_ok and click.confirm(
        "Deploy worker + API now (`railway up` — publishes to your Railway project)?",
        default=True,
    ):
        rb = _railway_bin()
        if rb:
            subprocess.call([rb, "up", "-s", "bumblebee-worker", "-d"], cwd=_repo_root())
            subprocess.call([rb, "up", "-s", "bumblebee-api", "-d"], cwd=_repo_root())

    _print_railway_cheat_sheet(entity, base_url, pg_service)

    click.echo("Inference: tunnel must end at the gateway only; see .env.example and bumblebee/inference_gateway/.")


def run_setup_wizard(
    *,
    profile: str = "ask",
    gateway_host: str = DEFAULT_GATEWAY_HOST,
    gateway_port: int = DEFAULT_GATEWAY_PORT,
    skip_tunnel_bootstrap: bool = False,
) -> None:
    root = _repo_root()
    env_path = _dotenv_path()

    click.echo("")
    click.echo(click.style("Bumblebee setup", fg="cyan", bold=True))
    click.echo(
        "Default path: hybrid — Ollama + gateway + Cloudflare tunnel at home, worker on Railway.\n"
        "This wizard can create the tunnel + DNS + config when `cloudflared` is installed, "
        "and push env vars / deploy via the Railway CLI when you are logged in and linked.\n"
        "Reference: README.md (Hybrid deployment), .env.example, configs/default.yaml\n"
    )
    click.echo(f"Repo: {root}")
    click.echo(f".env: {env_path}\n")

    if not env_path.is_file():
        if click.confirm(".env not found — create it?", default=True):
            env_path.write_text(
                "# Bumblebee — created by `bumblebee setup`\n",
                encoding="utf-8",
            )
            click.echo(f"Created {env_path}")

    p = profile.strip().lower()
    if p == "ask":
        hybrid_default = click.confirm(
            "Use the recommended hybrid setup (home Ollama + gateway + tunnel + Railway)?",
            default=True,
        )
        p = "hybrid" if hybrid_default else "local"

    hybrid_base: str | None = None
    hybrid_tok: str | None = None
    if p == "hybrid":
        click.echo(click.style("\nReadiness (fix any ✗ before continuing)\n", fg="cyan", bold=True))
        click.echo(format_preflight_text(collect_preflight(repo_root=root)))
        click.echo("")
        gh = gateway_host.strip() or DEFAULT_GATEWAY_HOST
        gp = gateway_port if gateway_port > 0 else DEFAULT_GATEWAY_PORT
        hybrid_base, hybrid_tok = _run_hybrid_env_and_home(
            env_path,
            gateway_host=gh,
            gateway_port=gp,
            skip_tunnel_bootstrap=skip_tunnel_bootstrap,
        )
    elif p == "local":
        _run_local_flow(env_path)
    else:
        raise click.ClickException(f"Unknown profile: {profile!r}")

    entity_choice = _entity_section()
    if p == "hybrid" and hybrid_base is not None and hybrid_tok is not None:
        click.echo("\n--- Railway (cloud body) ---\n")
        _run_hybrid_railway(entity_choice, hybrid_base, hybrid_tok)

    click.echo("\n--- Done ---\n")
    click.echo("Next:")
    click.echo("  • Entity YAML: add Discord / Telegram under presence.platforms if you use those bots.")
    click.echo("  • Local chat: bumblebee talk <entity>  |  Full presence: bumblebee run <entity>")
    click.echo("  • Hybrid: keep home stack up when the Railway worker should think.\n")

