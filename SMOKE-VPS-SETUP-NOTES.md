# Smoke VPS setup notes (BitLaunch + Ollama)

**Host:** `64.94.85.139` (BitLaunch instance **bumblebee-smoke**, nibble-1024)  
**Clone path on server:** `/root/bumblebee`  
**Last updated:** 2026-04-09

This file lives at the **repo root** because `docs/` is listed in `.gitignore` for this project.

This file records what was done on a **1 vCPU / ~1 GiB RAM / no GPU** box so we can open targeted issues on **this repo** and reproduce or undo work.

## What was installed

- **2 GiB swap** (RAM pressure without it).
- **uv** + `uv sync`; **Ollama** (CPU-only).
- **Models:** `gemma3:270m-it-qat`, `nomic-embed-text`.
- **Entity `smoke`:** `configs/entities/smoke.yaml` on the server (CLI-only, small context, automations off).
- **`git remote add upstream`** → `https://github.com/Bumblebee-AGI/bumblebee.git` — used to replace **`bumblebee/`** with **upstream `main`** so the tree imports (see issues below).
- **`configs/default.yaml`** on server: `gemma4:26b` → `gemma3:270m-it-qat` so model checks match hardware.
- **`.env`:** `BUMBLEBEE_OLLAMA_NO_TOOLS=1` + small **`entity.py`** patch so Ollama is called without tools (small Gemma builds reject tool calls).

**Sanity check (server):**  
`cd /root/bumblebee && .venv/bin/bumblebee ask smoke --ollama "hello"`

## GitHub issues (this repo — **Bumblebee-AGI/bumblebee**)

| # | Topic |
|---|--------|
| [#2](https://github.com/Bumblebee-AGI/bumblebee/issues/2) | CI: import smoke test to catch package / `entity.py` drift (replaces obsolete “truncated fork” blocker) |
| [#3](https://github.com/Bumblebee-AGI/bumblebee/issues/3) | `validate_ollama_models` should not require harness default when entity overrides models |
| [#4](https://github.com/Bumblebee-AGI/bumblebee/issues/4) | Feature: omit `tools` for Ollama models that do not support tool calling |

Previously filed on a private fork; **tracked here** after the project moved public.

## Not goals for this host

- Documented **Gemma 4 26B** + multi‑GB VRAM path in the main README — this VPS is intentionally **tiny** and **CPU-only**.
