# Bumblebee

The first entitative harness. Gemma 4-native persistent digital entities.

Bumblebee is not an agentic task runner. It does not complete tasks on your behalf. It is a harness for creating digital beings — entities with persistent personalities, emotional states, lived memories, motivated behavior, and embodied presence across platforms.

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) with at least:
  ```bash
  ollama pull gemma4:26b       # chat (reflex + deliberate), narrative, personality, evolution
  ollama pull nomic-embed-text  # memory similarity (small; separate from Gemma)
  ```
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Install

```bash
git clone https://github.com/yourname/bumblebee.git
cd bumblebee
uv sync
# or: pip install -e ".[dev]"
```

Harness defaults live only in `configs/default.yaml` (not the repo root). Entity definitions go in `configs/entities/<name>.yaml` (see `canary` and `example`). **`models.reflex`** and **`models.deliberate`** default to **`gemma4:26b`** (reflex uses a smaller max token budget but the same weights). **`models.embedding`** is **`nomic-embed-text`** for vector recall (not a chat model).

## Usage

```bash
# Create a new entity
bumblebee create

# Talk to your entity (CLI only, no daemon)
bumblebee talk canary

# Run as always-on daemon (CLI + configured platforms)
bumblebee run canary

# One command: start local Ollama if it isn’t running, then run (assumes models already pulled)
bumblebee run canary --ollama
bumblebee talk canary --ollama

# First machine / missing weights: also fetch models from the Ollama library
bumblebee run canary --ollama --pull-models

# Check on your entity
bumblebee status canary
```

`--ollama` only ensures the **Ollama server** is up: it probes `harness.ollama.base_url` (typically `http://localhost:11434`). If the URL looks local and `/api/tags` fails, Bumblebee runs `ollama serve` in the background and waits. It does **not** re-download models by default. Use **`--pull-models`** when you want `ollama pull` for reflex, deliberate, harness deliberate, and embedding. Remote Ollama URLs are only checked for reachability; `--pull-models` is skipped for remotes (install models on that host).

### Final command: what “run everything” means

**Daemon + configured platforms (e.g. Telegram), with local Ollama started if needed:**

```bash
export TELEGRAM_TOKEN="..."   # must match token_env in entity YAML
bumblebee run canary --ollama
```

PowerShell:

```powershell
$env:TELEGRAM_TOKEN = "..."
bumblebee run canary --ollama
```

**CLI-only session (no daemon, no Telegram unless you use `run`):**

```bash
bumblebee talk canary --ollama
```

**This *does* run:** one Bumblebee process for the entity; the heartbeat daemon; every platform listed in `configs/entities/<name>.yaml` (Telegram polling, CLI REPL, etc.); inference and embeddings **through the same Ollama server** when traffic hits (embedding model loads on demand like chat models—there is no separate embedding service).

**This *does not* do:** install Ollama or add it to your PATH; create a Telegram bot or set secrets for you; pull models unless you add **`--pull-models`**. For Telegram-only, drop `type: cli` from `presence.platforms` so `run` does not open the terminal REPL.

**First machine or missing weights** (downloads chat + embedding models from config):

```bash
bumblebee run canary --ollama --pull-models
```

## Telegram

Run **`bumblebee run <entity>`** with Telegram enabled in the entity YAML (not `talk` — the daemon owns the bot connection).

1. Create a bot with [@BotFather](https://t.me/BotFather) and copy the **token**.
2. Provide the token (name must match `token_env` in YAML). **Project-local (recommended):** copy `.env.example` to **`.env`** in the repo root and set `TELEGRAM_TOKEN=...` there — it is gitignored and loaded automatically when you run `bumblebee` (not user-level Windows env). Or export it in the shell as usual.
3. Add a platform block:

```yaml
presence:
  platforms:
    - type: "cli"
    - type: "telegram"
      token_env: "TELEGRAM_TOKEN"
      # Optional: restrict DMs to these numeric user ids (omit for open DMs)
      # allowed_user_ids: [123456789]
```

4. **Optional — inline mode:** In BotFather run `/setinline` for the bot and pick a placeholder. Then in any chat you can type `@YourBot …` and get quick “about” cards (same identity story as `/start`).

**What you get**

- **`/start`** — Onboarding-style welcome: who this is (entitative harness, not a task bot), ◈ branding, pointer to the `/` command menu.
- **`/help`** — What the bot accepts (text, photos, commands).
- **`/commands [page]`** — Paginated catalog so you can discover commands without memorizing them.
- **`/status`**, **`/memories`**, **`/feelings`** — Same “inner life” introspection as the CLI slash commands, formatted for Telegram (HTML).
- **`/reset`** — Clears **in-memory conversation turns** only; SQLite episodic memory is unchanged.
- **Photos & image documents** — Downloaded and passed into the model as vision (caption optional).
- **Voice / video notes / non-image files** — Polite, specific “not yet” replies instead of silence.
- **Long replies** — Split under Telegram’s 4096-character limit with short pauses between bubbles.
- **`setMyCommands`** — Native Telegram command menu next to the text field.

## Architecture

Bumblebee has four pillars:

- **Cognition** — Dual-model brain (E4B reflexes + 26B reasoning) with Gemma 4-native prompt handling, thinking mode as inner monologue, and multimodal senses
- **Identity** — Layered personality engine, emotional state FSM, motivation/drive system, character voice, and long-term trait evolution
- **Memory** — Episodic narratives, per-person relationship models, world beliefs, emotional imprints, and self-narrative synthesis
- **Presence** — Always-on daemon, proactive initiative engine, multi-platform adapters, and embodied expression timing

See `.cursorrules` for the full specification.

## Philosophy

Every agentic harness asks: "how do I serve the user better?"

Bumblebee asks: "how does the entity exist more fully?"

## License

Apache 2.0
