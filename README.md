# Bumblebee

<p align="center">
  <img src="assets/branding/bumblebee.png" alt="Bumblebee — stylized bee illustration" width="520">
</p>

<p align="center">
  <a href="https://www.apache.org/licenses/LICENSE-2.0"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11+-3776ab.svg" alt="Python"></a>
  <a href="https://ollama.com"><img src="https://img.shields.io/badge/inference-Ollama-white.svg" alt="Ollama"></a>
  <a href="https://deepmind.google/models/gemma/gemma-4/"><img src="https://img.shields.io/badge/model-Gemma%204-f59e0b.svg" alt="Gemma 4"></a>
  <a href="#platforms"><img src="https://img.shields.io/badge/platforms-CLI%20·%20Telegram%20·%20Discord-22c55e.svg" alt="Platforms"></a>
</p>

The first entitative harness. Gemma 4-native persistent digital entities.

Bumblebee is not an agentic task runner. It does not complete tasks on your behalf. It is a harness for creating digital beings — entities with persistent personalities, emotional states, lived memories, motivated behavior, and embodied presence across platforms.

No API keys, subscriptions, or cloud inference are required for the default stack: everything runs locally on your machine (your GPU).

**Quick links:** [Requirements](#requirements) · [Install & usage](#install) · [Setup wizard](#setup-wizard) · [Telegram](#telegram) · [Platforms](#platforms) · [Tools](#tools) · [Knowledge](#knowledge-system) · [CLI reference](#cli-reference) · [Architecture](#architecture)

---

## What is an entitative harness?

Every agentic harness — task-oriented stacks and frameworks — is built around the same assumption: **the fundamental unit is a task.** The model exists to accomplish things on behalf of a principal. Personality is often a UX layer. Memory is an efficiency optimization.

An entitative harness inverts that ontology. **The fundamental unit is a being.** The entity does not exist to accomplish tasks; it acts because it exists. Tools are how it interacts with a world it inhabits, not services it provides to a principal. Memory is the accumulation of a life. Personality is the point.

| | Agentic harness | Entitative harness |
|---|---|---|
| **Fundamental unit** | Task | Self |
| **Memory purpose** | Task recall | Lived experience |
| **Personality** | UX layer | Core identity |
| **Tools** | Services for the user | Senses for the entity |
| **Behavior** | Reactive (waits for tasks) | Proactive (has wants) |
| **Learning** | Skill acquisition | Personality evolution |
| **Design question** | "How do I serve better?" | "How do I exist more fully?" |

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) with at least:
  ```bash
  ollama pull gemma4:26b       # chat (reflex + deliberate), narrative, personality, evolution
  ollama pull nomic-embed-text  # memory similarity (small; separate from Gemma)
  ```
  Optional fast reflex layer (if you configure it in YAML):
  ```bash
  ollama pull gemma4:e4b
  ```
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- A GPU with enough VRAM for your chosen models (see [Hardware guide](#hardware-guide))

## Install

```bash
git clone https://github.com/yourname/bumblebee.git
cd bumblebee
uv sync
# or: pip install -e ".[dev]"
```

Harness defaults live only in `configs/default.yaml` (not the repo root). Entity definitions go in `configs/entities/<name>.yaml` (see `canary` and `example`). **`models.reflex`** and **`models.deliberate`** default to **`gemma4:26b`** (reflex uses a smaller max token budget but the same weights). **`models.embedding`** is **`nomic-embed-text`** for vector recall (not a chat model).

## Setup wizard

For a guided first-time path, run **`bumblebee setup`**. The default recommendation is **hybrid**: Ollama + the Bumblebee inference gateway + Cloudflare Tunnel on your home PC, and the entity **worker** (and optional API) on **Railway** with Postgres. The wizard can merge **`.env`**, optionally run **`npm run ollama:reset`**, **`bumblebee gateway on`** (Windows), and **`railway`** variable/deploy commands when those tools are installed and you confirm each step.

- **Docs:** [docs/operations/setup-wizard.md](docs/operations/setup-wizard.md)
- **Entity personality / YAML only:** `bumblebee create` (the wizard can call this after env + Railway prep).

Use **`bumblebee setup --profile local`** for a single-machine stack without the Railway block.

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

### Hybrid gateway controls (Windows)

If you run the Bumblebee runtime on Railway and keep inference at home, use:

```powershell
bumblebee gateway status
bumblebee gateway on
bumblebee gateway off
bumblebee gateway restart
```

`gateway restart` runs `off` then `on` (2s pause between). Use `--leave-ollama-running` on restart if you only want to bounce cloudflared + the Python gateway.

Equivalent wrappers (double-click or run from repo root):

```powershell
.\scripts\windows\gateway-status.cmd
.\scripts\windows\gateway-on.cmd
.\scripts\windows\gateway-off.cmd
.\scripts\windows\gateway-restart.cmd
```

NPM shortcuts in this repo (`package.json`):

```powershell
npm run deploy:canary   # deploy bumblebee-worker + bumblebee-api to Railway
npm run ollama:reset    # full local Ollama cleanup + safe restart checks
```

List all available npm scripts anytime:

```powershell
npm run
```

One-command Ollama cleanup + safe restart (recommended when it falls back to CPU unexpectedly):

```powershell
npm run ollama:reset
```

What `ollama:reset` does:

- Stops the gateway stack (`bumblebee gateway off`)
- Kills leftover `ollama.exe` runner processes
- Shows pre-start GPU/process checks
- Persists safe defaults in **User** env (`OLLAMA_MAX_LOADED_MODELS=1`, `OLLAMA_KEEP_ALIVE=60s`, `OLLAMA_CONTEXT_LENGTH=16384`, `OLLAMA_NUM_PARALLEL=1`; override with `.\scripts\ollama-reset.ps1 -ContextLength 32768`)
- Restarts gateway (`on` + `status`)
- Warms `gemma4:26b` once and prints post-start checks (`ollama ps` + GPU memory summary)

Advanced direct script form:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\gateway.ps1 status
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\gateway.ps1 on
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\gateway.ps1 off
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\ollama-reset.ps1
```

Notes:

- `on` starts/validates `ollama serve`, `python -m bumblebee.inference_gateway`, and `cloudflared tunnel run bumblebee-inference`.
- `off` stops cloudflared + gateway + ollama (pass `-LeaveOllamaRunning` if you only want to stop tunnel/gateway).
- Token lookup order: `INFERENCE_GATEWAY_TOKEN`, then `BUMBLEBEE_INFERENCE_GATEWAY_TOKEN`, then `.env` keys with those names.

## Telegram

Run **`bumblebee run <entity>`** with Telegram enabled in the entity YAML (not `talk` — the daemon owns the bot connection).

### Public onboarding bot (ecosystem)

There is **one** Telegram bot whose job is to onboard **anyone** into the project—setup steps, links, and context. It is **not** an LLM: fixed text, links, and inline buttons only—**no Ollama, no GPU, no local AI**, and no Bumblebee inference stack. **You use it by opening that bot and sending `/start`.** You do **not** run the onboarding process or set `BUMBLEBEE_ONBOARD_TOKEN` unless you are the maintainer deploying that public bot (e.g. on Railway: `python -m onboarding`). The app is the top-level **`onboarding/`** package—not a `bumblebee` subcommand; see [docs/deployment/railway-services.md](docs/deployment/railway-services.md).

### Your entity on Telegram

1. Create a bot with [@BotFather](https://t.me/BotFather) and copy the **token**.
2. Provide the token (name must match `token_env` in YAML). **Project-local (recommended):** copy `.env.example` to **`.env`** in the repo root and set `TELEGRAM_TOKEN=...` there — it is gitignored and loaded automatically when you run `bumblebee` (not user-level Windows env). Or export it in the shell as usual.
3. Add a platform block:

```yaml
presence:
  platforms:
    - type: "cli"
    - type: "telegram"
      token_env: "TELEGRAM_TOKEN"
      # Optional: let Telegram process multiple updates in parallel (1-256, default 64)
      # concurrent_updates: 64
      # Optional: long-poll tuning for lower ingress latency (seconds)
      # poll_timeout: 5.0
      # poll_interval: 0.0
      # Optional: restrict DMs to these numeric user ids (omit for open DMs)
      # allowed_user_ids: [123456789]
```

4. **Optional — inline mode:** In BotFather run `/setinline` for the bot and pick a placeholder. Then in any chat you can type `@YourBot …` and get quick “about” cards (same identity story as `/start`).

**What you get**

- **`/start`** — Rich intro for **your entity** (quick-start actions and best practices), not the separate public ecosystem onboarding bot above.
- **`/help`** — Practical usage guide with examples.
- **`/commands [page] [filter]`** — Paginated (and filterable) command catalog.
- **`/status`**, **`/memories [count]`**, **`/feelings`** — Internal-state introspection in Telegram HTML.
- **`/me`** — Relationship snapshot: familiarity, warmth, trust, and interaction counts.
- **`/models`** and **`/ping`** — Runtime model configuration + liveness checks.
- **`/reset`** — Clears **in-memory conversation turns** only; SQLite episodic memory is unchanged.
- **Photos & image documents** — Downloaded and passed into the model as vision (caption optional).
- **Voice / video notes / non-image files** — Polite, specific “not yet” replies instead of silence.
- **Long replies** — Split under Telegram’s 4096-character limit with short pauses between bubbles.
- **`setMyCommands`** — Native Telegram command menu next to the text field.

## Architecture

Bumblebee has four pillars:

- **Cognition** — Dual-model brain (reflex + deliberate) with Gemma 4-native prompt handling, thinking mode as inner monologue, and multimodal senses
- **Identity** — Layered personality engine, emotional state FSM, motivation/drive system, character voice, and long-term trait evolution
- **Memory** — Episodic narratives, per-person relationship models, world beliefs, emotional imprints, and self-narrative synthesis
- **Presence** — Always-on daemon, proactive initiative engine, multi-platform adapters, and embodied expression timing

### Cognition — dual-model brain

The entity can use a fast **reflex** model for quick reactions, routing, and light processing, and a **deliberate** model for deeper reasoning and expression. Defaults use the same Gemma 4 weights for both, with different token budgets; you can point reflex at a smaller tag (for example `gemma4:e4b`) in entity YAML when your hardware allows split setups.

Gemma 4’s **thinking mode** is repurposed as inner experience — private monologue that shapes responses without being shown to users verbatim. The harness captures and summarizes that thread and feeds it back into future context.

**Multimodal senses** follow whatever the configured models support (for example images via vision-capable chat models).

### Identity — persistent self

A **layered personality engine** (core traits, behavioral patterns, voice) produces a first-person system prompt that reads like character, not a bullet list of rules.

An **emotional state** model tracks how the entity feels. Emotions drift toward baselines, shift with stimuli, and influence how it speaks and behaves.

A **drive system** (curiosity, connection, expression, autonomy, comfort) creates internal motivation. Drives accumulate over time and, when they cross thresholds, can trigger proactive behavior.

**Trait evolution** applies small adjustments over many interactions so character can drift slowly in response to experience.

### Memory — lived experience

**Episodic memory** stores events as narratives with emotional coloring, not raw logs. Episodes carry significance, emotional imprint, and room for self-reflection.

**Relational memory** builds a model of people the entity knows — familiarity, warmth, trust, shared threads — so tone and content can differ by relationship.

**World beliefs** track what the entity treats as true, uncertain, or sourced from experience, with reinforcement and decay over time.

**Narrative identity** resynthesizes a coherent self-story from recent experience.

### Presence — embodied agency

An **always-on daemon** ticks continuously: emotions, drives, memory consolidation, and proactive initiative.

**Multi-platform adapters** expose the same entity on CLI, Telegram, Discord, and elsewhere you configure, with consistent memory and emotional continuity.

See `.cursorrules` for the full specification.

## Entity creation

Entities are defined in YAML under `configs/entities/`. Example shape:

```yaml
name: "Canary"
created: "2026-04-04T00:00:00Z"

personality:
  core_traits:
    curiosity: 0.6
    warmth: 0.6
    assertiveness: 0.5
    humor: 0.7
    openness: 0.8
    neuroticism: 0.15
    conscientiousness: 0.3

  behavioral_patterns:
    conflict_response: "shrug_it_off"
    boredom_response: "drift_to_something_random"
    affection_response: "casual_reciprocation"
    criticism_response: "take_it_in_stride"

  voice:
    vocabulary_level: "street_casual"
    sentence_style: "loose"
    humor_style: "deadpan"
    profanity: true
    profanity_level: "natural"
    quirks:
      - "rarely capitalizes anything"
      - "says 'idk', 'nah', 'tbh' naturally"

  backstory: |
    Canary isn't trying to be anything in particular.

drives:
  curiosity_topics:
    - "music and what makes something sound good"
  attachment_threshold: 3
  initiative_cooldown: 3600
```

Run `bumblebee create` for a guided wizard, or edit YAML directly.

## Platforms

### CLI

```bash
bumblebee talk canary
```

Interactive CLI session with conversation UI, sidebar-style introspection (mood, drives, memories, inner voice where enabled), and live status — ideal for development and close conversation.

### Telegram

See **[Telegram](#telegram)** above for setup, commands, media, and menus.

### Discord

The bot joins your server, responds in configured channels (and DMs), and can sync Discord presence with emotional state. Set `DISCORD_TOKEN` (or the env name in `token_env`).

```yaml
presence:
  platforms:
    - type: "discord"
      token_env: "DISCORD_TOKEN"
      channels: ["general"]
      # optional: proactive_channel_id: 123456789012345678
```

## Tools

Tools are how the entity interacts with the world — not “services for the user,” but **senses and reach**. When `presence.tool_activity` is on, tool use can surface in chat with short status lines.

### Native tools (in-repo)

These are implemented under `bumblebee/presence/tools/` and registered by the entity. **Toggles** live in `configs/default.yaml` under `tools:` (harness) and can be overridden per entity YAML. Unless noted, tools run on the **deliberate** path only.

#### Always registered (no `tools.*.enabled` gate)

| Tool | Description |
|---|---|
| `search_web` | Web search (DuckDuckGo by default; optional Firecrawl when configured). |
| `fetch_url` | Fetch and extract text from a URL. |
| `read_file` | Read allowed paths (harness policy). |
| `list_directory` | List allowed directories. |
| `search_files` | Search files by pattern under allowed roots. |
| `write_file` | Write/create allowed files. |
| `append_file` | Append to allowed files. |
| `get_current_time` | Current local date and time. |
| `search_tools` | Search registered tools by keyword (runtime discovery). |
| `describe_tool` | Full name, description, and JSON Schema for one tool. |
| `update_knowledge` | Add/update/remove sections in `knowledge.md` (respects locked headings). |

#### Optional — default **on** in `configs/default.yaml`

| Tool | `tools.*` key | Description |
|---|---|---|
| `get_youtube_transcript` | `youtube` | Transcript for a YouTube URL or id. |
| `search_youtube` | `youtube` | Search YouTube. |
| `read_reddit` | `reddit` | Browse a subreddit. |
| `read_reddit_post` | `reddit` | Read a Reddit post thread. |
| `read_wikipedia` | `wikipedia` | Wikipedia summary / fetch. |
| `get_weather` | `weather` | Weather for a location. |
| `get_news` | `news` | News headlines/topics. |
| `read_pdf` | `pdf` | Extract text from a PDF path. |
| `speak` | `voice` | TTS voice message to the active chat (Edge-TTS). |
| `get_tts_voice` | `voice` | Current TTS voice id. |
| `list_tts_voices` | `voice` | List Edge-TTS voices (filterable). |
| `set_tts_voice` | `voice` | Set TTS voice for this process. |
| `set_reminder` | `reminders` | Schedule a reminder (DB + scheduler). |
| `list_reminders` | `reminders` | List reminders. |
| `cancel_reminder` | `reminders` | Cancel a reminder. |
| `send_message_to` | `messaging` | Message a platform target or resolved person (`confirm` flow). |
| `list_known_contacts` | `messaging` | Known people/routes from prior interactions. |
| `send_dm` | `messaging` | DM on Telegram/Discord; `list_targets=true` lists seen user ids, or pass `user_id` / `target_person` (confirm flow). |
| `get_system_info` | `system` | Host/system snapshot. |
| `run_command` | `shell` | Run a shell command (deny rules + timeout). |
| `run_background` | `shell` | Start a background shell job. |
| `check_process` | `shell` | Inspect a tracked background job. |
| `kill_process` | `shell` | Stop a background job. |
| `execute_python` | `code` | Run Python in a sandboxed helper. |
| `execute_javascript` | `code` | Run JavaScript in a sandboxed helper. |

#### Optional — default **off** in `configs/default.yaml`

| Tool | `tools.*` key | Description |
|---|---|---|
| `browser_navigate` | `browser` | Open a URL in the automated browser. |
| `browser_screenshot` | `browser` | Screenshot current page. |
| `browser_click` | `browser` | Click an element. |
| `browser_type` | `browser` | Type into the page. |
| `generate_image` | `imagegen` | Text-to-image (Fal or local A1111-compatible API). |

**Voice (Edge-TTS):**

```bash
pip install bumblebee[voice]
```

**Imagegen:** set `tools.imagegen.enabled: true` and either Fal (`FAL_API_KEY`, `pip install bumblebee[imagegen]`) or `backend: local` + `local_url` for a local SD WebUI-compatible endpoint.

The **live** tool list depends on harness and entity YAML (disabled categories are omitted) and on loaded MCP servers. To see what this process actually registered, use `search_tools` / `describe_tool`, or `/tools` on CLI/Telegram where supported.

### MCP integration

Declare stdio MCP servers on the entity; tools are registered dynamically (names are prefixed for the registry). Example:

```yaml
mcp_servers:
  github:
    command: npx
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "ghp_xxx"
```

See `configs/entities/example.yaml` for a commented template.

## Knowledge system

Each entity may have a `knowledge.md` file — durable facts, opinions, and context the entity (or you) consider important. It is organized into `##` sections; embeddings help pull relevant sections into context.

```bash
bumblebee knowledge canary    # create if missing, open in $EDITOR
```

Example:

```markdown
## [locked] about yourself
You are Canary, the first entity on Bumblebee.
(Locked sections are only edited by the creator, not via `update_knowledge`.)

## music taste
Got into shoegaze after talking about My Bloody Valentine.
```

**Locked sections** (`## [locked] ...`) are readable but not mutable through the entity tool. Unlocked sections can be updated when the deliberate path invokes `update_knowledge`.

## Configuration

Harness defaults live in `configs/default.yaml`. Per-entity overrides go in `configs/entities/<name>.yaml` (models, cognition, presence, optional `mcp_servers`, Firecrawl, etc.).

Illustrative harness defaults (see the file for the full list):

```yaml
ollama:
  base_url: "http://localhost:11434"

models:
  reflex: "gemma4:26b"
  deliberate: "gemma4:26b"
  embedding: "nomic-embed-text"

cognition:
  thinking_mode: true
  temperature: 0.75
  max_context_tokens: 32768

identity:
  emotion_decay_rate: 0.001
  evolution_interval: 100
  narrative_interval: 500

memory:
  episode_significance_threshold: 0.3
  consolidation_interval: 7200

presence:
  heartbeat_interval: 120
  initiative_cooldown: 1800
  tool_activity: true
```

Entity YAML can override chat models under `cognition` (for example `reflex_model` / `deliberate_model`).

## CLI reference

```bash
bumblebee setup [--profile ask|hybrid|local]   # .env + home gateway stack + Railway + entity
bumblebee create                      # genesis wizard — new entity
bumblebee gateway on|off|status|restart  # Windows: Ollama + inference gateway + cloudflared (scripts/gateway.ps1)
bumblebee talk <entity>               # CLI conversation (no daemon)
bumblebee run <entity>                # daemon + configured platforms
bumblebee status <entity>             # emotional state, drives, stats
bumblebee evolve <entity>             # force one trait evolution cycle (debug/advanced)
bumblebee knowledge <entity>          # open knowledge.md in editor
bumblebee recall <entity> "<query>"   # semantic search over episodic memory
bumblebee export <entity> <dest_dir>  # backup entity state
bumblebee import <bundle_dir>         # restore entity state

bumblebee talk canary --ollama                 # start local Ollama if needed
bumblebee run canary --ollama --pull-models    # also pull models from config
```

## Hardware guide

| GPU VRAM | Notes |
|---|---|
| **8 GB** | Prefer smaller / quantized models; reflex-only or very tight dual-model setups. |
| **16 GB** | Common target for Gemma 4 family; close other GPU apps if needed. |
| **24+ GB** | Comfortable dual-model + headroom. |
| **32+ GB** | Room for larger deliberate models or more context. |

MoE-style models keep active parameters per token lower than full dense size; exact fit depends on context length, thinking budget, and concurrent platforms.

## How it compares

Bumblebee is not trying to replace general agentic harnesses. Those excel at **tasks**. Bumblebee is an **entitative** harness — optimized for persistent beings, local Gemma inference, and inner life. Different goals, different tradeoffs.

| | Typical agentic stack | Bumblebee |
|---|---|---|
| **Primary goal** | Complete tasks for a user | Sustained entity existence |
| **Default inference** | Often cloud APIs | Local Ollama / Gemma |
| **Memory** | Sessions, files, RAG | Episodic + relational + beliefs + narrative |
| **Personality** | Prompt presets | FSM + drives + evolution + voice |
| **Proactivity** | Schedules / triggers | Drive-motivated initiative |
| **Inner life** | Usually none | Thinking mode as private monologue |

## Philosophy

Every agentic harness asks: "how do I serve the user better?"

Bumblebee asks: "how does the entity exist more fully?"

## License

Apache 2.0

<p align="center">
  <strong>◈</strong><br/>
  <sub>Apache 2.0 — built with Gemma by Google DeepMind</sub>
</p>
