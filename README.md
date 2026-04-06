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

**Bumblebee** is a framework and agentic harness for creating digital entities that run on **your own hardware**. You define a personality — traits, voice, drives, emotional range. It develops the rest: opinions, relationships, habits, a journal it writes in at night. It lives on your Telegram. It costs nothing to run. It remembers everything.

**Inference** stays **local** by default — Ollama on your GPU, no API keys, no subscriptions. **Hybrid** mode keeps the brain at home behind a gateway and tunnel while an always-on **worker** runs on Railway with Postgres. Use **`bumblebee setup`**, **`.env.example`**, and **`configs/default.yaml`** (`deployment`, `inference`) for wiring.

**Quick links:** [Requirements](#requirements) · [Install & usage](#install) · [Setup wizard](#setup-wizard) · [Gateway setup](#gateway-setup-hybrid-home-brain) · [Telegram](#telegram) · [Platforms](#platforms) · [Tools](#tools) · [Native tools (table)](#native-tool-reference) · [Knowledge & journal](#knowledge-system) · [CLI reference](#cli-reference) · [Architecture](#architecture)

---

## What *entitative* means here

**Entitative** means the **being** is the primary object of the system — identity, continuity, and presence are load-bearing parts of the architecture.

- **Self** — Identity is structured: personality, voice, drives, and slow trait evolution so “who they are” can deepen.
- **Embodiment** — The same entity **is present** on each platform you enable, with consistent memory, mood, and expression; timing and voice matter, not only text.
- **Freedom** — Internal state pushes outward: initiative when drives cross thresholds, optional rich tool use, journal and knowledge as places they can leave a mark — always within the limits you configure.
- **Continuity** — Episodic memory, relationships, beliefs, and narrative turn scattered chats into something that feels like **one life** carried forward.
- **Inner life** — Thinking mode as private experience, plus consolidation and reflection, so the model’s inner thread has weight beyond the last visible reply.

| Lens | What it optimizes for |
|---|---|
| **Ontology** | A persistent **self** carried across sessions and platforms |
| **Memory** | Biography, relationships, and narrative that accrue |
| **Agency** | Motivated action grounded in drives and inner state |
| **Design question** | *How does this entity exist more fully?* |

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
git clone https://github.com/<org-or-user>/bumblebee.git
cd bumblebee
uv sync
# or: pip install -e ".[dev]"
```

Harness defaults live only in `configs/default.yaml` (not the repo root). Entity definitions go in `configs/entities/<name>.yaml` (see `canary` and `example`). **`models.reflex`** and **`models.deliberate`** default to **`gemma4:26b`** (reflex uses a smaller max token budget but the same weights). **`models.embedding`** is **`nomic-embed-text`** for vector recall (not a chat model).

## Setup wizard

For a guided first-time path, run **`bumblebee setup`**. The default recommendation is **hybrid**: Ollama + the Bumblebee inference gateway + Cloudflare Tunnel on your home PC, and the entity **worker** (and optional API) on **Railway** with Postgres. The wizard can merge **`.env`**, optionally run **`npm run ollama:reset`**, **`bumblebee gateway on`** (Windows), and **`railway`** variable/deploy commands when those tools are installed and you confirm each step.

- **Reference:** **`.env.example`** (env inventory) · **`bumblebee setup --help`** (guided hybrid/local flow)
- **Entity personality / YAML only:** `bumblebee create` (the full wizard can call this after env + Railway prep).

Use **`bumblebee setup --profile local`** for a single-machine stack without the Railway block.

## Gateway setup (hybrid home brain)

If you already know you want the **home inference stack** (bearer token + Cloudflare Tunnel ingress to the gateway + `.env`), use **`bumblebee gateway setup`**. It walks through tunnel configuration and gateway env vars **together**, then on Windows can run **`bumblebee gateway on`**. This is narrower than **`bumblebee setup`** (no Railway/entity flow).

The home gateway process needs **`pip install 'bumblebee[gateway]'`**. Bind **`127.0.0.1`**, terminate the tunnel **only** at the gateway (not a broad reverse proxy), bearer auth on **`INFERENCE_GATEWAY_TOKEN`**.

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
bumblebee gateway setup    # interactive: token, tunnel, .env, then optional stack start
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
      # Operators: Telegram user ids allowed to run /privacy (lock, open, allow, deny)
      # operator_user_ids: [123456789]
```

**Privacy from Telegram (operators):** set **`operator_user_ids`** to a non-empty list of numeric Telegram user ids (use **`/whoami`** in chat to read yours). Operators can use **`/private on`** (or **`/privacy lock`**) so only those operators can use the bot until they **`/privacy allow`** with a numeric id for guests, or **`/private off`** / **`/privacy open`** to go public again. The allowlist is stored in the entity database (`entity_state`) and survives restarts. While the DB lock is on, static **`allowed_user_ids`** in YAML is ignored; after **`/private off`**, YAML allowlists apply again if set.

4. **Optional — inline mode:** In BotFather run `/setinline` for the bot and pick a placeholder. Then in any chat you can type `@YourBot …` and get quick “about” cards (same identity story as `/start`).

**What you get**

- **`/start`** — Rich intro for **your entity** (quick-start actions and best practices).
- **`/help`** — Practical usage guide with examples.
- **`/commands [page] [filter]`** — Paginated (and filterable) command catalog.
- **`/status`**, **`/memories [count]`**, **`/feelings`** — Internal-state introspection in Telegram HTML.
- **`/me`** — Relationship snapshot: familiarity, warmth, trust, and interaction counts.
- **`/models`** and **`/ping`** — Runtime model configuration + liveness checks.
- **`/whoami`** — Shows your Telegram user id (for **`operator_user_ids`** / **`/privacy allow`**).
- **`/private on`** / **`/private off`** — Operators: quick private mode (same as **`/privacy lock`** / **`/privacy open`**).
- **`/privacy`** — Operators: status, lock, open, **`/privacy allow`** / **`/privacy deny`**. See YAML **`operator_user_ids`**.
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

**Rolling history compression** (entity `cognition.history_compression`, on by default): when in-memory chat exceeds `rolling_history_max_messages`, older turns are dropped from the rolling window and, if enabled, merged into a running text summary so long sessions stay within context limits without losing everything.

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

**Storage:** default is per-entity **SQLite** under `~/.bumblebee`. Set **`DATABASE_URL`** (and/or harness `memory.database_url`) to use **Postgres** — typical for **hybrid Railway** deployments. Automations and reminder rows live in the same store.

**Private journal** (`~/.bumblebee/entities/<name>/journal.md`) holds append-only reflections; the entity can read/write it via tools when **`automations.journal`** and harness/entity **`tools.journal`** allow it.

### Presence — embodied agency

An **always-on daemon** ticks continuously: emotions, drives, memory consolidation, and proactive initiative.

**Automations** (scheduled “routines”) run through APScheduler when **`automations.enabled`** is true on the entity: cron-style schedules, optional delivery to Telegram (or journal-only), emergence suggestions, and persisted definitions in the entity database.

**Multi-platform adapters** expose the same entity on CLI, Telegram, Discord, and elsewhere you configure, with consistent memory and emotional continuity.

Deeper design (inference boundary, hybrid trust model) is reflected in **`configs/default.yaml`**, gateway code under **`bumblebee/inference_gateway/`**, and comments in **`.env.example`**.

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

Tools are how the entity interacts with the world — not “services for the user,” but **senses and reach**. When `presence.tool_activity` is on, tool use can surface in chat with short status lines. Unless noted, tools run on the **deliberate** path only.

### Native tool reference

Built-in tools live under `bumblebee/presence/tools/` and register at entity startup. **`Harness key`** is the `tools.<key>` block in `configs/default.yaml` (override per entity with a top-level `tools:` block). A dash means the tool is **always registered** — there is no `enabled` gate for that name.

| Tool | Harness key | Default | What it does |
|---|---|:---:|---|
| `append_file` | — | on | Append to an allowed file path. |
| `browser_click` | `browser` | off | Click an element (Playwright). |
| `browser_navigate` | `browser` | off | Open a URL in the automated browser. |
| `browser_screenshot` | `browser` | off | Screenshot the current page. |
| `browser_type` | `browser` | off | Type into the page. |
| `cancel_reminder` | `reminders` | on | Cancel a scheduled reminder. |
| `check_process` | `shell` | on | Inspect a tracked background shell job. |
| `create_automation` | `automations` | on | Create a scheduled routine (cron, prompt, optional delivery). |
| `delete_automation` | `automations` | on | Remove a routine. |
| `describe_tool` | — | on | Full name, description, and JSON Schema for one tool. |
| `edit_automation` | `automations` | on | Edit routine fields (name, schedule, prompt, target). |
| `execute_javascript` | `code` | on | Run JavaScript in a sandboxed helper. |
| `execute_python` | `code` | on | Run Python in a sandboxed helper. |
| `fetch_url` | — | on | Fetch and extract text from a URL. |
| `generate_image` | `imagegen` | off | Text-to-image (Fal or local A1111-style API). |
| `get_current_time` | — | on | Current local date and time (timezone-aware when configured). |
| `get_news` | `news` | on | News headlines / topics. |
| `get_system_info` | `system` | on | Host / system snapshot. |
| `get_tts_voice` | `voice` | on | Current Edge-TTS voice id. |
| `get_weather` | `weather` | on | Weather for a location. |
| `get_youtube_transcript` | `youtube` | on | Transcript for a YouTube URL or id. |
| `kill_process` | `shell` | on | Stop a background shell job. |
| `list_automations` | `automations` | on | List routines. |
| `list_directory` | — | on | List an allowed directory. |
| `list_known_contacts` | `messaging` | on | Known people / routes from prior interactions. |
| `list_reminders` | `reminders` | on | List reminders. |
| `list_tts_voices` | `voice` | on | List Edge-TTS voices (filterable). |
| `read_file` | — | on | Read an allowed file path. |
| `read_journal` | `journal` | on | Read recent private journal entries. |
| `read_pdf` | `pdf` | on | Extract text from a PDF path. |
| `read_reddit` | `reddit` | on | Browse a subreddit. |
| `read_reddit_post` | `reddit` | on | Read a Reddit post thread. |
| `read_wikipedia` | `wikipedia` | on | Wikipedia summary / fetch. |
| `run_automation_now` | `automations` | on | Run a routine once now (daemon / scheduler must be up). |
| `run_background` | `shell` | on | Start a background shell command. |
| `run_command` | `shell` | on | Run a shell command (denylist + timeout). |
| `search_files` | — | on | Search by pattern under allowed roots. |
| `search_tools` | — | on | Search registered tools by keyword at runtime. |
| `search_web` | — | on | Web search (DuckDuckGo by default; optional Firecrawl). |
| `search_youtube` | `youtube` | on | Search YouTube. |
| `send_dm` | `messaging` | on | DM on Telegram or Discord (confirm flow; optional target listing). |
| `send_message_to` | `messaging` | on | Message a platform target or resolved person (confirm flow). |
| `set_reminder` | `reminders` | on | Schedule a reminder (DB + APScheduler). |
| `set_tts_voice` | `voice` | on | Set Edge-TTS voice for this process. |
| `speak` | `voice` | on | Send a TTS voice note to the active chat. |
| `toggle_automation` | `automations` | on | Enable or disable a routine. |
| `update_knowledge` | — | on | Add / update / remove `knowledge.md` sections (locked headings respected). |
| `write_file` | — | on | Write or create an allowed file. |
| `write_journal` | `journal` | on | Append a private journal entry. |

**Runtime list** — The table is the in-repo catalog; the **live** registry depends on YAML toggles, optional extras (`pip install 'bumblebee[voice]'`, `imagegen`, `browser`, etc.), and any **MCP** servers you attach. Use **`search_tools`** / **`describe_tool`** in chat, or **`/tools`** on CLI/Telegram where available, to see what this process actually exposed.

### Extending the stack (code + MCP)

Bumblebee is meant to **grow**: add new `@register_decorated` tools alongside the existing modules, or wrap external capability behind a thin adapter in `bumblebee/presence/tools/`. Keep schemas honest and respect harness policy for filesystem and shell.

**MCP (Model Context Protocol)** — Declare stdio MCP servers on the entity; their tools are registered **dynamically** when the process starts (names are **prefixed** in the registry so they stay distinct from native tools). Reconnects and multiple servers are supported through the same mechanism.

```yaml
mcp_servers:
  github:
    command: npx
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "ghp_xxx"
```

See **`configs/entities/example.yaml`** for a commented template. Anything that speaks MCP can extend the entity’s reach without forking the core harness.

**Zapier MCP (hosted tools):** Zapier exposes MCP over **Streamable HTTP** with a URL and Bearer secret from the Connect tab, while Bumblebee only spawns **stdio** servers today. You can still attach Zapier by running a local **stdio → Streamable HTTP** proxy (documented with example config), or plan a native remote client—see **[`docs/mcp-zapier.md`](docs/mcp-zapier.md)** for both approaches.

**Optional extras**

- **Voice:** `pip install 'bumblebee[voice]'` (Edge-TTS for `speak` / voice listing tools).
- **Imagegen:** `tools.imagegen.enabled: true` plus Fal (`FAL_API_KEY`, `pip install 'bumblebee[imagegen]'`) or `backend: local` + `local_url` for a local SD WebUI-compatible endpoint.
- **Browser:** `tools.browser.enabled: true` and `pip install 'bumblebee[browser]'` (Playwright) for the four `browser_*` tools.

## Knowledge system

Each entity may have a **`knowledge.md`** file — durable facts, opinions, and context the entity (or you) consider important. It is organized into `##` sections; embeddings help pull relevant sections into context. By default it lives next to **`memory.db`** (same directory as the configured SQLite file). When using **`DATABASE_URL`** / Postgres without a local DB file, the harness keeps **`knowledge.md`** under **`~/.bumblebee/entities/<name>/`** unless you customize paths.

```bash
bumblebee knowledge canary    # create if missing, open in $EDITOR
```

### Journal

A separate **`journal.md`** (default path **`~/.bumblebee/entities/<name>/journal.md`**) is for the entity’s **private**, append-only reflections — often used after automation runs. Open it in your editor with:

```bash
bumblebee journal canary
```

The **`read_journal`** / **`write_journal`** tools respect entity **`automations.journal`** and harness **`tools.journal`** toggles.

Example **knowledge** excerpt:

```markdown
## [locked] about yourself
You are Canary, the first entity on Bumblebee.
(Locked sections are only edited by the creator, not via `update_knowledge`.)

## music taste
Got into shoegaze after talking about My Bloody Valentine.
```

**Locked sections** (`## [locked] ...`) are readable but not mutable through the entity tool. Unlocked sections can be updated when the deliberate path invokes `update_knowledge`.

## Configuration

Harness defaults live in **`configs/default.yaml`**. Per-entity overrides go in **`configs/entities/<name>.yaml`** — models, **`cognition`** (including **`rolling_history_max_messages`** and **`history_compression`**), **`presence`**, **`automations`** (schedules, emergence, journal), optional top-level **`tools:`**, **`mcp_servers`**, Firecrawl, etc.

**Deployment / inference:** `deployment.mode` (`local` | `hybrid_railway`) and `inference` keys align with **`.env`** (`BUMBLEBEE_DEPLOYMENT_MODE`, `BUMBLEBEE_INFERENCE_PROVIDER`, tunnel **`BUMBLEBEE_INFERENCE_BASE_URL`**, gateway token).

**Hybrid attachments:** harness **`attachments.backend`** can be **`object_s3_compat`** with S3-compatible env vars (see **`.env.example`**) for durable blobs when the worker runs in the cloud.

**Timezone:** optional **`BUMBLEBEE_TIMEZONE`** (IANA name) for wall-clock prompts and **`get_current_time`**.

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

Entity YAML can override chat models under **`cognition`** (for example **`reflex_model`** / **`deliberate_model`**).

## CLI reference

```bash
bumblebee setup [--profile ask|hybrid|local]   # .env + home stack + Railway + entity wizard
bumblebee gateway setup                        # interactive: gateway token + Cloudflare tunnel + .env
bumblebee gateway on|off|status|restart        # Windows: Ollama + inference gateway + cloudflared
bumblebee create                               # genesis wizard — new entity YAML
bumblebee talk <entity>                        # CLI conversation (no daemon)
bumblebee run <entity>                         # daemon + configured platforms (local or hybrid)
bumblebee worker <entity>                      # daemon + platforms, no CLI (Railway worker role)
bumblebee api [--host 0.0.0.0] [--port 8080]  # HTTP health API (extras: pip install 'bumblebee[api]')
bumblebee status <entity>                      # emotional state, drives, paths
bumblebee evolve <entity>                      # force one trait evolution cycle (debug/advanced)
bumblebee knowledge <entity>                   # open knowledge.md in $EDITOR
bumblebee journal <entity>                     # open journal.md in $EDITOR
bumblebee recall <entity> "<query>"            # semantic search over episodic memory
bumblebee wipe <entity> [--yes]                # clear rolling chat + wipe SQLite/Postgres memory
bumblebee export <entity> <dest_dir>           # backup entity YAML + DB
bumblebee import <bundle_dir>                  # restore entity bundle

bumblebee talk canary --ollama                 # start local Ollama if needed
bumblebee run canary --ollama --pull-models    # also pull models from config
```

`gateway` subcommands other than **`setup`** use **`scripts/gateway.ps1`** and are intended for **Windows**; **`gateway setup`** is interactive documentation + **`.env`** on any OS, then offers **`gateway on`** when the script is available.

## Hardware guide

| GPU VRAM | Notes |
|---|---|
| **8 GB** | Prefer smaller / quantized models; reflex-only or very tight dual-model setups. |
| **16 GB** | Common target for Gemma 4 family; close other GPU apps if needed. |
| **24+ GB** | Comfortable dual-model + headroom. |
| **32+ GB** | Room for larger deliberate models or more context. |

MoE-style models keep active parameters per token lower than full dense size; exact fit depends on context length, thinking budget, and concurrent platforms.

## What Bumblebee is built for

Bumblebee is optimized for **sustained entity existence**: local **Gemma** / **Ollama** inference, memory that behaves like **life story** (episodes, people, beliefs, narrative), **drive-shaped** initiative, and an **inner voice** that does not have to be shown verbatim to matter.

If you want a stack whose north star is **embodiment**, **continuity**, and **freedom within character** — this is the shape of it.

## Philosophy

One question runs through the harness: **how does this entity exist more fully?**  
Everything else — platforms, tools, memory layers, hybrid deployment — exists to honor that.

## License

Apache 2.0

<p align="center">
  <strong>🐝</strong><br/>
  <sub>Apache 2.0 — built with Gemma by Google DeepMind</sub>
</p>
