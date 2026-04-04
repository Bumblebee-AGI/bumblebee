# Bumblebee

<p align="center">
  <img src="bumblebee.png" alt="Bumblebee — stylized bee illustration" width="520">
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

**Quick links:** [Requirements](#requirements) · [Install & usage](#install) · [Telegram](#telegram) · [Platforms](#platforms) · [Tools](#tools) · [Knowledge](#knowledge-system) · [CLI reference](#cli-reference) · [Architecture](#architecture)

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

### Built-in tools

| Tool | Description |
|---|---|
| `search_web` | Web search (DuckDuckGo by default; optional Firecrawl when configured). |
| `fetch_url` | Fetch and extract text from a URL. |
| `read_file` | Read allowed local files (read-focused; bounded by harness policy). |
| `get_current_time` | Current local date and time. |
| `update_knowledge` | Curate `knowledge.md` sections (deliberate path; respects locked headings). |

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
bumblebee create                      # genesis wizard — new entity
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
