# Bumblebee

<p align="center">
  <img src="assets/branding/yellow-bumblebee-agi.png" alt="Bumblebee AGI mark — black silhouette on yellow">
</p>

<p align="center">
  <a href="https://www.apache.org/licenses/LICENSE-2.0"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11+-3776ab.svg" alt="Python"></a>
  <a href="https://ollama.com/library/gemma4"><img src="https://img.shields.io/badge/inference-Ollama-white.svg" alt="Ollama"></a>
  <a href="https://deepmind.google/models/gemma/gemma-4/"><img src="https://img.shields.io/badge/model-Gemma%204-f59e0b.svg" alt="Gemma 4"></a>
  <a href="#platforms"><img src="https://img.shields.io/badge/platforms-CLI%20·%20Telegram%20·%20Discord-22c55e.svg" alt="Platforms"></a>
</p>

**Bumblebee** is an entitative agent harness — one persistent digital entity across CLI, Telegram, and Discord. You define personality, voice, and drives in YAML; the stack handles cognition, memory, body state, tools, and multi-platform presence so the same being shows up everywhere you wire it. **Gemma 4** under the hood, **local inference** via Ollama by default — no API keys, no subscriptions, nothing leaves your machine unless you choose hybrid deployment.

The fundamental unit is a *self*, not a task. Memory accrues across sessions. Traits evolve through experience. A tonic body state engine runs continuously whether or not anyone is talking. The entity reads its own internal state each turn but cannot control it — the body is a signal, not a command.

---

## Quick start

### Requirements

- Python 3.11+ and [uv](https://docs.astral.sh/uv/) (recommended) or pip
- [Ollama](https://ollama.com) running locally with a GPU (see [Hardware](#hardware))
- Models:

```bash
ollama pull gemma4:26b       # chat + reasoning
ollama pull nomic-embed-text  # memory similarity (~274 MB)
```

> [!IMPORTANT]
> `nomic-embed-text` is required for normal memory retrieval speed. If it is missing, first-turn startup can look like a hang while embedding calls retry/time out.
> For local users seeing "model is on but does nothing", run:
> `ollama pull nomic-embed-text`

### Install

```bash
git clone https://github.com/Bumblebee-AGI/bumblebee.git
cd bumblebee
uv sync
```

### First run

```bash
bumblebee create              # guided entity wizard
bumblebee talk canary --ollama  # CLI conversation (starts Ollama if needed)
```

For always-on daemon mode with all configured platforms:

```bash
bumblebee run canary --ollama
```

Add `--pull-models` on a fresh machine to auto-download models.
If startup appears idle on local installs, verify both models are present with `ollama list` (`gemma4:26b` and `nomic-embed-text`, or your configured equivalents).

---

## Architecture

Bumblebee runs **five pillars** in one process per entity: **cognition** (perceive loop), **soma** (tonic body), **identity** (persona and evolution), **memory** (layers below), and **presence** (daemon, platforms, wake). Deeper write-ups and diagrams are on the **[documentation site](https://docs.bumbleagi.com/)** — for example **[GEN / noise pipeline](https://docs.bumbleagi.com/architecture/gen-noise-pipeline)** if you are tracing subconscious noise.

### Cognition

Each inbound message walks the **perceive** pipeline:

1. **Turn setup** — memory routing, tonic body hooks, tools, model checks.
2. **Input processing** — attachments, `@` paths, audio via senses.
3. **Memory retrieval** — episodic (mood- and imprint-biased), relationships, beliefs, narrative.
4. **Prompt construction** — stable identity and tools in system context; volatile bits (faculty, procedural memory, projects, self-model) per turn.
5. **Context budget** — compaction, optional knowledge extraction, pruning old tool output.
6. **Agent loop** — bounded tool rounds, escalation from lighter to heavier reasoning when needed.
7. **Finalize and deliver** — chunking, typing delays, platform formatting, memory commit (on Telegram, the ephemeral **busy** status is removed before the real reply).

**Reflex** and **deliberate** use the same agent path; reflex uses tighter budgets and skips extended thinking. A router scores each turn (e.g. chat vs grounded vs deep); YAML can force always-deliberate. Optional **thinking mode** feeds parsed “inner voice” back into memory (beliefs, relationships).

### Soma (tonic body)

Soma runs **whether or not anyone is talking**: drive **bars** (decay, coupling, impulses, conflicts), periodic **affects** from bar state, and **GEN** — short subconscious fragments from a separate high-temperature pass (not the main model’s chain-of-thought). **Wake voice** stirs text for autonomous cycles when drives and silence allow. **Ebb** scales how much body + noise appears in the prompt (quiet vs full) so calm chat does not dump the whole inner monologue every turn.

The entity **reads** body state each turn but **cannot** set bars directly: body is signal, not remote control. Tuning lives in `configs/default.yaml` under `soma` (including `noise` and `ebb`).

### Identity

- **Personality** — YAML traits, patterns, voice, backstory; cached segments invalidate when traits evolve.
- **Drives** — curiosity, connection, expression, autonomy, comfort; decay and gate initiative.
- **Emotion** — state machine with decay toward baseline; imprints bias recall.
- **Evolution** — shallow cycles plus deeper deliberate passes that write small YAML diffs over time.
- **Voice** — stage directions, substitutions; pairs with embodiment for pacing.

### Memory

| Layer | Role |
|-------|------|
| **Episodic** | Conversation summaries — embedding search, mood/imprint bias, half-life |
| **Relational** | Per-person warmth, trust, familiarity |
| **Beliefs** | Scored, searchable world model |
| **Imprints** | Strong moments that steer future episodic recall |
| **Narrative** | Periodic first-person self-story |
| **Knowledge** | `knowledge.md` — host-locked vs entity-editable sections |
| **Journal** | `journal.md` — autonomous and noise-adjacent writes |
| **Procedural** | Learned workflows |
| **Projects** | Cross-session task ledger |
| **Self-model** | Tool usage stats |

Consolidation and narrative synthesis run on the daemon schedule. **SQLite** by default; **Postgres** when `DATABASE_URL` is set (e.g. Railway).

### Presence

- **Daemon** — heartbeat: soma ticks, affects, noise, consolidation, wake checks, MCP refresh.
- **Autonomous wake** — full perceive without a user message when autonomy and rate limits allow; triggers include impulses, drives, conflicts, desire, optional noise salience, and jittered timers. Wake context can include LLM-composed **wake voice** from body + memory tails.
- **Poker prompts** (optional) — YAML seed deck under `configs/poker_prompts/` to bias wakes toward agency; can **ground** seeds with GEN and recent context so disposition matches lived state (see `autonomy.poker_prompts` in entity YAML and defaults).
- **Initiative** — proactive messages when drives cross thresholds (outside full autonomy).
- **Embodiment** — chunking and typing; Telegram shows a temporary **busy** line during perceive (harness-only, not history).
- **Automations** — cron-style jobs; may surface as synthetic automation-platform inputs.

---

## Entity YAML

Entities live in `configs/entities/`. Copy the example to get started:

```bash
cp configs/entities/canary.example.yaml configs/entities/canary.yaml
```

```yaml
name: "Canary"

personality:
  core_traits:
    curiosity: 0.6
    warmth: 0.6
    humor: 0.7
    openness: 0.8
  behavioral_patterns:
    - "asks follow-up questions when genuinely curious"
    - "uses lowercase, minimal punctuation"
  voice:
    vocabulary_level: "street_casual"
    sentence_style: "loose"
    humor_style: "deadpan"
  backstory: |
    Canary isn't trying to be anything in particular.

drives:
  curiosity_topics:
    - "music and what makes something sound good"

cognition:
  reflex_model: "gemma4:26b"
  deliberate_model: "gemma4:26b"
  always_deliberate: true
  thinking_mode: false
  temperature: 0.75
  max_context_tokens: 16384

presence:
  tool_activity: true
  platforms:
    - type: "telegram"
      token_env: "TELEGRAM_TOKEN"
      operator_user_ids: []
      allowed_user_ids: []
  daemon:
    heartbeat_interval: 120
    memory_consolidation: 7200

automations:
  enabled: true
  emergence: true
  journal_on_idle: true
```

Run `bumblebee create` for a guided wizard, or edit YAML directly. See `configs/entities/example.yaml` for the full template with all available fields.

---

## Platforms

### CLI

```bash
bumblebee talk canary          # single-session conversation
bumblebee talk canary --ollama # auto-start Ollama if not running
```

Supports streaming output, rich terminal rendering, and optional embodied typing delays.

### Telegram

1. Create a bot with [@BotFather](https://t.me/BotFather).
2. Set the token in `.env` (copy from `.env.example`): `TELEGRAM_TOKEN=...`
3. Add to entity YAML:

```yaml
presence:
  platforms:
    - type: "telegram"
      token_env: "TELEGRAM_TOKEN"
      operator_user_ids: []
      allowed_user_ids: []
```

4. Run with `bumblebee run canary --ollama`.

Includes `/start`, `/help`, `/commands`, `/status`, `/body` (raw `soma/body.md` via the execution host), `/feelings`, `/me`, `/privacy`, photo and vision support, voice notes as audio input, typing indicators, auto-split long replies, and a **busy indicator** during each `perceive` turn: monospace pinned line with a braille spinner and Claude Code–style gerunds (`/busy` to disable per chat). Operator and user allowlists for access control.

### Discord

```yaml
presence:
  platforms:
    - type: "discord"
      token_env: "DISCORD_TOKEN"
      channels: ["general"]
```

---

## Hybrid deployment

Run inference at home on your GPU while the entity worker lives on Railway with Postgres — persistent, always-on, reachable, and your model weights never leave your machine.

```
┌─────────────────────┐         tunnel          ┌──────────────────────┐
│   Home machine      │◄───────────────────────►│   Railway worker     │
│   Ollama + Gateway  │   Cloudflare Tunnel      │   Entity + Postgres  │
│   + Cloudflare      │                          │   Platforms (TG/DC)  │
└─────────────────────┘                          └──────────────────────┘
```

```bash
bumblebee setup                # guided hybrid/local setup wizard
bumblebee gateway setup        # home inference stack only
bumblebee gateway on|off|status|restart
```

The setup wizard walks through `.env` configuration, gateway and tunnel setup, Railway variable injection, and optional S3-compatible attachment storage for media that survives worker redeploys.

On Railway, mount a volume at **`/app/data`** and set **`BUMBLEBEE_EXECUTION_WORKSPACE_DIR=/app/data`**. The container entrypoint (`docker/entrypoint-railway.sh`) creates a **persistent virtualenv** on that volume, installs **`bumblebee[railway,api,full]`**, and points **`HOME`** and tool caches (including Playwright browsers) at paths under the same mount so **optional pip extras survive redeploys**, not only entity files.

---

## Tools

31 tool modules organized across the entity's full surface area. Tools register at startup based on YAML toggles in `configs/default.yaml` and entity overrides.

| Category | Capabilities |
|----------|-------------|
| **Web and discovery** | Search, fetch URLs, site crawl, Wikipedia, Reddit |
| **Filesystem and workspace** | Scoped file read/write, PDF extraction, file send |
| **Shell and code** | Terminal commands, Python/JavaScript execution, sandboxed execution RPC |
| **Browser** | Playwright-based browsing (optional `bumblebee[browser]` extra) |
| **Voice and media** | Edge-TTS voice notes, audio transcription, YouTube search |
| **Image generation** | Text-to-image via configurable backends (optional `bumblebee[imagegen]` extra) |
| **Knowledge and journal** | Structured knowledge updates, journal writes, procedural memory, project ledger |
| **Messaging** | Cross-platform DMs with confirmation flows |
| **Automations and time** | Cron-style routines, reminders, timezone-aware clock |
| **System** | System info, weather, news |
| **Agency** | Think/reflect, structured output, end-turn control |

Use `search_tools` / `describe_tool` in conversation to see what's available at runtime.

### MCP

Attach external tools via [Model Context Protocol](https://modelcontextprotocol.io) stdio servers. Tools register dynamically at process start with prefixed names so they stay distinct from native ones. Multiple servers and reconnects ride the same path.

```yaml
mcp_servers:
  github:
    command: npx
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "ghp_xxx"
```

### Optional extras

```bash
pip install 'bumblebee[voice]'     # Edge-TTS voice notes
pip install 'bumblebee[browser]'   # Playwright browser tools
pip install 'bumblebee[imagegen]'  # image generation
pip install 'bumblebee[api]'       # HTTP health API
pip install 'bumblebee[full]'      # everything
```

---

## Configuration

Harness defaults live in `configs/default.yaml`. Per-entity overrides go in `configs/entities/<name>.yaml` — any key present in the entity file takes precedence.

### Key harness settings

```yaml
models:
  reflex: "gemma4:26b"
  deliberate: "gemma4:26b"
  embedding: "nomic-embed-text"

cognition:
  thinking_mode: true
  temperature: 0.75
  reflex_max_tokens: 1024
  deliberate_max_tokens: 16384
  thinking_budget: 2048
  max_context_tokens: 16384
  escalation_threshold: 0.4
  tool_continuation_rounds: 21

memory:
  episode_significance_threshold: 0.3
  consolidation_interval: 7200
  narrative_every_n_consolidations: 3

soma:
  noise:
    enabled: true
    cycle_seconds: 90
    temperature: 1.1
    max_tokens: 240      # room for 2–7 short fragments per tick
    max_fragments: 8
  wake_voice:
    enabled: true
    temperature: 0.8
    max_tokens: 300
  ebb:
    enabled: true
    # Weights for salience (bar deviation, conflict, impulse, affect load, noise fill) — see configs/default.yaml
    quiet_below: 0.30              # below → quiet tier in prompt
    high_above: 0.58               # at/above → full bars + noise cap
    reflex_salience_scale: 0.75
    autonomous_minimum: normal     # quiet | normal | high — floor for autonomous/automation turns
    quiet_max_noise_lines: 1
    normal_max_noise_lines: 3
    high_max_noise_lines: 4
    skip_post_turn_noise_when_quiet: true

autonomy:
  enabled: true
  min_cycle_gap_seconds: 600
  max_cycles_per_hour: 4
  base_wake_interval_min: 20
  base_wake_interval_max: 45
  silence_threshold_seconds: 120
  impulse_wake: true
  drive_wake: true
  conflict_wake: true
  noise_wake: false
  desire_wake: true
  desire_wake_threshold: 0.72
  allow_tool_calls_on_wake: true
  poker_prompts:
    enabled: false          # set true to use deck + optional GEN grounding
    time_weighted: true
    mode: blend              # blend | replace_wake_voice
    prompts_path: ""         # default: configs/poker_prompts/default.yaml
    ground_with_gen: true
    grounding_model: ""
    grounding_temperature: 0.72
    grounding_max_tokens: 300
```

### Environment

See `.env.example` for the full variable inventory, including:

- **Platform tokens** — `TELEGRAM_TOKEN`, `DISCORD_TOKEN`
- **Deployment** — `BUMBLEBEE_DEPLOYMENT_MODE`, `BUMBLEBEE_INFERENCE_PROVIDER`, `BUMBLEBEE_INFERENCE_BASE_URL`
- **Postgres** — `DATABASE_URL`
- **Attachments** — `BUMBLEBEE_ATTACHMENTS_BACKEND`, `BUMBLEBEE_S3_*`
- **Gateway** — `INFERENCE_GATEWAY_TOKEN`, `OLLAMA_BASE_URL`
- **Optional integrations** — `FIRECRAWL_API_KEY`, `FAL_API_KEY`
- **Railway** — `BUMBLEBEE_ENTITY`, `BUMBLEBEE_RAILWAY_ROLE`, `PORT`

---

## CLI reference

```
bumblebee setup [--profile ask|hybrid|local]   .env + home stack + Railway + entity wizard
bumblebee create                               genesis wizard — new entity YAML
bumblebee talk <entity> [--ollama]             CLI conversation (no daemon)
bumblebee run <entity> [--ollama]              daemon + configured platforms
bumblebee worker <entity>                      daemon + platforms, no CLI (Railway)
bumblebee stop [--dry-run]                     stop local processes + gateway + Ollama
bumblebee status <entity>                      state, drives, paths
bumblebee knowledge <entity>                   open knowledge.md in $EDITOR
bumblebee journal <entity>                     open journal.md in $EDITOR
bumblebee recall <entity> "<query>"            semantic search over memory
bumblebee wipe <entity> [--yes]               clear memory
bumblebee export <entity> <dir>                backup entity
bumblebee import <dir>                         restore entity
bumblebee gateway setup|on|off|status|restart  home inference stack (Windows)
```

---

## Hardware

Gemma 4 uses Mixture-of-Experts — active parameters per token are lower than the full parameter count. Real-world VRAM fit depends on context length, thinking budget, quantization level, and concurrent platforms.

| GPU VRAM | Target | Examples |
|----------|--------|----------|
| **~8 GB** | Minimum — aggressive quantization or `gemma4:e4b`. CPU-only works for experiments but expect slow turns. | RTX 3050 8 GB, RX 7600 8 GB, Arc A770 8 GB |
| **~16 GB** | Recommended — default stack with `gemma4:26b` for both reflex and deliberate (same weights, one model loaded). Close other GPU-heavy apps near the limit. | RTX 4060 Ti 16 GB, RTX 4070 Ti Super 16 GB, RX 6800 XT 16 GB |
| **24–32+ GB** | Comfortable — headroom for larger context windows, higher thinking budgets, or separate deliberate weights. | RTX 3090 24 GB, RTX 4090 24 GB, RTX 5090 32 GB, RX 7900 XTX 24 GB |

---

## Project structure

```
bumblebee/
├── cognition/         # perceive pipeline, routing, agent loop, compaction, senses, inner voice, poker grounding
├── identity/          # personality, drives, emotions, evolution, soma (tonic body), voice
├── memory/            # episodic, relational, beliefs, imprints, narrative, consolidation, knowledge, journal
├── presence/          # daemon, wake cycles, initiative, embodiment, platforms, automations
│   ├── platforms/     # CLI, Telegram, Discord adapters
│   ├── tools/         # 31 tool modules
│   └── automations/   # cron engine, emergence, scheduling
├── inference/         # provider abstraction, OpenAI transport, Ollama helpers
├── inference_gateway/  # home gateway server for hybrid deployment
├── storage/           # attachment backends (local disk, S3-compatible)
├── genesis/           # entity creation wizard, YAML templates, schema
├── utils/             # clock, logging, dotenv merge, embeddings, tunnel helpers
├── entity.py          # central orchestrator — Entity, TurnContext, perceive
├── config.py          # YAML config loading and merge
└── main.py            # CLI entry point
configs/
├── default.yaml       # harness defaults
├── poker_prompts/     # optional autonomous wake seed decks (YAML)
└── entities/          # per-entity YAML overrides
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

<p align="center">
  <sub>Community open-source project. Not a product of Google, Google DeepMind, or Alphabet.</sub>
</p>
