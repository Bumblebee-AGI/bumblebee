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

---

## Architecture

Five interconnected pillars run as a single process per entity:

### Cognition

The perceive pipeline runs on every inbound message:

1. **Turn setup** — route memory, fire tonic body events, bootstrap tools, check model availability.
2. **Input processing** — handle attachments, expand `@` paths, transcribe audio via senses.
3. **Memory retrieval** — episodic recall (mood-biased, imprint-weighted), relationship context, belief snapshot, narrative summary.
4. **Prompt construction** — stable identity and tool schemas in the system prompt; volatile turn context (faculty, procedural memory, project ledger, self-model) injected per-message.
5. **Context budget** — proactive compaction before inference: structured summaries, optional knowledge extraction, old tool-output pruning.
6. **Agent loop** — bounded multi-step tool execution with configurable continuation rounds, post-tool nudges, and escalation from reflex to deliberate when complexity warrants it.
7. **Finalize and deliver** — embodied chunking with typing delays, platform-native formatting, memory commit.

Reflex and deliberate share the same `DeliberateCognition` execution path — reflex is not a stripped-down caller, it uses the same agent loop with tighter token budgets and thinking disabled. Routing (`CognitionRouter`) classifies turns as CHAT, GROUNDED, EXACT, or DEEP via heuristic scoring and optional reflex-model classification, with an `always_deliberate` override.

When `thinking_mode` is enabled, the deliberate profile requests extended thinking from the model. Thinking output is parsed by `InnerVoiceProcessor` for cues, belief updates, and relationship hints that feed back into memory.

### Soma (tonic body state)

A continuous internal-experience engine independent of conversation, implemented as `TonicBody`:

- **Bar engine** — quantitative drive bars with natural decay, momentum, inter-bar coupling (`when`/`effect` DSL), impulses, and conflicts. Bars tick on a configurable cadence whether or not the entity is in conversation.
- **Affect engine** — periodically derives felt affects from a vocabulary of ~50 textures via an LLM pass against current bar state and recent context.
- **Generative Entropic Noise (GEN)** — a second model (or the reflex model at high temperature) produces raw associative thought fragments between turns, giving the entity a continuous stream of consciousness. Configurable cycle time, temperature, and fragment budget. Those fragments are also the raw material for **optional poker grounding** on autonomous wake: when `autonomy.poker_prompts.ground_with_gen` is enabled, a short reflex pass weaves a deck seed together with GEN, soma events, and recent memory context so the wake disposition is anchored in what the entity is actually living (see **Presence → Autonomous wake**).
- **Wake voice** — subconscious stirring that generates prompts for autonomous wake cycles; the entity can initiate conversation on its own when internal state warrants it.
- **Ebb** (`soma.ebb`) — salience-based scaling of how much **body + GEN** is injected into each **perceive** prompt. Bars, affects, conflicts, impulses, and GEN keep updating in the background; a 0–1 **salience** score (weighted blend of deviation from resting bar values, active conflict/impulse intensity, affect load, and GEN buffer fill) maps to **quiet**, **normal**, or **high** presentation. Quiet tiers use compact drive lines and fewer noise fragments; high matches the full markdown layout. **Reflex** turns multiply salience by `reflex_salience_scale` so routine replies stay lighter. **Autonomous** and **automation** platforms apply `autonomous_minimum` (default **normal**) so internal wake cycles are not stuck in whisper mode. The on-disk `body.md` flush and operator-facing renders still use the full layout. With `skip_post_turn_noise_when_quiet: true`, GEN is not regenerated after a turn when the tier is quiet, so the subconscious does not constantly shout during calm chat. Set `ebb.enabled: false` to always inject the full body block (legacy behavior). Defaults live in `configs/default.yaml`.

The entity reads a rendered body-state summary each turn (tier chosen as above when ebb is enabled). It sees its own drives, affects, and noise — but cannot set bar values directly. That separation is deliberate: the body provides signal the mind interprets, not commands it executes.

### Identity

- **Personality engine** — builds the system prompt from YAML-defined core traits, behavioral patterns, voice configuration, backstory, and anti-assistant voice blocks. Prompt segments are cached and invalidated on trait evolution.
- **Drive system** — curiosity, connection, expression, autonomy, and comfort drives with natural decay. Drives gate initiative and influence routing.
- **Emotion engine** — FSM-style emotional state with decay toward baseline and imprint-based recall weighting.
- **Evolution engine** — shallow stats cycles plus deep cycles that use the deliberate model to produce micro trait, behavior, and curiosity updates persisted as YAML diffs. Traits drift naturally through experience over time.
- **Voice controller** — expression metadata, stage-direction stripping, substitution rules. Works with `Embodiment` for typing delays and natural chunk pacing.

### Memory

| Layer | What it stores | How it works |
|-------|---------------|--------------|
| **Episodic** | Narrative summaries of conversations | Significance scoring, mood-biased embedding search, imprint-weighted recall with configurable half-life |
| **Relational** | Per-person relationship state | Warmth, trust, familiarity scores; upserted after each conversation |
| **Beliefs** | Categorized world beliefs | Confidence-scored, embedding-indexed, decayable |
| **Imprints** | Emotional imprints from significant moments | Affinity multipliers that bias future episodic recall toward emotionally resonant memories |
| **Narrative** | First-person self-story | Synthesized periodically from episodes, relationships, and beliefs into a coherent identity narrative |
| **Knowledge** | Structured `knowledge.md` file | Section-based, with `[locked]` host-only sections and unlocked sections the entity can update |
| **Journal** | `journal.md` maintained by the entity | Written to autonomously during wake cycles and noise processing |
| **Procedural** | Skill and procedure recall | Task-oriented memory for learned workflows |
| **Projects** | Project ledger | Tracks ongoing work items across sessions |
| **Self-model** | Introspective tool-usage stats | JSON record of tool successes, failures, and patterns |

Consolidation runs on a daemon interval. Episode significance decays over time. The narrative synthesizer runs every N consolidation cycles, producing a first-person story the entity uses as part of its identity context.

Storage backends: **SQLite** by default (local file per entity), **Postgres** when `DATABASE_URL` is set (typical for hybrid/Railway deployment).

### Presence

- **Daemon** — APScheduler-based heartbeat that ticks soma bars, refreshes affects and noise, runs memory consolidation, checks wake-cycle triggers, and refreshes MCP server connections.
- **Autonomous wake** — when `autonomy.enabled` is true and silence/rate limits allow, the daemon runs a full `perceive` cycle without an external user message. Triggers include soma impulses, drives over threshold, active conflicts, synthesized desire pressure, optional noise salience, and a randomized timer between `base_wake_interval_min` and `base_wake_interval_max`. Context includes the wake reason, platform channels, top desire pressures, and **wake voice** (`soma.wake_voice`): an LLM-composed first-person stirring from body state + memory tails.
- **Poker prompts** (optional, `autonomy.poker_prompts`) — a YAML **deck of loose seeds** in `configs/poker_prompts/` (bundled `default.yaml`, or your own via `prompts_path`). Seeds bias the cycle toward real-world agency: explore, research, build, message, learn, try something new — always as **taste**, not scripted homework. One line is selected per wake (**time-of-day weighted** across `low` / `medium` / `high` energy). With **`ground_with_gen: true`** (default when poker is enabled in config), `bumblebee/cognition/poker_grounding.py` runs a short reflex call that **braids that seed with GEN fragments, recent soma events, journal tail, last conversation, and relationship blurbs**, so the internal disposition emerges from perception instead of the static file alone. **`mode: blend`** keeps wake voice and the disposition block; **`replace_wake_voice`** skips the wake-voice LLM and uses only the (optionally grounded) disposition. Set `poker_prompts.enabled: true` in harness YAML to turn this on.
- **Initiative engine** — proactive messaging when drives cross thresholds (non-autonomy path).
- **Embodiment** — message chunking, typing simulation, and platform-native delivery pacing.
- **Automations** — cron-style routines with emergence (the entity can create its own scheduled routines). Automations fire as synthetic `Input` with `platform="automation"`.

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

Includes `/start`, `/help`, `/status`, `/feelings`, `/me`, `/privacy`, photo and vision support, voice note transcription, typing indicators, and auto-split long replies. Operator and user allowlists for access control.

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
    max_tokens: 150
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
