# TODO: Hybrid Railway Setup — Volume, Persistence, and Knowledge Seeding

Captured from the session on 2026-04-08. All the code changes are done;
this is the documentation, wizard integration, and onboarding work that
needs to follow.

## Context

The hybrid Railway deployment now has three persistence layers:
1. **Postgres** (volume-backed) — episodic memory, relationships, beliefs, entity_state (including soma bar state)
2. **Worker volume** (`/app/data`) — knowledge.md, journal.md, soma checkpoint, workspace files the entity creates
3. **Ephemeral container disk** (`/app`, `~/.bumblebee`) — code, configs, logs — lost on redeploy

Users setting up hybrid need to understand this and get the volume created
before their entity starts writing files that disappear.

## What needs to happen

### 1. Documentation: `docs/hybrid-railway-persistence.md`

A standalone guide covering:

- The three persistence layers and what lives where
- Why `BUMBLEBEE_EXECUTION_WORKSPACE_DIR=/app/data` matters
- Step-by-step: creating the volume via Railway CLI
  ```bash
  railway service bumblebee-worker
  railway volume add --mount-path /app/data
  railway variable set BUMBLEBEE_EXECUTION_WORKSPACE_DIR=/app/data
  railway redeploy --service bumblebee-worker --yes
  ```
- What files land on the volume after setup:
  ```
  /app/data/
    knowledge.md          — entity knowledge (semantic retrieval)
    journal.md            — private reflections
    soma/                 — tonic body bar state checkpoint
    *.md, *.txt, etc.     — files the entity creates via write_file
  ```
- How to seed knowledge.md for a new entity
- How to verify persistence survives a redeploy
- How to back up the volume (railway volume list, SSH + tar)
- Troubleshooting: files disappearing = volume not mounted or WORKSPACE_DIR not set

### 2. Setup wizard enhancement: `bumblebee setup --profile hybrid`

The setup wizard (`bumblebee/setup.py` or wherever the hybrid flow lives)
should add these steps when the user chooses hybrid:

- **Check for Railway CLI** (`railway --version`)
- **Check/create volume**: `railway volume list` → if no volume on bumblebee-worker at /app/data, offer to create it
- **Set WORKSPACE_DIR**: check `railway variable list` for `BUMBLEBEE_EXECUTION_WORKSPACE_DIR` → set if missing
- **Seed knowledge.md**: offer to write a starter knowledge.md to the volume (either via SSH or by setting it as a Railway variable that the worker writes on startup)
- **Verify**: after redeploy, SSH and confirm /app/data exists and is writable

The wizard should explain each step clearly so users understand WHY, not
just run commands blindly.

### 3. README updates

The existing hybrid section in README.md should reference:
- The new persistence guide
- The volume requirement (not optional — files WILL be lost without it)
- The `BUMBLEBEE_EXECUTION_WORKSPACE_DIR` env var

### 4. First-run entity bootstrap

Consider adding a startup check in the worker: if `knowledge.md` doesn't
exist at the configured path, write a minimal template. This way a fresh
deploy gets a knowledge file automatically instead of requiring manual
seeding. Could be in Entity.__init__ or in _run() in main.py.

Template could be:
```markdown
## [locked] about yourself
{entity_name} is an entity running on the Bumblebee harness.

## notes
(empty — the entity will fill this in as it learns)
```

### 5. Railway filesystem skill update

Update `.cursor/skills/railway-filesystem/SKILL.md` to include:
- The persistent vs ephemeral path distinction
- Volume verification commands
- How to seed/edit knowledge.md via SSH

## Dependencies

All code changes are already merged:
- `config.py`: knowledge_path(), journal_path(), soma_dir() respect BUMBLEBEE_EXECUTION_WORKSPACE_DIR
- `entity_state` table: soma bar state persists in Postgres
- Railway volume: already created on the current deploy at /app/data
- knowledge.md: already seeded on the current container

This is purely documentation + wizard work. No harness code changes needed.
