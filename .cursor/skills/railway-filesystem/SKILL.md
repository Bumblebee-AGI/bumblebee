---
name: railway-filesystem
description: Access and inspect files on the Railway worker container filesystem. Use when the user wants to see what files their bumblebee agent has written, read file contents from the Railway container, or debug workspace issues on the deployed worker.
---

# Railway Filesystem Access

How to inspect files on the bumblebee worker container running on Railway.

## Prerequisites

- Railway CLI installed and authenticated (`railway login`)
- Project linked (`railway link` or already linked via `railway.toml`)
- Service linked to the worker: `railway service bumblebee-worker`

## Quick Commands

### List files in the workspace

```bash
railway ssh --service bumblebee-worker -- ls -la /app/data/
```

### Read a specific file

```bash
railway ssh --service bumblebee-worker -- cat /app/data/canary_profile.md
```

### Search for files by name

```bash
railway ssh --service bumblebee-worker -- find /app/data -name "*.md"
```

### Check disk usage

```bash
railway ssh --service bumblebee-worker -- du -sh /app/data/
```

### List the bumblebee entity data directory

```bash
railway ssh --service bumblebee-worker -- ls -la ~/.bumblebee/entities/canary/
```

This contains: `memory.db` (if SQLite), `knowledge.md`, `journal.md`, `soma/` (bar state), `skills/`, `projects.json`, `self_model.json`.

### Read the agent's journal

```bash
railway ssh --service bumblebee-worker -- cat ~/.bumblebee/entities/canary/journal.md
```

### Read the agent's knowledge

```bash
railway ssh --service bumblebee-worker -- cat ~/.bumblebee/entities/canary/knowledge.md
```

### Check soma bar state

```bash
railway ssh --service bumblebee-worker -- cat ~/.bumblebee/entities/canary/soma/soma-bar-state.json
```

## Important Paths

| Path | What | Persistent? |
|------|------|-------------|
| `/app/data/` | Agent workspace (files it creates/reads) | Yes (volume) |
| `/app/` | Container root (code, README, configs) | No (ephemeral) |
| `~/.bumblebee/entities/<name>/` | Entity data (memory, journal, knowledge, soma) | No unless on volume |
| `/var/lib/postgresql/data/` | Postgres data | Yes (volume) |

## Volume Setup

The workspace volume should be mounted at `/app/data`. Verify with:

```bash
railway volume list
```

If missing, add one:

```bash
railway service bumblebee-worker
railway volume add --mount-path /app/data
railway variable set BUMBLEBEE_EXECUTION_WORKSPACE_DIR=/app/data
railway redeploy --service bumblebee-worker --yes
```

## Troubleshooting

**"No such file or directory"**: The file may have been lost on a redeploy if it was written to `/app/` (ephemeral) rather than `/app/data/` (volume). Check `BUMBLEBEE_EXECUTION_WORKSPACE_DIR` is set.

**SSH hangs**: The container may be restarting. Wait for the deploy to finish, then retry.

**Permission denied**: Files created by the agent run as root in the container. SSH also runs as root, so this shouldn't happen -- but if it does, check the container logs.
