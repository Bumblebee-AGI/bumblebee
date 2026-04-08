---
name: railway-filesystem
description: Access and inspect files on the Railway worker container filesystem. Use when the user wants to see what files their bumblebee agent has written, read file contents from the Railway container, or debug workspace issues on the deployed worker.
---

# Railway Filesystem Access

How to inspect files on the bumblebee worker container running on Railway.

## Persistent vs Ephemeral Paths

The worker has two kinds of storage. Knowing which is which prevents surprise data loss.

| Path | Backing | Survives redeploy? | What lives here |
|------|---------|---------------------|-----------------|
| `/app/data/` | **Railway volume** | **Yes** | knowledge.md, journal.md, soma/, workspace files the entity creates |
| `/app/` | Ephemeral container disk | No | Application code, configs, README, Dockerfile artifacts |
| `~/.bumblebee/` | Ephemeral container disk | No | Entity data **only if BUMBLEBEE_EXECUTION_WORKSPACE_DIR is NOT set** (avoid this) |

When `BUMBLEBEE_EXECUTION_WORKSPACE_DIR=/app/data` is set (the recommended hybrid config), knowledge, journal, and soma state all land on the volume. Without it, they fall back to `~/.bumblebee/entities/<name>/` on the ephemeral disk and are lost on every redeploy.

## Prerequisites

- Railway CLI installed and authenticated (`railway login`)
- Project linked (`railway link` or already linked via `railway.toml`)
- Service linked to the worker: `railway service bumblebee-worker`

## Volume Verification

### Check the volume exists

```bash
railway volume list
```

Look for a volume mounted at `/app/data` on the `bumblebee-worker` service.

### Check the env var is set

```bash
railway variable list -s bumblebee-worker | grep WORKSPACE_DIR
```

Should show `BUMBLEBEE_EXECUTION_WORKSPACE_DIR=/app/data`.

### Create the volume if missing

```bash
railway service bumblebee-worker
railway volume add --mount-path /app/data
railway variable set BUMBLEBEE_EXECUTION_WORKSPACE_DIR=/app/data
railway redeploy --service bumblebee-worker --yes
```

### Verify persistence survives a redeploy

```bash
# Write a marker
railway ssh --service bumblebee-worker -- sh -c 'echo "test $(date)" > /app/data/.persist-check'
# Redeploy
railway redeploy --service bumblebee-worker --yes
# After redeploy completes, check the marker
railway ssh --service bumblebee-worker -- cat /app/data/.persist-check
```

## Quick Commands

### List files in the workspace (persistent volume)

```bash
railway ssh --service bumblebee-worker -- ls -la /app/data/
```

### Read a specific file

```bash
railway ssh --service bumblebee-worker -- cat /app/data/knowledge.md
```

### Search for files by name

```bash
railway ssh --service bumblebee-worker -- find /app/data -name "*.md"
```

### Check disk usage

```bash
railway ssh --service bumblebee-worker -- du -sh /app/data/
```

### Read the entity's knowledge

```bash
railway ssh --service bumblebee-worker -- cat /app/data/knowledge.md
```

### Read the entity's journal

```bash
railway ssh --service bumblebee-worker -- cat /app/data/journal.md
```

### Check soma bar state

```bash
railway ssh --service bumblebee-worker -- cat /app/data/soma/soma-bar-state.json
```

### List the ephemeral entity data directory (if WORKSPACE_DIR is unset)

```bash
railway ssh --service bumblebee-worker -- ls -la ~/.bumblebee/entities/canary/
```

## Seeding / Editing knowledge.md via SSH

### Write a new knowledge file from scratch

```bash
railway ssh --service bumblebee-worker -- tee /app/data/knowledge.md << 'EOF'
## [locked] about yourself
Canary is an entity running on the Bumblebee harness.

## notes
(empty — the entity will fill this in as it learns)
EOF
```

### Append a new section

```bash
railway ssh --service bumblebee-worker -- tee -a /app/data/knowledge.md << 'EOF'

## new topic
Content for this section.
EOF
```

### Edit in-place with sed (simple replacements)

```bash
railway ssh --service bumblebee-worker -- sed -i 's/old text/new text/g' /app/data/knowledge.md
```

### Download, edit locally, upload

```bash
railway ssh --service bumblebee-worker -- cat /app/data/knowledge.md > knowledge-local.md
# edit knowledge-local.md with your editor
cat knowledge-local.md | railway ssh --service bumblebee-worker -- tee /app/data/knowledge.md
```

## Backup and Restore

### Download a full backup

```bash
railway ssh --service bumblebee-worker -- tar czf - /app/data/ > bumblebee-data-backup.tar.gz
```

### Restore from backup

```bash
cat bumblebee-data-backup.tar.gz | railway ssh --service bumblebee-worker -- tar xzf - -C /
```

## Troubleshooting

**"No such file or directory"**: The file may have been lost on a redeploy if it was written to `/app/` (ephemeral) rather than `/app/data/` (volume). Check `BUMBLEBEE_EXECUTION_WORKSPACE_DIR` is set.

**Knowledge file exists under ~/.bumblebee but not /app/data**: The env var is not set. Fix: `railway variable set -s bumblebee-worker BUMBLEBEE_EXECUTION_WORKSPACE_DIR=/app/data` then redeploy.

**SSH hangs**: The container may be restarting. Wait for the deploy to finish, then retry.

**Permission denied**: Files created by the agent run as root in the container. SSH also runs as root, so this shouldn't happen — but if it does, check the container logs via `railway logs -s bumblebee-worker`.

**Volume is empty after first deploy**: Expected on a brand-new volume. The entity automatically creates a starter knowledge.md on first startup. Soma state and journal are created as the entity runs.

Full persistence guide: `docs/hybrid-railway-persistence.md`
