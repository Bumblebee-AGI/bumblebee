"""Load and validate harness + entity YAML into typed config objects."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def project_configs_dir() -> Path:
    """Single source: `<repo>/configs` (same directory that contains `default.yaml`)."""
    return _repo_root() / "configs"


def harness_default_path() -> Path:
    return project_configs_dir() / "default.yaml"


def _expand(path: str, entity_name: str) -> str:
    return str(Path(path.replace("{entity_name}", entity_name)).expanduser())


@dataclass
class DeploymentSettings:
    """Where the body runs: local workstation vs split Railway + home inference."""

    mode: str = "local"  # local | hybrid_railway


@dataclass
class InferenceSettings:
    """Brain endpoint selection (tunnel should terminate only at the inference gateway)."""

    provider: str = ""  # local | remote_gateway — empty derives from deployment.mode
    base_url: str = ""  # remote gateway or OpenAI-compatible root; empty uses ollama.base_url
    api_key_env: str = "BUMBLEBEE_INFERENCE_GATEWAY_TOKEN"
    model: str = ""  # optional documentation default / operator hint
    timeout: float = 120.0
    # When True, chat requests include Ollama-style ``options.num_ctx`` from entity cognition.
    # Set False if your /v1/chat/completions server rejects unknown top-level fields.
    pass_num_ctx: bool = True


@dataclass
class AttachmentStorageSettings:
    """Blob storage for durable attachments (hybrid must not rely on container disk)."""

    backend: str = "local_disk"  # local_disk | object_s3_compat
    local_dir: str = "~/.bumblebee/entities/{entity_name}/attachments"
    # S3-compatible (optional)
    endpoint_url_env: str = "BUMBLEBEE_S3_ENDPOINT_URL"
    bucket_env: str = "BUMBLEBEE_S3_BUCKET"
    access_key_env: str = "BUMBLEBEE_S3_ACCESS_KEY"
    secret_key_env: str = "BUMBLEBEE_S3_SECRET_KEY"
    prefix: str = "bumblebee"


@dataclass
class OllamaSettings:
    base_url: str = "http://localhost:11434"
    timeout: float = 120.0
    retry_attempts: int = 3
    retry_delay: float = 2.0


@dataclass
class ModelSettings:
    reflex: str = "gemma4:26b"
    deliberate: str = "gemma4:26b"
    embedding: str = "nomic-embed-text"


@dataclass
class CognitionSettings:
    thinking_mode: bool = True
    temperature: float = 0.75
    reflex_max_tokens: int = 512
    deliberate_max_tokens: int = 2048
    thinking_budget: int = 4096
    max_context_tokens: int = 32768
    escalation_threshold: float = 0.4
    image_token_budget: int = 280
    # After tool results, nudge the model up to this many times when it still sounds mid-task
    # but returned text instead of tool calls (0 disables).
    tool_continuation_rounds: int = 4


@dataclass
class IdentityHarnessSettings:
    emotion_decay_rate: float = 0.001
    drive_tick_interval: int = 120
    evolution_interval: int = 100
    narrative_interval: int = 500


@dataclass
class MemoryHarnessSettings:
    database_path: str = "~/.bumblebee/entities/{entity_name}/memory.db"
    """When set (e.g. postgresql://...), use Postgres instead of SQLite file path."""
    database_url: str = ""
    episode_significance_threshold: float = 0.3
    consolidation_interval: int = 7200
    memory_decay_rate: float = 0.0001
    max_recall_results: int = 10
    embedding_dimensions: int = 768
    narrative_every_n_consolidations: int = 3
    imprint_decay_half_life_seconds: float = 86400.0 * 30
    imprint_recall_weight: float = 0.35


@dataclass
class PresenceHarnessSettings:
    heartbeat_interval: int = 120
    initiative_cooldown: int = 1800
    typing_speed_base: float = 30.0
    typing_speed_variance: float = 0.3
    message_chunk_max: int = 400
    chunk_delay: float = 1.5


@dataclass
class LoggingSettings:
    level: str = "INFO"
    file: str = "~/.bumblebee/entities/{entity_name}/bumblebee.log"
    format: str = "json"


@dataclass
class FirecrawlSettings:
    """Optional Firecrawl API (Bearer key in env). When set, preferred for fetch_url / search_web."""

    api_key_env: str = "FIRECRAWL_API_KEY"
    base_url: str = "https://api.firecrawl.dev/v1"
    prefer_for_fetch: bool = True
    prefer_for_search: bool = True


def default_soma_config() -> dict[str, Any]:
    return {
        "enabled": True,
        "bars": {
            "variables": [
                {"name": "social", "initial": 50, "decay_rate": -2.0, "floor": 0, "ceiling": 100},
                {"name": "curiosity", "initial": 60, "decay_rate": -1.0, "floor": 0, "ceiling": 100},
                {"name": "creative", "initial": 40, "decay_rate": -1.5, "floor": 0, "ceiling": 100},
                {"name": "tension", "initial": 20, "decay_rate": -3.0, "floor": 0, "ceiling": 100},
                {"name": "comfort", "initial": 65, "decay_rate": -0.5, "floor": 0, "ceiling": 100},
            ],
            "momentum_window": 6,
        },
        "coupling": [
            {"when": "social > 80", "effect": "curiosity.decay_rate *= 1.5"},
            {"when": "tension > 70", "effect": "comfort.decay_rate *= 2.0"},
        ],
        "event_effects": {
            "message_received": {"social": 8, "curiosity": 3},
            "message_sent": {"social": 5, "creative": 2},
            "action": {"curiosity": 4},
            "idle": {"social": -1, "curiosity": 1},
            "idle_cycle": {"comfort": 5, "tension": -3},
            "mood_declared": {"comfort": 2},
        },
        "impulses": [
            {"drive": "social", "threshold": 80, "type": "reach_out", "label": "reach_out", "cooldown_minutes": 30, "relief": {"social": -25}},
            {"drive": "curiosity", "threshold": 85, "type": "explore", "label": "explore_something", "cooldown_minutes": 20, "relief": {"curiosity": -20}},
        ],
        "conflicts": [
            {"drives": ["curiosity", "comfort"], "threshold": 70, "label": "restless comfort", "tension_per_tick": 0.5, "comfort_per_tick": -0.3},
        ],
        "affect_cycle_seconds": 180,
        "noise": {
            "enabled": True,
            "model": "",
            "cycle_seconds": 60,
            "temperature": 1.1,
            "max_tokens": 150,
            "max_fragments": 8,
        },
        "wake_voice": {
            "enabled": True,
            "model": "",
            "temperature": 0.8,
            "max_tokens": 300,
        },
    }


@dataclass
class SummonSettings:
    enabled: bool = True
    timeout_seconds: int = 30


@dataclass
class AutonomySettings:
    """Autonomous wake cycle configuration."""

    enabled: bool = False
    min_cycle_gap_seconds: int = 600
    max_cycles_per_hour: int = 4
    messages_per_cycle: int = 2
    base_wake_interval_min: int = 20
    base_wake_interval_max: int = 45
    silence_threshold_seconds: int = 120
    impulse_wake: bool = True
    drive_wake: bool = True
    conflict_wake: bool = True
    noise_wake: bool = False
    summon: SummonSettings = field(default_factory=SummonSettings)


def default_tools_config() -> dict[str, Any]:
    return {
        "shell": {
            "enabled": True,
            "deny": ["rm -rf /", "sudo rm", "shutdown", "reboot", "mkfs", "dd if="],
            "timeout": 30,
        },
        "browser": {"enabled": False},
        "code": {"enabled": True, "timeout": 30},
        "imagegen": {
            "enabled": False,
            "backend": "fal",
            "fal_api_key_env": "FAL_API_KEY",
            "local_url": "",
        },
        "voice": {"enabled": True, "voice_id": "en-US-GuyNeural"},
        "youtube": {"enabled": True},
        "reddit": {"enabled": True},
        "pdf": {"enabled": True},
        "reminders": {"enabled": True},
        "messaging": {"enabled": True},
        "automations": {"enabled": True},
        "journal": {"enabled": True},
        "wikipedia": {"enabled": True},
        "weather": {"enabled": True},
        "news": {"enabled": True},
        "system": {"enabled": True},
        # Shared dangerous-tool execution settings (RPC or container-local fallback when safe).
        "execution": {
            "base_url": "",
            "token_env": "BUMBLEBEE_EXECUTION_RPC_TOKEN",
            "timeout": 45,
            "allow_local": False,
            "require_railway": False,
            "workspace_dir": "",
            "rpc_path": "/rpc",
        },
    }


@dataclass
class HarnessConfig:
    deployment: DeploymentSettings = field(default_factory=DeploymentSettings)
    inference: InferenceSettings = field(default_factory=InferenceSettings)
    ollama: OllamaSettings = field(default_factory=OllamaSettings)
    models: ModelSettings = field(default_factory=ModelSettings)
    cognition: CognitionSettings = field(default_factory=CognitionSettings)
    identity: IdentityHarnessSettings = field(default_factory=IdentityHarnessSettings)
    memory: MemoryHarnessSettings = field(default_factory=MemoryHarnessSettings)
    presence: PresenceHarnessSettings = field(default_factory=PresenceHarnessSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)
    firecrawl: FirecrawlSettings = field(default_factory=FirecrawlSettings)
    attachments: AttachmentStorageSettings = field(default_factory=AttachmentStorageSettings)
    tools: dict[str, Any] = field(default_factory=default_tools_config)
    soma: dict[str, Any] = field(default_factory=default_soma_config)
    autonomy: AutonomySettings = field(default_factory=AutonomySettings)


@dataclass
class EntityPersonality:
    core_traits: dict[str, float] = field(default_factory=dict)
    behavioral_patterns: dict[str, str] = field(default_factory=dict)
    voice: dict[str, Any] = field(default_factory=dict)
    backstory: str = ""


@dataclass
class EntityDrives:
    curiosity_topics: list[str] = field(default_factory=list)
    attachment_threshold: int = 5
    restlessness_decay: int = 3600
    initiative_cooldown: int = 1800


@dataclass
class HistoryCompressionSettings:
    """When rolling chat history exceeds ``rolling_history_max_messages``, oldest turns are dropped.
    If ``enabled``, those turns are merged into an in-memory summary prepended to model context."""

    enabled: bool = True
    summary_max_chars: int = 4500
    max_merge_input_chars: int = 12000
    merge_max_tokens: int = 900
    format_per_message_chars: int = 2200


@dataclass
class EntityCognition:
    reflex_model: str = ""
    deliberate_model: str = ""
    always_deliberate: bool = False
    fast_deliberate_mode: bool = False
    deliberate_max_tokens: int = 0  # 0 => use harness.cognition.deliberate_max_tokens
    system_prompt_char_limit: int = 12000
    rolling_history_max_messages: int = 40
    knowledge_recent_turns: int = 10
    history_message_char_limit: int = 4000
    thinking_mode: bool = True
    temperature: float = 0.75
    max_context_tokens: int = 32768
    thinking_budget: int = 4096
    history_compression: HistoryCompressionSettings = field(default_factory=HistoryCompressionSettings)
    # 0 = use harness.cognition.tool_continuation_rounds
    tool_continuation_rounds: int = 0

    def ollama_num_ctx(self) -> int | None:
        n = int(self.max_context_tokens or 0)
        return n if n > 0 else None


@dataclass
class AutomationsEmergenceSettings:
    enabled: bool = True
    analysis_interval: int = 7200
    max_suggestions: int = 3


@dataclass
class AutomationsJournalSettings:
    enabled: bool = True
    max_entries: int = 1000


@dataclass
class EntityAutomationsSettings:
    enabled: bool = True
    max_automations: int = 50
    max_failures: int = 5
    emergence: AutomationsEmergenceSettings = field(default_factory=AutomationsEmergenceSettings)
    journal: AutomationsJournalSettings = field(default_factory=AutomationsJournalSettings)


@dataclass
class EntityPresence:
    platforms: list[dict[str, Any]] = field(default_factory=list)
    daemon: dict[str, int] = field(default_factory=dict)
    tool_activity: bool = True


@dataclass
class EntityConfig:
    name: str
    harness: HarnessConfig
    personality: EntityPersonality
    drives: EntityDrives
    cognition: EntityCognition
    presence: EntityPresence
    automations: EntityAutomationsSettings = field(default_factory=EntityAutomationsSettings)
    raw: dict[str, Any] = field(default_factory=dict)

    def db_path(self) -> str:
        return _expand(self.harness.memory.database_path, self.name)

    def database_url(self) -> str:
        """Effective DB URL: DATABASE_URL env, then harness memory.database_url, else empty (SQLite file)."""
        env_url = (os.environ.get("DATABASE_URL") or "").strip()
        if env_url:
            return env_url
        return (self.harness.memory.database_url or "").strip()

    def knowledge_path(self) -> str:
        """Curated knowledge markdown: persistent volume when available, else beside DB file."""
        ws = (os.environ.get("BUMBLEBEE_EXECUTION_WORKSPACE_DIR") or "").strip()
        if ws:
            return str(Path(ws) / "knowledge.md")
        if self.database_url():
            return _expand("~/.bumblebee/entities/{entity_name}/knowledge.md", self.name)
        return str(Path(self.db_path()).expanduser().parent / "knowledge.md")

    def log_path(self) -> str:
        return _expand(self.harness.logging.file, self.name)

    def effective_ollama_num_ctx(self) -> int | None:
        """Ollama OpenAI-compat ``options.num_ctx`` when enabled and entity sets a positive window."""
        if not self.harness.inference.pass_num_ctx:
            return None
        return self.cognition.ollama_num_ctx()

    def journal_path(self) -> str:
        ws = (os.environ.get("BUMBLEBEE_EXECUTION_WORKSPACE_DIR") or "").strip()
        if ws:
            return str(Path(ws) / "journal.md")
        return _expand("~/.bumblebee/entities/{entity_name}/journal.md", self.name)

    def skills_dir(self) -> str:
        return _expand("~/.bumblebee/entities/{entity_name}/skills", self.name)

    def projects_path(self) -> str:
        return _expand("~/.bumblebee/entities/{entity_name}/projects.json", self.name)

    def self_model_path(self) -> str:
        return _expand("~/.bumblebee/entities/{entity_name}/self_model.json", self.name)

    def soma_dir(self) -> str:
        ws = (os.environ.get("BUMBLEBEE_EXECUTION_WORKSPACE_DIR") or "").strip()
        if ws:
            return str(Path(ws) / "soma")
        return _expand("~/.bumblebee/entities/{entity_name}/soma", self.name)


def _merge_dict(base: dict, override: dict) -> dict:
    out = {**base}
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge_dict(out[k], v)
        else:
            out[k] = v
    return out


def _dict_to_harness(d: dict[str, Any]) -> HarnessConfig:
    fc_raw = {**FirecrawlSettings().__dict__, **(d.get("firecrawl") or {})}
    for bkey in ("prefer_for_fetch", "prefer_for_search"):
        if bkey in fc_raw:
            fc_raw[bkey] = bool(fc_raw[bkey])
    att_raw = {**AttachmentStorageSettings().__dict__, **(d.get("attachments") or {})}
    mem_raw = {**MemoryHarnessSettings().__dict__, **(d.get("memory") or {})}
    tools_raw = _merge_dict(default_tools_config(), d.get("tools") or {})
    soma_raw = _merge_dict(default_soma_config(), d.get("soma") or {})
    auto_raw = d.get("autonomy") or {}
    summon_raw = auto_raw.pop("summon", None) or {}
    autonomy = AutonomySettings(
        **{**AutonomySettings().__dict__, "summon": SummonSettings(**{**SummonSettings().__dict__, **summon_raw}), **{k: v for k, v in auto_raw.items() if k != "summon"}},
    )
    return HarnessConfig(
        deployment=DeploymentSettings(
            **{**DeploymentSettings().__dict__, **(d.get("deployment") or {})}
        ),
        inference=InferenceSettings(
            **{**InferenceSettings().__dict__, **(d.get("inference") or {})}
        ),
        ollama=OllamaSettings(**{**OllamaSettings().__dict__, **(d.get("ollama") or {})}),
        models=ModelSettings(**{**ModelSettings().__dict__, **(d.get("models") or {})}),
        cognition=CognitionSettings(**{**CognitionSettings().__dict__, **(d.get("cognition") or {})}),
        identity=IdentityHarnessSettings(
            **{**IdentityHarnessSettings().__dict__, **(d.get("identity") or {})}
        ),
        memory=MemoryHarnessSettings(**mem_raw),
        presence=PresenceHarnessSettings(
            **{**PresenceHarnessSettings().__dict__, **(d.get("presence") or {})}
        ),
        logging=LoggingSettings(**{**LoggingSettings().__dict__, **(d.get("logging") or {})}),
        firecrawl=FirecrawlSettings(**fc_raw),
        attachments=AttachmentStorageSettings(**att_raw),
        tools=tools_raw if isinstance(tools_raw, dict) else default_tools_config(),
        soma=soma_raw if isinstance(soma_raw, dict) else default_soma_config(),
        autonomy=autonomy,
    )


def apply_harness_env_overrides(h: HarnessConfig) -> None:
    """Railway-friendly env overrides (see .env.example)."""
    dm = (os.environ.get("BUMBLEBEE_DEPLOYMENT_MODE") or "").strip().lower()
    if dm in ("local", "hybrid_railway"):
        h.deployment.mode = dm
    ip = (os.environ.get("BUMBLEBEE_INFERENCE_PROVIDER") or "").strip().lower()
    if ip in ("local", "remote_gateway"):
        h.inference.provider = ip
    ib = (os.environ.get("BUMBLEBEE_INFERENCE_BASE_URL") or "").strip()
    if ib:
        h.inference.base_url = ib.rstrip("/")
    oh = (os.environ.get("OLLAMA_HOST") or "").strip()
    if oh and not ib:
        h.ollama.base_url = oh.rstrip("/")
    ike = (os.environ.get("BUMBLEBEE_INFERENCE_API_KEY_ENV") or "").strip()
    if ike:
        h.inference.api_key_env = ike
    ito = (os.environ.get("BUMBLEBEE_INFERENCE_TIMEOUT") or "").strip()
    if ito:
        try:
            h.inference.timeout = float(ito)
        except ValueError:
            pass
    att = (os.environ.get("BUMBLEBEE_ATTACHMENTS_BACKEND") or "").strip().lower()
    if att in ("local_disk", "object_s3_compat"):
        h.attachments.backend = att
    ewd = (os.environ.get("BUMBLEBEE_EXECUTION_WORKSPACE_DIR") or "").strip()
    if ewd:
        exec_cfg = h.tools.get("execution")
        if isinstance(exec_cfg, dict):
            exec_cfg["workspace_dir"] = ewd


def load_harness_config(path: Path | None = None) -> HarnessConfig:
    p = path if path is not None else harness_default_path()
    if not p.exists():
        h = HarnessConfig()
        apply_harness_env_overrides(h)
        return h
    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    h = _dict_to_harness(data)
    apply_harness_env_overrides(h)
    return h


def load_entity_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_firecrawl_settings(
    harness: HarnessConfig,
    entity_raw: dict[str, Any],
) -> FirecrawlSettings:
    """Merge harness defaults with optional top-level ``firecrawl:`` on the entity YAML."""
    fc = harness.firecrawl
    o = entity_raw.get("firecrawl") or {}
    if not o:
        return fc
    merged = {
        "api_key_env": str(o.get("api_key_env", fc.api_key_env)),
        "base_url": str(o.get("base_url", fc.base_url)).rstrip("/"),
        "prefer_for_fetch": bool(o.get("prefer_for_fetch", fc.prefer_for_fetch)),
        "prefer_for_search": bool(o.get("prefer_for_search", fc.prefer_for_search)),
    }
    return FirecrawlSettings(**merged)


def validate_traits(traits: dict[str, float]) -> None:
    for k, v in traits.items():
        if not isinstance(v, (int, float)) or not (0.0 <= float(v) <= 1.0):
            raise ValueError(f"Trait {k} must be in [0,1], got {v}")


def entity_from_dict(harness: HarnessConfig, data: dict[str, Any]) -> EntityConfig:
    name = data.get("name")
    if not name or not isinstance(name, str):
        raise ValueError("Entity requires string 'name'")
    pers = data.get("personality") or {}
    traits = (pers.get("core_traits") or {}) if isinstance(pers, dict) else {}
    validate_traits(traits)
    personality = EntityPersonality(
        core_traits=traits,
        behavioral_patterns=dict(pers.get("behavioral_patterns") or {}),
        voice=dict(pers.get("voice") or {}),
        backstory=str(pers.get("backstory") or ""),
    )
    dr = data.get("drives") or {}
    drives = EntityDrives(
        curiosity_topics=list(dr.get("curiosity_topics") or []),
        attachment_threshold=int(dr.get("attachment_threshold", 5)),
        restlessness_decay=int(dr.get("restlessness_decay", 3600)),
        initiative_cooldown=int(dr.get("initiative_cooldown", 1800)),
    )
    cog = data.get("cognition") or {}
    hc_raw = cog.get("history_compression")
    if isinstance(hc_raw, dict):
        history_compression = HistoryCompressionSettings(
            enabled=bool(hc_raw.get("enabled", True)),
            summary_max_chars=max(500, int(hc_raw.get("summary_max_chars", 4500) or 4500)),
            max_merge_input_chars=max(
                2000, int(hc_raw.get("max_merge_input_chars", 12000) or 12000)
            ),
            merge_max_tokens=max(200, int(hc_raw.get("merge_max_tokens", 900) or 900)),
            format_per_message_chars=max(
                400, int(hc_raw.get("format_per_message_chars", 2200) or 2200)
            ),
        )
    else:
        history_compression = HistoryCompressionSettings()
    ec = EntityCognition(
        reflex_model=str(cog.get("reflex_model") or harness.models.reflex),
        deliberate_model=str(cog.get("deliberate_model") or harness.models.deliberate),
        always_deliberate=bool(cog.get("always_deliberate", False)),
        fast_deliberate_mode=bool(cog.get("fast_deliberate_mode", False)),
        deliberate_max_tokens=max(0, int(cog.get("deliberate_max_tokens", 0) or 0)),
        system_prompt_char_limit=max(
            2000, int(cog.get("system_prompt_char_limit", 12000) or 12000)
        ),
        rolling_history_max_messages=max(
            8, int(cog.get("rolling_history_max_messages", 40) or 40)
        ),
        knowledge_recent_turns=max(2, int(cog.get("knowledge_recent_turns", 10) or 10)),
        history_message_char_limit=max(
            600, int(cog.get("history_message_char_limit", 4000) or 4000)
        ),
        thinking_mode=bool(cog.get("thinking_mode", harness.cognition.thinking_mode)),
        temperature=float(cog.get("temperature", harness.cognition.temperature)),
        max_context_tokens=int(
            cog.get("max_context_tokens", harness.cognition.max_context_tokens)
        ),
        thinking_budget=int(cog.get("thinking_budget", harness.cognition.thinking_budget)),
        history_compression=history_compression,
        tool_continuation_rounds=max(
            0, int(cog.get("tool_continuation_rounds", 0) or 0)
        ),
    )
    pr = data.get("presence") or {}
    presence = EntityPresence(
        platforms=list(pr.get("platforms") or [{"type": "cli"}]),
        daemon=dict(pr.get("daemon") or {}),
        tool_activity=bool(pr.get("tool_activity", True)),
    )
    au = data.get("automations") or {}
    em = au.get("emergence") or {}
    jr = au.get("journal") or {}
    automations = EntityAutomationsSettings(
        enabled=bool(au.get("enabled", True)),
        max_automations=max(1, int(au.get("max_automations", 50) or 50)),
        max_failures=max(1, int(au.get("max_failures", 5) or 5)),
        emergence=AutomationsEmergenceSettings(
            enabled=bool(em.get("enabled", True)),
            analysis_interval=max(60, int(em.get("analysis_interval", 7200) or 7200)),
            max_suggestions=max(1, min(10, int(em.get("max_suggestions", 3) or 3))),
        ),
        journal=AutomationsJournalSettings(
            enabled=bool(jr.get("enabled", True)),
            max_entries=max(10, int(jr.get("max_entries", 1000) or 1000)),
        ),
    )
    return EntityConfig(
        name=name,
        harness=harness,
        personality=personality,
        drives=drives,
        cognition=ec,
        presence=presence,
        automations=automations,
        raw=data,
    )


def load_entity_config(entity_name: str, harness: HarnessConfig | None = None) -> EntityConfig:
    h = harness or load_harness_config()
    ent_path = project_configs_dir() / "entities" / f"{entity_name}.yaml"
    if not ent_path.exists():
        raise FileNotFoundError(f"Entity file not found: {ent_path}")
    merged = load_entity_yaml(ent_path)
    return entity_from_dict(h, merged)


def validate_entity_env(entity: EntityConfig) -> list[str]:
    """Return list of warnings if platform env vars missing."""
    warnings: list[str] = []
    mode = (entity.harness.deployment.mode or "local").strip().lower()
    prov = (entity.harness.inference.provider or "").strip().lower()
    if mode == "hybrid_railway" or prov == "remote_gateway":
        envk = entity.harness.inference.api_key_env or "BUMBLEBEE_INFERENCE_GATEWAY_TOKEN"
        if not (os.environ.get(envk) or "").strip():
            warnings.append(
                f"Hybrid/remote inference: set {envk} (bearer token for the home inference gateway)"
            )
        if not (entity.harness.inference.base_url or "").strip() and mode == "hybrid_railway":
            warnings.append(
                "Hybrid Railway: set inference.base_url or BUMBLEBEE_INFERENCE_BASE_URL to the tunneled gateway URL"
            )
    if mode == "hybrid_railway" and not entity.database_url():
        warnings.append(
            "Hybrid Railway: set DATABASE_URL (Postgres) — container-local SQLite is not durable"
        )
    if mode == "hybrid_railway" and entity.harness.attachments.backend == "local_disk":
        warnings.append(
            "Hybrid Railway: prefer attachments.backend: object_s3_compat with S3 env vars — local_disk is not durable on Railway disks"
        )
    for pl in entity.presence.platforms:
        t = (pl.get("type") or "").lower()
        if t == "discord":
            envk = pl.get("token_env", "DISCORD_TOKEN")
            if not os.environ.get(envk):
                warnings.append(f"Discord platform: env {envk} not set")
        if t == "telegram":
            envk = pl.get("token_env", "TELEGRAM_TOKEN")
            if not os.environ.get(envk):
                warnings.append(f"Telegram platform: env {envk} not set")
    fc = resolve_firecrawl_settings(entity.harness, entity.raw)
    if (fc.prefer_for_fetch or fc.prefer_for_search) and not (
        os.environ.get(fc.api_key_env) or ""
    ).strip():
        warnings.append(
            f"Firecrawl: env {fc.api_key_env} not set — fetch_url/search_web fall back to ddgs/aiohttp"
        )
    return warnings


async def validate_ollama_models(entity: EntityConfig, provider: Any) -> tuple[bool, list[str]]:
    """Require chat models on the inference backend. Embedding is optional."""
    fn = getattr(provider, "ensure_models", None)
    if not callable(fn):
        return True, []
    ok, missing = await fn(
        entity.cognition.reflex_model,
        entity.cognition.deliberate_model,
        entity.harness.models.deliberate,
    )
    return ok, missing
