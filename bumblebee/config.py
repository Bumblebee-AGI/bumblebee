"""Load and validate harness + entity YAML into typed config objects."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from bumblebee.utils.ollama_client import OllamaClient


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
class OllamaSettings:
    base_url: str = "http://localhost:11434"
    timeout: float = 120.0
    retry_attempts: int = 3
    retry_delay: float = 2.0


@dataclass
class ModelSettings:
    reflex: str = "gemma4:e4b"
    deliberate: str = "gemma4:e4b"
    embedding: str = "nomic-embed-text"


@dataclass
class CognitionSettings:
    thinking_mode: bool = True
    temperature: float = 0.8
    reflex_max_tokens: int = 256
    deliberate_max_tokens: int = 2048
    thinking_budget: int = 4096
    max_context_tokens: int = 32768
    escalation_threshold: float = 0.4
    image_token_budget: int = 280


@dataclass
class IdentityHarnessSettings:
    emotion_decay_rate: float = 0.001
    drive_tick_interval: int = 60
    evolution_interval: int = 100
    narrative_interval: int = 500


@dataclass
class MemoryHarnessSettings:
    database_path: str = "~/.bumblebee/entities/{entity_name}/memory.db"
    episode_significance_threshold: float = 0.3
    consolidation_interval: int = 3600
    memory_decay_rate: float = 0.0001
    max_recall_results: int = 10
    embedding_dimensions: int = 768
    narrative_every_n_consolidations: int = 3
    imprint_decay_half_life_seconds: float = 86400.0 * 30
    imprint_recall_weight: float = 0.35


@dataclass
class PresenceHarnessSettings:
    heartbeat_interval: int = 60
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
class HarnessConfig:
    ollama: OllamaSettings = field(default_factory=OllamaSettings)
    models: ModelSettings = field(default_factory=ModelSettings)
    cognition: CognitionSettings = field(default_factory=CognitionSettings)
    identity: IdentityHarnessSettings = field(default_factory=IdentityHarnessSettings)
    memory: MemoryHarnessSettings = field(default_factory=MemoryHarnessSettings)
    presence: PresenceHarnessSettings = field(default_factory=PresenceHarnessSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)


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
class EntityCognition:
    reflex_model: str = ""
    deliberate_model: str = ""
    thinking_mode: bool = True
    temperature: float = 0.8
    max_context_tokens: int = 32768
    thinking_budget: int = 4096


@dataclass
class EntityPresence:
    platforms: list[dict[str, Any]] = field(default_factory=list)
    daemon: dict[str, int] = field(default_factory=dict)


@dataclass
class EntityConfig:
    name: str
    harness: HarnessConfig
    personality: EntityPersonality
    drives: EntityDrives
    cognition: EntityCognition
    presence: EntityPresence
    raw: dict[str, Any] = field(default_factory=dict)

    def db_path(self) -> str:
        return _expand(self.harness.memory.database_path, self.name)

    def log_path(self) -> str:
        return _expand(self.harness.logging.file, self.name)


def _merge_dict(base: dict, override: dict) -> dict:
    out = {**base}
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge_dict(out[k], v)
        else:
            out[k] = v
    return out


def _dict_to_harness(d: dict[str, Any]) -> HarnessConfig:
    return HarnessConfig(
        ollama=OllamaSettings(**{**OllamaSettings().__dict__, **d.get("ollama", {})}),
        models=ModelSettings(**{**ModelSettings().__dict__, **d.get("models", {})}),
        cognition=CognitionSettings(**{**CognitionSettings().__dict__, **d.get("cognition", {})}),
        identity=IdentityHarnessSettings(
            **{**IdentityHarnessSettings().__dict__, **d.get("identity", {})}
        ),
        memory=MemoryHarnessSettings(**{**MemoryHarnessSettings().__dict__, **d.get("memory", {})}),
        presence=PresenceHarnessSettings(
            **{**PresenceHarnessSettings().__dict__, **d.get("presence", {})}
        ),
        logging=LoggingSettings(**{**LoggingSettings().__dict__, **d.get("logging", {})}),
    )


def load_harness_config(path: Path | None = None) -> HarnessConfig:
    p = path if path is not None else harness_default_path()
    if not p.exists():
        return HarnessConfig()
    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return _dict_to_harness(data)


def load_entity_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


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
    ec = EntityCognition(
        reflex_model=str(cog.get("reflex_model") or harness.models.reflex),
        deliberate_model=str(cog.get("deliberate_model") or harness.models.deliberate),
        thinking_mode=bool(cog.get("thinking_mode", harness.cognition.thinking_mode)),
        temperature=float(cog.get("temperature", harness.cognition.temperature)),
        max_context_tokens=int(
            cog.get("max_context_tokens", harness.cognition.max_context_tokens)
        ),
        thinking_budget=int(cog.get("thinking_budget", harness.cognition.thinking_budget)),
    )
    pr = data.get("presence") or {}
    presence = EntityPresence(
        platforms=list(pr.get("platforms") or [{"type": "cli"}]),
        daemon=dict(pr.get("daemon") or {}),
    )
    return EntityConfig(
        name=name,
        harness=harness,
        personality=personality,
        drives=drives,
        cognition=ec,
        presence=presence,
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
    return warnings


async def validate_ollama_models(entity: EntityConfig, client: OllamaClient) -> tuple[bool, list[str]]:
    """Require entity chat models plus harness deliberate (personality/narrative/evolution) and embedding."""
    ok, missing = await client.ensure_models(
        entity.cognition.reflex_model,
        entity.cognition.deliberate_model,
        entity.harness.models.deliberate,
        entity.harness.models.embedding,
    )
    return ok, missing
