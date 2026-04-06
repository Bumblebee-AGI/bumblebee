"""Filesystem and host tools respect hybrid_railway off-Railway the same as execution_rpc."""

from __future__ import annotations

import json

import pytest

from bumblebee.config import HarnessConfig, entity_from_dict
from bumblebee.presence.tools.filesystem import list_directory, read_file, search_files
from bumblebee.presence.tools.runtime import ToolRuntimeContext, reset_tool_runtime, set_tool_runtime
from bumblebee.presence.tools.system_info import get_system_info

_MIN = {
    "name": "FsHybrid",
    "personality": {
        "core_traits": {k: 0.5 for k in ["curiosity", "warmth", "assertiveness", "humor", "openness", "neuroticism", "conscientiousness"]},
        "behavioral_patterns": {},
        "voice": {},
        "backstory": "",
    },
    "drives": {"curiosity_topics": [], "attachment_threshold": 5, "restlessness_decay": 3600, "initiative_cooldown": 1800},
    "cognition": {},
    "presence": {"platforms": [], "daemon": {}},
}


class _Holder:
    __slots__ = ("config",)

    def __init__(self, config: object) -> None:
        self.config = config


@pytest.mark.asyncio
async def test_filesystem_blocked_hybrid_without_railway_env(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAILWAY_ENVIRONMENT", raising=False)
    h = HarnessConfig()
    h.deployment.mode = "hybrid_railway"
    h.memory.database_path = str(tmp_path / "m.db")
    ec = entity_from_dict(h, _MIN)
    tok = set_tool_runtime(ToolRuntimeContext(entity=_Holder(ec), inp=None))
    try:
        for coro in (
            read_file(str(tmp_path / "nope.txt")),
            list_directory(str(tmp_path)),
            search_files(str(tmp_path), "*"),
            get_system_info(),
        ):
            out = await coro
            data = json.loads(out)
            assert "error" in data
            assert "hybrid_railway" in data["error"] or "Railway" in data["error"]
    finally:
        reset_tool_runtime(tok)


@pytest.mark.asyncio
async def test_list_directory_local_mode_works(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAILWAY_ENVIRONMENT", raising=False)
    h = HarnessConfig()
    h.deployment.mode = "local"
    h.memory.database_path = str(tmp_path / "m.db")
    ec = entity_from_dict(h, _MIN)
    tok = set_tool_runtime(ToolRuntimeContext(entity=_Holder(ec), inp=None))
    try:
        out = await list_directory(str(tmp_path))
        data = json.loads(out)
        assert "entries" in data
    finally:
        reset_tool_runtime(tok)
