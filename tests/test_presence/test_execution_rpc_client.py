"""Execution client: hybrid must not use the home PC as shell/fs host when RPC is unset."""

from __future__ import annotations

import pytest

from bumblebee.config import HarnessConfig, entity_from_dict
from bumblebee.presence.tools import execution_rpc
from bumblebee.presence.tools.runtime import ToolRuntimeContext, reset_tool_runtime, set_tool_runtime

_MIN_ENTITY = {
    "name": "ExecClientTest",
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


class _EntityHolder:
    __slots__ = ("config",)

    def __init__(self, config: object) -> None:
        self.config = config


@pytest.fixture(autouse=True)
def _clear_execution_clients() -> None:
    execution_rpc._CLIENTS.clear()
    yield
    execution_rpc._CLIENTS.clear()


@pytest.mark.parametrize(
    ("deployment_mode", "railway_env", "raw_tools", "expect_local"),
    [
        ("hybrid_railway", "", {}, False),
        ("hybrid_railway", "production", {}, True),
        ("hybrid_railway", "", {"tools": {"execution": {"allow_local": True}}}, True),
        ("local", "", {}, True),
    ],
)
def test_allow_local_backend_hybrid_only_on_railway_or_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    deployment_mode: str,
    railway_env: str,
    raw_tools: dict,
    expect_local: bool,
) -> None:
    monkeypatch.delenv("BUMBLEBEE_EXECUTION_RPC_URL", raising=False)
    if railway_env:
        monkeypatch.setenv("RAILWAY_ENVIRONMENT", railway_env)
    else:
        monkeypatch.delenv("RAILWAY_ENVIRONMENT", raising=False)

    h = HarnessConfig()
    h.deployment.mode = deployment_mode
    h.memory.database_path = str(tmp_path / "m.db")
    data = {**_MIN_ENTITY, **raw_tools}
    ec = entity_from_dict(h, data)
    holder = _EntityHolder(ec)
    tok = set_tool_runtime(ToolRuntimeContext(entity=holder, inp=None))
    try:
        client = execution_rpc.get_execution_client()
        assert client.allow_local_backend is expect_local
    finally:
        reset_tool_runtime(tok)
