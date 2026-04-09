from __future__ import annotations

from types import SimpleNamespace

from bumblebee.config import AutonomySettings
from bumblebee.presence.wake_cycle import WakeCycleEngine


def _engine(*, allow_tools: bool) -> WakeCycleEngine:
    auto = AutonomySettings(allow_tool_calls_on_wake=allow_tools)
    cfg = SimpleNamespace(harness=SimpleNamespace(autonomy=auto))
    return WakeCycleEngine(cfg)  # type: ignore[arg-type]


def _entity() -> SimpleNamespace:
    return SimpleNamespace(_platforms={"telegram": object(), "discord": object()})


def test_build_context_mentions_tool_venture_when_enabled() -> None:
    eng = _engine(allow_tools=True)
    text = eng._build_context(_entity(), tonic=None, stirring="stir", reason="timer")
    assert "venture out with tools" in text
    assert "Use any relevant tools" in text


def test_build_context_constrains_tools_when_disabled() -> None:
    eng = _engine(allow_tools=False)
    text = eng._build_context(_entity(), tonic=None, stirring="stir", reason="timer")
    assert "constrained right now" in text
    assert "do not venture into tool use" in text


def test_resolve_wake_delivery_prefers_last_conversation_route() -> None:
    eng = _engine(allow_tools=True)
    tg = object()
    entity = SimpleNamespace(
        _platforms={"telegram": tg, "discord": object()},
        _last_conversation={"platform": "telegram", "channel": "12345"},
    )
    ch, pf = eng._resolve_wake_delivery(entity)
    assert ch == "12345"
    assert pf is tg


def test_resolve_wake_delivery_falls_back_to_platform_last_channel() -> None:
    eng = _engine(allow_tools=True)
    dc = SimpleNamespace(last_channel_id="777")
    entity = SimpleNamespace(
        _platforms={"cli": object(), "discord": dc},
        _last_conversation={"platform": "telegram", "channel": "12345"},
    )
    ch, pf = eng._resolve_wake_delivery(entity)
    assert ch == "777"
    assert pf is dc

