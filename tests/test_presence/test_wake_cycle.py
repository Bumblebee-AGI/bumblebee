from __future__ import annotations

import json
from types import SimpleNamespace

from bumblebee.config import AutonomySettings
from bumblebee.presence.wake_cycle import WakeCycleEngine, _meta_leak_detected


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


def test_build_context_includes_internal_disposition_when_poker_set() -> None:
    eng = _engine(allow_tools=True)
    text = eng._build_context(
        _entity(),
        tonic=None,
        stirring="stir",
        reason="timer",
        poker_disposition="notice the loose end",
    )
    assert "[Internal disposition" in text
    assert "loose permission" in text
    assert "generative noise (GEN)" in text
    assert "notice the loose end" in text
    assert "[Your subconscious stirring]" in text


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


def test_meta_leak_detector_flags_control_flow_speech() -> None:
    assert _meta_leak_detected("The user has seen my summary. I'll just end the turn.")
    assert not _meta_leak_detected("i found two interesting posts and saved notes")


def test_load_recent_threads_reads_episode_tail(tmp_path) -> None:
    eng = _engine(allow_tools=True)
    cfg = SimpleNamespace(journal_path=lambda: str(tmp_path / "journal.md"))
    entity = SimpleNamespace(config=cfg)
    p = tmp_path / "wake_episodes.jsonl"
    p.write_text(
        "\n".join(
            [
                json.dumps({"carryover_threads": ["alpha", "beta"]}),
                json.dumps({"carryover_threads": ["beta", "gamma"]}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    threads = eng._load_recent_threads(entity)
    assert threads[:3] == ["beta", "gamma", "alpha"]


def test_build_user_end_card_contains_continuity() -> None:
    eng = _engine(allow_tools=True)
    card = eng._build_user_end_card(
        reason="timer",
        wake_intent="explore_question",
        rounds_completed=3,
        tools_all=["search_web", "fetch_url"],
        carryover=["explore_question from timer"],
    )
    joined = "\n".join(card)
    assert "lean: explore_question" in joined
    assert "next pull" in joined


def test_extract_lingering_sparks_pulls_old_opportunities() -> None:
    eng = _engine(allow_tools=True)
    entity = SimpleNamespace(
        _history=[
            {"role": "user", "content": "hey"},
            {"role": "user", "content": "you should learn how to parse rss feeds later"},
            {"role": "assistant", "content": "we could build a tiny game sometime"},
        ]
    )
    sparks = eng._extract_lingering_sparks(entity)
    assert any("learn how to parse rss feeds" in s for s in sparks)
    assert any("build a tiny game" in s for s in sparks)


def test_session_memory_board_is_bounded_and_structured() -> None:
    eng = _engine(allow_tools=True)
    mem = eng._new_session_memory(
        reason="timer",
        wake_intent="build_something",
        wake_want="I want to build a relay prototype.",
        lingering_sparks=["build that relay", "learn rss parsing"],
        project_lines=["relay [active] — ship parser | next: tests"],
        skill_lines=["rss parsing: parse feed + dedupe entries"],
        continuity=["build_something from timer"],
    )
    eng._update_session_memory(
        mem,
        tool_names=["search_web", "fetch_url", "update_skill"],
        reply_text="found a clean parser pattern. next i should implement retries.",
        round_idx=1,
    )
    board = eng._render_session_memory_block(mem, max_chars=520)
    assert "objective:" in board
    assert "want:" in board
    assert "threads:" in board
    assert "recent_actions:" in board
    assert len(board) <= 520


def test_compose_wake_want_mentions_intent_and_sources() -> None:
    eng = _engine(allow_tools=True)
    want = eng._compose_wake_want(
        wake_intent="learn_something",
        reason="desire:learn:rss",
        lingering_sparks=["learn rss parsing from last chat"],
        project_lines=["feed relay [active] — unfinished"],
        skill_lines=["http retries: backoff + jitter"],
    )
    assert "I want to learn" in want
    assert "spark:" in want

