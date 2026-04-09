from bumblebee.presence.automations.models import Automation, AutomationOrigin
from bumblebee.presence.platforms.telegram_format import (
    _relative_time,
    build_status_html,
    command_menu_items,
    telegram_registered_slash_command_names,
    format_commands_page,
    format_remote_session_caption,
    format_session_status_html,
    format_routines_html,
    format_start_html,
    format_tools_html,
    split_telegram_chunks,
)


def test_relative_time_short_forms():
    now = __import__("time").time()
    assert _relative_time(None) == "—"
    assert _relative_time(now - 30) == "just now"
    assert _relative_time(now - 120).endswith("m ago")
    assert _relative_time(now - 7200).endswith("h ago")
    assert _relative_time(now - 3 * 86400).endswith("d ago")
    assert _relative_time(now - 14 * 86400).endswith("w ago")


def test_split_telegram_chunks_respects_limit():
    text = "Paragraph one.\n\n" + ("word " * 1200)
    chunks = split_telegram_chunks(text, limit=500)
    assert len(chunks) > 1
    assert all(len(c) <= 500 for c in chunks)
    rebuilt = " ".join(c.strip() for c in chunks).strip()
    assert "Paragraph one." in rebuilt
    assert "word word" in rebuilt


def test_start_html_includes_personalized_name():
    out = format_start_html("Ari", "0.9.1", first_name="Maya")
    assert "Maya" in out
    assert "/commands" in out
    assert "bumblebee v0.9.1" in out


def test_commands_page_supports_filter():
    out, page, total = format_commands_page(0, query="model")
    assert page == 0
    assert total >= 1
    assert "/models" in out


def test_command_menu_items_are_short():
    items = command_menu_items()
    assert items
    assert any(name == "start" for name, _ in items)
    assert any(name == "tools" for name, _ in items)
    assert any(name == "routines" for name, _ in items)
    assert any(name == "session_start" for name, _ in items)
    assert any(name == "session_status" for name, _ in items)
    assert any(name == "session_stop" for name, _ in items)
    assert any(name == "compact" for name, _ in items)
    assert all(len(desc) <= 256 for _, desc in items)


def test_telegram_registered_slash_names_include_aliases():
    names = telegram_registered_slash_command_names()
    assert "about" in names
    assert "privacy" in names
    assert "private" in names
    assert "whoami" in names
    assert "session_start" in names
    assert "session_status" in names
    assert "session_stop" in names
    assert "compact" in names
    for name, _ in command_menu_items():
        assert name in names


def test_routines_html_lists_automations():
    a = Automation(
        name="Morning check-in",
        description="Brief scan",
        schedule_natural="every day at 9am",
        cron_expression="0 9 * * *",
        origin=AutomationOrigin.SELF,
        enabled=True,
        run_count=2,
        last_result_summary="done",
        deliver_to="telegram:1",
        last_run=__import__("time").time() - 7200,
    )
    out = format_routines_html(
        "Bee",
        [a],
        automations_enabled=True,
        scheduler_ready=True,
    )
    assert "Bee's Routines" in out
    assert "1 active · 0 paused" in out
    assert "Morning check-in" in out
    assert "every day at 9am" in out
    assert "→ Telegram" in out
    assert "Ran 2 times" in out
    assert "2h ago" in out
    assert '"Brief scan"' in out
    assert "0 9 * * *" not in out


def test_routines_html_empty_and_notes():
    out = format_routines_html(
        "Bee",
        [],
        automations_enabled=False,
        scheduler_ready=False,
    )
    assert "No routines yet" in out
    assert "scheduling" in out and "disabled" in out
    assert "daemon is running" in out


def test_routines_html_internal_skips_description():
    a = Automation(
        name="Nightly reflection",
        description="Write in journal",
        schedule_natural="every night at 11pm",
        cron_expression="0 23 * * *",
        origin=AutomationOrigin.INTERNAL,
        enabled=True,
        run_count=1,
        last_run=__import__("time").time() - 3600,
    )
    out = format_routines_html("Canary", [a], automations_enabled=True, scheduler_ready=True)
    assert "📓" in out
    assert "Internal" in out
    assert "Write in journal" not in out


def test_routines_html_sorts_active_before_paused():
    newer = Automation(
        name="A",
        schedule_natural="daily",
        origin=AutomationOrigin.USER,
        enabled=True,
        next_run=200.0,
        run_count=0,
    )
    older = Automation(
        name="B",
        schedule_natural="daily",
        origin=AutomationOrigin.USER,
        enabled=True,
        next_run=100.0,
        run_count=0,
    )
    paused = Automation(
        name="Z",
        schedule_natural="daily",
        origin=AutomationOrigin.USER,
        enabled=False,
        next_run=50.0,
        run_count=0,
    )
    out = format_routines_html("E", [newer, older, paused], automations_enabled=True, scheduler_ready=True)
    pos_b = out.find("<b>⏰  B</b>")
    pos_a = out.find("<b>⏰  A</b>")
    pos_z = out.find("<b>⏸  Z</b>")
    assert pos_b < pos_a < pos_z


def test_tools_html_lists_registered_tools():
    class _Tools:
        def list_tools(self):
            return [
                ("search_web", "Search the web"),
                ("get_current_time", "Get current date and time"),
            ]

    class _Entity:
        tools = _Tools()

    out = format_tools_html(_Entity())
    assert "Active tools" in out
    assert "search_web" in out
    assert "get_current_time" in out


def test_session_status_html_handles_empty_and_active():
    empty = format_session_status_html("Bee", None)
    assert "No active desktop session" in empty
    active = format_session_status_html(
        "Bee",
        {
            "session_id": "sess_123",
            "status": "running",
            "active_app": "Firefox",
            "last_action": "opened docs",
            "summary": "Watching the browser window.",
        },
    )
    assert "sess_123" in active
    assert "Firefox" in active
    assert "opened docs" in active


def test_remote_session_caption_stays_compact():
    caption = format_remote_session_caption(
        "Bee",
        {
            "status": "running",
            "active_app": "Firefox",
            "last_action": "opened docs",
            "summary": "Watching the browser window.",
        },
    )
    assert "Bee remote session" in caption
    assert "Firefox" in caption
    assert len(caption) <= 1024


def test_build_status_html_reports_soma_gen_snapshot():
    class _SessionCtx:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class _Store:
        def session(self):
            return _SessionCtx()

        async def count_episodes(self, _db):
            return 12

        async def count_relationships(self, _db):
            return 4

        async def min_episode_timestamp(self, _db):
            return __import__("time").time() - 3600

    class _Emotions:
        class _State:
            class _Primary:
                value = "curious"

            primary = _Primary()

        def get_state(self):
            return self._State()

    class _Drives:
        def all_drives(self):
            return []

    class _Tools:
        def openai_tools(self):
            return [{"function": {"name": "search_web"}}]

    class _Bars:
        ordered_names = ["social", "curiosity"]
        _active_conflicts = [{"label": "restless comfort"}]
        _active_impulses = [{"label": "reach_out", "on_cooldown": False}]

        def snapshot_pct(self):
            return {"social": 72, "curiosity": 65}

    class _Noise:
        cycle_seconds = 90.0
        temperature = 1.1
        max_tokens = 150

        def current_fragments(self):
            return ["thinking about that deploy note"]

    class _Tonic:
        bars = _Bars()
        noise = _Noise()
        _noise_enabled = True
        _noise_model = ""
        _current_affects = [{"name": "fascination", "intensity": 0.7}]
        _appraisal_enabled = True

    class _Cog:
        reflex_model = "gemma3:4b"
        max_context_tokens = 32000
        rolling_history_max_messages = 40

        class _HistoryCompression:
            enabled = True
            compaction_threshold_ratio = 0.75
            compaction_target_ratio = 0.20

        history_compression = _HistoryCompression()

    class _Harness:
        class _Autonomy:
            enabled = True
            impulse_wake = True
            drive_wake = True
            conflict_wake = True
            noise_wake = False
            desire_wake = True
            desire_wake_threshold = 0.72
            max_desires_considered = 3
            allow_tool_calls_on_wake = True

        class _Memory:
            class _Distillation:
                enabled = True
                cycle_seconds = 300.0

            distillation = _Distillation()

        autonomy = _Autonomy()
        memory = _Memory()

    class _Cfg:
        name = "Bee"
        cognition = _Cog()
        harness = _Harness()
        raw = {"created": "2026-04-09T00:00:00Z"}

    class _Entity:
        store = _Store()
        config = _Cfg()
        emotions = _Emotions()
        drives = _Drives()
        tools = _Tools()
        tonic = _Tonic()

        @staticmethod
        def list_known_person_routes(_platform):
            return ["123"]

    out = __import__("asyncio").run(build_status_html(_Entity(), "1.2.3"))
    assert "Architecture" in out
    assert "SOMA" in out
    assert "GEN" in out
    assert "Autonomy & wake" in out
    assert "Cognition" in out
    assert "Memory pipeline" in out
    assert "Daemon/automations loop" in out
    assert "Dominant bar: social 72%" in out
    assert "GEN model: <code>gemma3:4b</code>" in out
