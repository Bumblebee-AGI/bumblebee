"""Display timezone selection for wall clock strings."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

from bumblebee.utils.clock import (
    display_timezone,
    format_wall_clock_tool_line,
    wall_now_for_display,
)


def test_display_timezone_prefers_bumblebee(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BUMBLEBEE_TIMEZONE", "Europe/Berlin")
    monkeypatch.setenv("TZ", "America/New_York")
    assert display_timezone() == ZoneInfo("Europe/Berlin")


def test_display_timezone_invalid_bumblebee_falls_through(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BUMBLEBEE_TIMEZONE", "Not/A_Real_Zone")
    monkeypatch.setenv("TZ", "UTC")
    assert display_timezone() == ZoneInfo("UTC")


def test_display_timezone_skips_colon_tz(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BUMBLEBEE_TIMEZONE", raising=False)
    monkeypatch.setenv("TZ", ":/etc/localtime")
    monkeypatch.delenv("RAILWAY_ENVIRONMENT", raising=False)
    monkeypatch.delenv("BUMBLEBEE_DEPLOYMENT_MODE", raising=False)
    assert display_timezone() is None


def test_display_timezone_railway_defaults_eastern(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BUMBLEBEE_TIMEZONE", raising=False)
    monkeypatch.delenv("TZ", raising=False)
    monkeypatch.setenv("RAILWAY_ENVIRONMENT", "production")
    monkeypatch.delenv("BUMBLEBEE_DEPLOYMENT_MODE", raising=False)
    assert display_timezone() == ZoneInfo("America/New_York")


def test_display_timezone_hybrid_railway_defaults_eastern(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BUMBLEBEE_TIMEZONE", raising=False)
    monkeypatch.delenv("TZ", raising=False)
    monkeypatch.delenv("RAILWAY_ENVIRONMENT", raising=False)
    monkeypatch.setenv("BUMBLEBEE_DEPLOYMENT_MODE", "hybrid_railway")
    assert display_timezone() == ZoneInfo("America/New_York")


def test_wall_now_converts_from_utc(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BUMBLEBEE_TIMEZONE", "America/New_York")
    fixed_utc = datetime(2026, 4, 5, 9, 38, tzinfo=timezone.utc)

    real_datetime = __import__("datetime").datetime

    def fake_now(tz=None):
        if tz is timezone.utc:
            return fixed_utc
        return real_datetime.now()

    with patch("bumblebee.utils.clock.datetime") as mock_dt:
        mock_dt.now = fake_now
        mock_dt.timezone = timezone
        got = wall_now_for_display()
    assert got.hour == 5
    assert got.minute == 38
    assert got.date().isoformat() == "2026-04-05"


def test_format_wall_clock_tool_line_uses_wall_now(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BUMBLEBEE_TIMEZONE", "UTC")
    aware = datetime(2026, 4, 5, 12, 0, tzinfo=timezone.utc)
    with patch("bumblebee.utils.clock.wall_now_for_display", return_value=aware):
        s = format_wall_clock_tool_line()
    assert "12:00" in s
    assert "April" in s
