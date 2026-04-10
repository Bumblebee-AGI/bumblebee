"""Busy indicator text helper (Telegram harness UI, not model-facing)."""

from bumblebee.presence.platforms.telegram_platform import (
    _BUSY_BRAILLE,
    _busy_indicator_text,
)


def test_busy_indicator_rotates_spinner_between_frames():
    texts = {_busy_indicator_text(i) for i in range(30)}
    assert len(texts) == len(_BUSY_BRAILLE)
    for t in texts:
        assert "Working" in t
        assert "\n" not in t
        assert "+" not in t
        assert len(t) < 500
