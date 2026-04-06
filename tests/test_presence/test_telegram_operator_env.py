"""BUMBLEBEE_TELEGRAM_OPERATOR_IDS merged with YAML operator_user_ids."""

from __future__ import annotations

import pytest

from bumblebee.presence.platforms.telegram_platform import merge_telegram_operator_user_ids


@pytest.mark.parametrize(
    ("yaml_ids", "env", "expect"),
    [
        ([], "", None),
        ([123], "", {123}),
        ([], "456", {456}),
        ([123], "456", {123, 456}),
        ([123], " 456 , 789 ", {123, 456, 789}),
        ([], "notanumber,42", {42}),
    ],
)
def test_merge_telegram_operator_user_ids(
    monkeypatch: pytest.MonkeyPatch,
    yaml_ids: list[int],
    env: str,
    expect: set[int] | None,
) -> None:
    monkeypatch.delenv("BUMBLEBEE_TELEGRAM_OPERATOR_IDS", raising=False)
    if env:
        monkeypatch.setenv("BUMBLEBEE_TELEGRAM_OPERATOR_IDS", env)
    got = merge_telegram_operator_user_ids(yaml_ids)
    assert got == expect
