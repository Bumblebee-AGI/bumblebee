from bumblebee.presence.platforms.telegram_format import (
    command_menu_items,
    format_commands_page,
    format_start_html,
    split_telegram_chunks,
)


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
    assert all(len(desc) <= 256 for _, desc in items)
