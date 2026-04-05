from bumblebee.identity.voice import strip_stage_directions


def test_strip_leading_parenthetical():
    raw = "(Chuckles softly)\n\nNah, nothing major."
    assert strip_stage_directions(raw) == "Nah, nothing major."


def test_strip_whole_line_parenthetical():
    raw = "Oh. Right.\n\n(A beat of silence.)\n\nMan, you know how it is."
    out = strip_stage_directions(raw)
    assert "(A beat of silence.)" not in out
    assert "Man, you know how it is." in out


def test_strip_lone_asterisk_action_line():
    raw = "*shrugs*\n\nwhatever"
    assert strip_stage_directions(raw) == "whatever"


def test_strip_media_html_tags():
    raw = '<audio src="/tmp/bb_voice_1.mp3"></audio>\n\nstill here'
    assert strip_stage_directions(raw) == "still here"
