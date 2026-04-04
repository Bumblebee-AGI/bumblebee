from bumblebee.cognition import gemma


def test_parse_thought_channel_exact_tokens():
    text = f"{gemma.CHANNEL_THOUGHT}\nsecret line\n{gemma.CHANNEL_END}Hello user"
    p = gemma.parse_assistant_output(text)
    assert p.thinking == "secret line"
    assert p.visible_user_text == "Hello user"


def test_parse_tool_call_gemma_delimiters():
    inner = 'call:weather{location:<|"|>London<|"|>}'
    raw = f"{gemma.TOOL_CALL_START}{inner}{gemma.TOOL_CALL_END}"
    p = gemma.parse_assistant_output(raw + "Nice day.")
    assert len(p.tool_call_raws) == 1
    name, args = gemma.parse_tool_call_inner(p.tool_call_raws[0])
    assert name == "weather"
    assert args.get("location") == "London"


def test_strip_thinking_for_history_preserves_tool_call():
    thought = f"{gemma.CHANNEL_THOUGHT}\ninside\n{gemma.CHANNEL_END}"
    tc = f"{gemma.TOOL_CALL_START}call:x{{}}{gemma.TOOL_CALL_END}"
    text = thought + tc + "tail"
    stripped = gemma.strip_thinking_for_history(text)
    assert "inside" not in stripped
    assert "call:x" in stripped
    assert "tail" in stripped


def test_inject_thinking():
    s = "You are kind."
    out = gemma.inject_thinking_instruction(s, True)
    assert gemma.THINK in out


def test_empty_thought_fragment_contains_channel():
    frag = gemma.empty_thought_channel_fragment()
    assert gemma.CHANNEL_THOUGHT in frag
    assert gemma.CHANNEL_END in frag


def test_separate_plaintext_cot_then_reply():
    raw = (
        "Thinking Process:\n\n"
        "1. Analyze input.\n"
        "2. Draft reply.\n\n"
        "yeah, man. just vibing."
    )
    vis, think = gemma.separate_plaintext_chain_of_thought(raw)
    assert "yeah, man" in vis
    assert "Thinking Process" in think
    assert "Analyze input" in think


def test_separate_plaintext_cot_only_triggers_empty_visible():
    raw = "Thinking Process:\n\n1. foo\n2. bar"
    vis, think = gemma.separate_plaintext_chain_of_thought(raw)
    assert vis == ""
    assert think == raw
