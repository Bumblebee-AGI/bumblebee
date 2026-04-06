from bumblebee.config import HarnessConfig, entity_from_dict
from bumblebee.cognition import gemma
from bumblebee.cognition.deliberate import (
    DeliberateCognition,
    fit_system_prompt_to_budget,
    should_inject_tool_continuation,
)
from bumblebee.presence.tools.registry import TOOL_SYSTEM_PROMPT_PREFIX
from bumblebee.inference.types import ChatCompletionResult, ToolCallSpec


def _entity_config():
    h = HarnessConfig()
    data = {
        "name": "Test",
        "personality": {
            "core_traits": {
                k: 0.5
                for k in [
                    "curiosity",
                    "warmth",
                    "assertiveness",
                    "humor",
                    "openness",
                    "neuroticism",
                    "conscientiousness",
                ]
            },
            "behavioral_patterns": {},
            "voice": {},
            "backstory": "Test being.",
        },
        "drives": {
            "curiosity_topics": [],
            "attachment_threshold": 5,
            "restlessness_decay": 3600,
            "initiative_cooldown": 1800,
        },
        "cognition": {},
        "presence": {"platforms": [{"type": "cli"}], "daemon": {}},
    }
    return entity_from_dict(h, data)


def test_fit_system_prompt_keeps_tools_suffix_when_truncating():
    prefix = "A" * 500
    suffix = f"{TOOL_SYSTEM_PROMPT_PREFIX} — rest of tools block here."
    full = prefix + suffix
    cap = 400
    out = fit_system_prompt_to_budget(full, cap)
    assert suffix in out
    assert out.endswith("rest of tools block here.")


def test_build_messages_truncation_preserves_tool_appendix():
    ec = _entity_config()
    # _build_messages uses max(2000, cognition.system_prompt_char_limit)
    ec.cognition.system_prompt_char_limit = 2000
    d = DeliberateCognition(ec, client=object())  # type: ignore[arg-type]
    long_prefix = "P" * 3500
    sys_full = long_prefix + f"{TOOL_SYSTEM_PROMPT_PREFIX}\nYou may call: x."
    msgs = d._build_messages(sys_full, [{"role": "user", "content": "hi"}])
    body = msgs[0]["content"]
    assert TOOL_SYSTEM_PROMPT_PREFIX in body
    assert "You may call: x." in body
    assert len(body) <= 2000


def test_build_messages_preserves_user_multimodal_content():
    ec = _entity_config()
    d = DeliberateCognition(ec, client=object())  # type: ignore[arg-type]
    img_payload = [
        {"type": "text", "text": "what do you think of her?"},
        {
            "type": "image_url",
            "image_url": {
                "url": "data:image/jpeg;base64,AAA",
                "detail": "auto",
            },
        },
    ]
    msgs = d._build_messages(
        "sys",
        [
            {"role": "user", "content": img_payload},
            {
                "role": "assistant",
                "content": f"{gemma.CHANNEL_THOUGHT}\nsecret\n{gemma.CHANNEL_END}hi",
            },
        ],
    )
    assert isinstance(msgs[1]["content"], list)
    assert msgs[1]["content"][1]["type"] == "image_url"
    assert "secret" not in msgs[2]["content"]


def test_should_inject_tool_continuation_false_without_prior_tools():
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}]
    res = ChatCompletionResult(content="just chatting", tool_calls=[])
    assert should_inject_tool_continuation(msgs, res) is False


def test_should_inject_after_tool_error_and_install_intent():
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "list voices"},
        {"role": "tool", "content": '{"ok": false, "error": "edge-tts not installed"}'},
    ]
    res = ChatCompletionResult(
        content="missing package — hang on, let me install it",
        tool_calls=[],
    )
    assert should_inject_tool_continuation(msgs, res) is True


def test_should_inject_on_length_finish_with_tools_in_context():
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "tool", "content": '{"ok": true}'},
    ]
    res = ChatCompletionResult(content="partial", tool_calls=[], finish_reason="length")
    assert should_inject_tool_continuation(msgs, res) is True


def test_should_not_inject_when_model_emits_tools():
    msgs = [{"role": "tool", "content": '{"error": "x"}'}]
    res = ChatCompletionResult(
        content="",
        tool_calls=[ToolCallSpec(name="run_command", arguments={}, id="1")],
    )
    assert should_inject_tool_continuation(msgs, res) is False


def test_should_not_inject_on_try_to_in_normal_prose_after_tool():
    """'try to land' etc. must not match — spurious continuation produced 'I'm done' replies."""
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "tool", "content": '{"ok": true, "title": "Artemis"}'},
    ]
    res = ChatCompletionResult(
        content=(
            "before they actually try to land people on the moon later on. "
            "it's basically the test run."
        ),
        tool_calls=[],
        finish_reason="stop",
    )
    assert should_inject_tool_continuation(msgs, res) is False

