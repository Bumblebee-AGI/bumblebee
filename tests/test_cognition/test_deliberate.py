from bumblebee.config import HarnessConfig, entity_from_dict
from bumblebee.cognition import gemma
from bumblebee.cognition.deliberate import DeliberateCognition, should_inject_tool_continuation
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

