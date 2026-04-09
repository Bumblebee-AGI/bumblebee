import pytest

from bumblebee.config import HarnessConfig, entity_from_dict
from bumblebee.cognition.deliberate import AgentLoopState, DeliberateCognition
from bumblebee.inference.types import ChatCompletionResult, ToolCallSpec
from bumblebee.models import Input


class _SequenceProvider:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    async def chat_completion(
        self,
        model,
        messages,
        *,
        tools=None,
        temperature=0.7,
        max_tokens=1024,
        think=False,
        stream=False,
        num_ctx=None,
    ):
        _ = (model, messages, tools, temperature, max_tokens, think, stream, num_ctx)
        if self.calls >= len(self._responses):
            raise AssertionError("chat_completion called more times than expected")
        res = self._responses[self.calls]
        self.calls += 1
        return res


@pytest.fixture
def entity_config():
    h = HarnessConfig()
    data = {
        "name": "LoopTest",
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
            "backstory": "Loop test being.",
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


@pytest.mark.asyncio
async def test_loop_continues_past_progress_chatter(entity_config):
    provider = _SequenceProvider(
        [
            ChatCompletionResult(content="nah, i don't see one in here... checking the root directory right now."),
            ChatCompletionResult(content="/app\nknowledge.md not found"),
        ]
    )
    cog = DeliberateCognition(entity_config, provider)
    inp = Input(text="do you have a knowledge.md file?", person_id="u", person_name="U")

    async def final_checker(text: str, res: ChatCompletionResult, state: AgentLoopState):
        _ = (res, state)
        if "checking the root directory" in (text or "").lower():
            return False, "Same turn. That was only progress chatter. Continue until you can answer."
        return True, None

    events = []
    async for ev in cog.iter_responses(
        inp,
        "system",
        [{"role": "user", "content": inp.text}],
        final_checker=final_checker,
    ):
        events.append(ev)

    assert len(events) == 2
    assert events[0].kind == "intermediate"
    assert events[0].display_text == ""
    assert events[1].kind == "final"
    assert "/app" in events[1].display_text


@pytest.mark.asyncio
async def test_loop_continues_after_tool_until_grounded_answer(entity_config):
    provider = _SequenceProvider(
        [
            ChatCompletionResult(
                content="",
                tool_calls=[ToolCallSpec(name="list_directory", arguments={"path": "."}, id="tc1")],
            ),
            ChatCompletionResult(content="done"),
            ChatCompletionResult(content="/app\nentries: knowledge.md missing"),
        ]
    )
    cog = DeliberateCognition(entity_config, provider)
    inp = Input(text="use tools and check the workspace", person_id="u", person_name="U")
    tool_calls = []

    async def tool_executor(spec: ToolCallSpec):
        tool_calls.append(spec.name)
        return '{"ok": true, "path": "/app", "entries": []}'

    async def final_checker(text: str, res: ChatCompletionResult, state: AgentLoopState):
        _ = res
        if state.tool_calls_seen > 0 and (text or "").strip().lower() == "done":
            return False, "Same turn. That is not the final grounded answer. Answer from the tool result."
        return True, None

    events = []
    async for ev in cog.iter_responses(
        inp,
        "system",
        [{"role": "user", "content": inp.text}],
        tools=[{"type": "function", "function": {"name": "list_directory", "description": "", "parameters": {}}}],
        tool_executor=tool_executor,
        final_checker=final_checker,
    ):
        events.append(ev)

    assert tool_calls == ["list_directory"]
    assert len(events) == 3
    assert [ev.kind for ev in events] == ["intermediate", "intermediate", "final"]
    assert all(ev.display_text == "" for ev in events[:-1])
    assert "/app" in events[-1].display_text
