import pytest

from bumblebee.config import HarnessConfig, entity_from_dict
from bumblebee.cognition.router import CognitionRouter, ContextPackage
from bumblebee.models import EmotionalState, Input
from bumblebee.utils.ollama_client import OllamaClient


@pytest.fixture
def entity_config():
    h = HarnessConfig()
    data = {
        "name": "Test",
        "personality": {
            "core_traits": {k: 0.5 for k in ["curiosity", "warmth", "assertiveness", "humor", "openness", "neuroticism", "conscientiousness"]},
            "behavioral_patterns": {},
            "voice": {},
            "backstory": "Test being.",
        },
        "drives": {"curiosity_topics": [], "attachment_threshold": 5, "restlessness_decay": 3600, "initiative_cooldown": 1800},
        "cognition": {},
        "presence": {"platforms": [{"type": "cli"}], "daemon": {}},
    }
    return entity_from_dict(h, data)


def test_heuristic_reflex_greeting(entity_config):
    r = CognitionRouter(entity_config, OllamaClient())
    inp = Input(text="hi", person_id="u", person_name="U")
    emo = EmotionalState()
    assert r.heuristic_route(inp, emo) == "reflex"


def test_heuristic_deliberate_long(entity_config):
    r = CognitionRouter(entity_config, OllamaClient())
    inp = Input(text="x" * 300, person_id="u", person_name="U")
    emo = EmotionalState()
    assert r.heuristic_route(inp, emo) == "deliberate"


def test_heuristic_identity_questions_not_reflex_short_q(entity_config):
    r = CognitionRouter(entity_config, OllamaClient())
    emo = EmotionalState()
    for text in (
        "what is your name?",
        "who are you?",
        "what should I call you?",
    ):
        inp = Input(text=text, person_id="u", person_name="U")
        assert r.heuristic_route(inp, emo) == "deliberate"


def test_heuristic_capability_questions_use_deliberate(entity_config):
    r = CognitionRouter(entity_config, OllamaClient())
    emo = EmotionalState()
    inp = Input(text="what can you do?", person_id="u", person_name="U")
    assert r.heuristic_route(inp, emo) == "deliberate"


@pytest.mark.asyncio
async def test_route_short_casual_skips_classifier_escalation(entity_config):
    r = CognitionRouter(entity_config, OllamaClient())
    emo = EmotionalState()
    ctx = ContextPackage(emotional_state=emo)
    inp = Input(text="nah nuthin much here either, wagwan", person_id="u", person_name="U")
    kind, _ = await r.route(inp, emo, ctx)
    assert kind == "reflex"


@pytest.mark.asyncio
async def test_route_tiny_what_stays_reflex(entity_config):
    r = CognitionRouter(entity_config, OllamaClient())
    emo = EmotionalState()
    ctx = ContextPackage(emotional_state=emo)
    inp = Input(text="WHAT?", person_id="u", person_name="U")
    kind, _ = await r.route(inp, emo, ctx)
    assert kind == "reflex"
