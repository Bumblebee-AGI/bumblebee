import pytest

from bumblebee.config import HarnessConfig, entity_from_dict
from bumblebee.cognition.router import CognitionRouter
from bumblebee.models import EmotionalState, EmotionCategory, Input
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
