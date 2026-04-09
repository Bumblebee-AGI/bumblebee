import pytest

from bumblebee.config import HarnessConfig, entity_from_dict
from bumblebee.memory.episodic import EpisodicMemory
from bumblebee.memory.store import MemoryStore
from bumblebee.models import EmotionCategory, Episode


@pytest.fixture
def entity_config():
    h = HarnessConfig()
    data = {
        "name": "Mem",
        "personality": {
            "core_traits": {k: 0.5 for k in ["curiosity", "warmth", "assertiveness", "humor", "openness", "neuroticism", "conscientiousness"]},
            "behavioral_patterns": {},
            "voice": {},
            "backstory": "",
        },
        "drives": {"curiosity_topics": [], "attachment_threshold": 5, "restlessness_decay": 3600, "initiative_cooldown": 1800},
        "cognition": {},
        "presence": {"platforms": [], "daemon": {}},
    }
    return entity_from_dict(h, data)


@pytest.mark.asyncio
async def test_save_and_recall(tmp_path, entity_config):
    db = tmp_path / "t.db"
    store = MemoryStore(str(db))
    mem = EpisodicMemory(entity_config, store)
    conn = await store.connect()
    try:
        ep = Episode(
            id="e1",
            timestamp=1.0,
            summary="hello world",
            participants=["u1"],
            emotional_imprint=EmotionCategory.CONTENT,
            emotional_intensity=0.5,
            significance=0.8,
            tags=["t"],
            embedding=[1.0, 0.0, 0.0],
        )
        await mem.save_episode(conn, ep)
        got = await mem.get_by_id(conn, "e1")
        assert got is not None
        assert got.summary == "hello world"
    finally:
        await conn.close()
