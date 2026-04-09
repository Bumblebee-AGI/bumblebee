"""Choose SQLite file vs Postgres from entity config + DATABASE_URL."""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

from bumblebee.memory.postgres_store import PostgresMemoryStore
from bumblebee.memory.store import MemoryStore

if TYPE_CHECKING:
    from bumblebee.config import EntityConfig

MemoryStoreBackend = Union[MemoryStore, PostgresMemoryStore]


def create_memory_store(entity: EntityConfig) -> MemoryStoreBackend:
    url = entity.database_url()
    if url.startswith("postgresql://") or url.startswith("postgres://"):
        return PostgresMemoryStore(url, display_name="postgresql")
    return MemoryStore(entity.db_path())
