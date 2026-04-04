"""Entity-subjective time (wall clock + session offset)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class EntityClock:
    """Tracks subjective session time alongside wall time."""

    session_start: float = field(default_factory=time.time)
    subjective_offset: float = 0.0

    def now(self) -> float:
        return time.time()

    def subjective_elapsed(self) -> float:
        return (time.time() - self.session_start) + self.subjective_offset
