"""File-backed procedural memory ("skills") for reusable know-how."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bumblebee.config import EntityConfig
    from bumblebee.inference.protocol import InferenceProvider


_SLUG_BAD = re.compile(r"[^a-z0-9._-]+")


def _slugify(name: str) -> str:
    s = (name or "").strip().lower().replace(" ", "-")
    s = _SLUG_BAD.sub("-", s)
    s = s.strip("-._")
    return s or "skill"


@dataclass
class SkillEntry:
    name: str
    slug: str
    content: str


class ProceduralMemoryStore:
    def __init__(self, entity: EntityConfig, client: InferenceProvider) -> None:
        self.entity = entity
        self.client = client
        self.path = Path(entity.skills_dir()).expanduser()
        self.path.mkdir(parents=True, exist_ok=True)

    def _file_path(self, name: str) -> Path:
        return self.path / f"{_slugify(name)}.md"

    async def upsert_skill(self, name: str, content: str) -> SkillEntry:
        raw_name = (name or "").strip()
        raw_content = (content or "").strip()
        if not raw_name:
            raise RuntimeError("skill name is required")
        if not raw_content:
            raise RuntimeError("skill content is required")
        slug = _slugify(raw_name)
        p = self._file_path(raw_name)

        def _write() -> None:
            p.write_text(raw_content + ("\n" if not raw_content.endswith("\n") else ""), encoding="utf-8")

        await asyncio.get_running_loop().run_in_executor(None, _write)
        return SkillEntry(name=raw_name, slug=slug, content=raw_content)

    async def read_skill(self, name: str) -> SkillEntry | None:
        p = self._file_path(name)
        if not p.is_file():
            return None

        def _read() -> str:
            return p.read_text(encoding="utf-8", errors="replace")

        content = await asyncio.get_running_loop().run_in_executor(None, _read)
        return SkillEntry(name=p.stem, slug=p.stem, content=content)

    async def list_skills(self) -> list[SkillEntry]:
        def _scan() -> list[SkillEntry]:
            rows: list[SkillEntry] = []
            for p in sorted(self.path.glob("*.md")):
                try:
                    content = p.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                rows.append(SkillEntry(name=p.stem, slug=p.stem, content=content))
            return rows

        return await asyncio.get_running_loop().run_in_executor(None, _scan)

    async def query(self, text: str, limit: int = 3) -> list[str]:
        q = (text or "").strip().lower()
        if not q:
            return []
        rows = await self.list_skills()
        scored: list[tuple[int, SkillEntry]] = []
        for row in rows:
            body = row.content.lower()
            score = 0
            if q in row.slug or q in body:
                score += 10
            for token in {tok for tok in re.split(r"\W+", q) if len(tok) >= 4}:
                if token in row.slug:
                    score += 3
                if token in body:
                    score += 1
            if score > 0:
                scored.append((score, row))
        scored.sort(key=lambda x: (-x[0], x[1].slug))
        out: list[str] = []
        for _, row in scored[: max(1, min(limit, 8))]:
            snippet = row.content.strip()
            if len(snippet) > 900:
                snippet = snippet[:900].rstrip() + "\n… [truncated]"
            out.append(f"procedural memory: {row.name}\n\n{snippet}")
        return out
