"""Abstract platform interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable


class Platform(ABC):
    @abstractmethod
    async def connect(self) -> None:
        ...

    @abstractmethod
    async def send_message(self, channel: str, content: str) -> None:
        ...

    @abstractmethod
    async def send_tool_activity(self, description: str) -> None:
        """Harness-only status line before an external tool runs (web, file, MCP)."""
        ...

    @abstractmethod
    async def on_message(self, callback: Callable[..., Any]) -> None:
        ...

    @abstractmethod
    async def set_presence(self, status: str) -> None:
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        ...

    @abstractmethod
    def get_person_id(self, message: Any) -> str:
        ...

    @abstractmethod
    def get_person_name(self, message: Any) -> str:
        ...

    async def send_attachment_bytes(
        self,
        channel: str,
        data: bytes,
        *,
        content_type: str | None = None,
        filename: str = "attachment.bin",
    ) -> None:
        """Send binary outbound (e.g. bytes loaded from ``Entity.read_stored_attachment``). CLI: no-op."""
        return None

    async def send_audio(self, channel: str, path: str) -> bool:
        """Send an audio/voice file from disk. Return ``True`` on successful delivery."""
        return False

    async def send_image(self, channel: str, path: str) -> None:
        """Send an image file from disk (platform-specific best effort)."""
        return None
