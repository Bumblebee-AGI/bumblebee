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
