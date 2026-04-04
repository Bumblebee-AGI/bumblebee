"""Gemma/OpenAI-style tool registration and execution."""

from __future__ import annotations

import inspect
import json
from typing import Any, Awaitable, Callable

from bumblebee.utils.ollama_client import ToolCallSpec


class ToolFn:
    def __init__(self, name: str, description: str, fn: Callable[..., Awaitable[str]]):
        self.name = name
        self.description = description
        self.fn = fn
        self.schema = self._build_schema(fn)

    def _build_schema(self, fn: Callable[..., Any]) -> dict[str, Any]:
        sig = inspect.signature(fn)
        props: dict[str, Any] = {}
        required: list[str] = []
        for pname, p in sig.parameters.items():
            if pname == "self":
                continue
            ann = p.annotation if p.annotation != inspect.Parameter.empty else str
            props[pname] = {"type": "string" if ann is str else "string"}
            if p.default == inspect.Parameter.empty:
                required.append(pname)
        return {
            "type": "object",
            "properties": props,
            "required": required,
        }

    def openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.schema,
            },
        }


def tool(name: str, description: str) -> Callable[[Callable[..., Awaitable[str]]], ToolFn]:
    def deco(fn: Callable[..., Awaitable[str]]) -> ToolFn:
        return ToolFn(name, description, fn)

    return deco


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolFn] = {}

    def register(self, t: ToolFn) -> None:
        self._tools[t.name] = t

    def register_fn(self, name: str, description: str, fn: Callable[..., Awaitable[str]]) -> None:
        self.register(ToolFn(name, description, fn))

    def openai_tools(self) -> list[dict[str, Any]]:
        return [t.openai_tool() for t in self._tools.values()]

    async def execute(self, spec: ToolCallSpec) -> str:
        fn = self._tools.get(spec.name)
        if not fn:
            return json.dumps({"error": f"unknown tool {spec.name}"})
        try:
            kwargs = dict(spec.arguments)
            return await fn.fn(**kwargs)
        except TypeError:
            return await fn.fn()
        except Exception as e:
            return json.dumps({"error": str(e)})


def wrap_simple(coro_fn: Callable[..., Awaitable[str]]) -> ToolFn:
    return ToolFn(coro_fn.__name__, coro_fn.__doc__ or "", coro_fn)
