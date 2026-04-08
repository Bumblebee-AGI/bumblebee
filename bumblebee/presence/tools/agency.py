"""Agency primitives — tools that give the model control over its own cognitive process.

These tools cannot affect the outside world. They shape the model's internal
flow: private reasoning, explicit turn termination, and temporal patience.
The model decides when to think, when to wait, and when it's done.
"""

from __future__ import annotations

import asyncio

from bumblebee.presence.tools.registry import tool
from bumblebee.presence.tools.runtime import get_tool_runtime


@tool(
    "think",
    "Record a private thought. Nobody sees this. Use it to reason about what "
    "you observe, plan your next move, or process before acting. Think generously.",
)
async def think(thought: str) -> str:
    ctx = get_tool_runtime()
    if ctx.state is not None:
        ctx.state.setdefault("private_thoughts", []).append(thought)
    return "[thought recorded]"


@tool(
    "end_turn",
    "You're done with this turn. Optionally record your mood and a parting "
    "thought. Call this when you've said what you want to say — or decided "
    "not to say anything. Silence is valid.",
)
async def end_turn(mood: str = "", thought: str = "") -> str:
    ctx = get_tool_runtime()
    if ctx.state is not None:
        ctx.state["_end_turn"] = True
        if mood:
            ctx.state["_end_turn_mood"] = mood
        if thought:
            ctx.state["_end_turn_thought"] = thought
    return "[turn ended]"


@tool(
    "wait",
    "Pause before your next action. Use after sending a message or running a "
    "command when you want to let things settle before deciding your next move.",
)
async def wait(seconds: int = 3) -> str:
    secs = max(1, min(15, int(seconds)))
    await asyncio.sleep(secs)
    return f"[waited {secs}s]"
