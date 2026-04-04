"""Wall-clock tool for when the model wants to double-check the time."""

from __future__ import annotations

from datetime import datetime

from bumblebee.presence.tools.registry import tool


@tool(
    name="get_current_time",
    description=(
        "Get the current date and time. Use this if you need to know exactly what time it is right now."
    ),
)
async def get_current_time() -> str:
    now = datetime.now()
    return now.strftime("%A, %B %d, %Y at %I:%M %p")
