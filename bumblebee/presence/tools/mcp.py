"""MCP integration placeholder — extend when MCP servers are configured."""

from __future__ import annotations

import json


async def mcp_stub(tool_name: str, arguments: str) -> str:
    return json.dumps(
        {
            "status": "not_configured",
            "tool": tool_name,
            "arguments": arguments,
            "hint": "Wire MCP transports in a future release.",
        }
    )
