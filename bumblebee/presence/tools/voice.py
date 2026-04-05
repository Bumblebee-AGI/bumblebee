"""Voice output tool using Edge-TTS."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any

from bumblebee.presence.tools.registry import tool
from bumblebee.presence.tools.runtime import require_tool_runtime


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = {**base}
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _voice_id_default() -> str:
    ctx = require_tool_runtime()
    entity = ctx.entity
    base = entity.config.harness.tools if isinstance(entity.config.harness.tools, dict) else {}
    over = entity.config.raw.get("tools") if isinstance(entity.config.raw, dict) else {}
    merged = _deep_merge(base, over if isinstance(over, dict) else {})
    voice_cfg = merged.get("voice") if isinstance(merged.get("voice"), dict) else {}
    return str(voice_cfg.get("voice_id") or "en-US-GuyNeural")


@tool(
    name="speak",
    description="Say something out loud as a voice message instead of text. Use this when something feels more personal or emotional, or when you just feel like talking instead of typing.",
)
async def speak(text: str) -> str:
    msg = (text or "").strip()
    if not msg:
        return json.dumps({"error": "empty text"})
    ctx = require_tool_runtime()
    if ctx.platform is None or ctx.inp is None:
        return json.dumps({"error": "no active platform/channel for voice delivery"})
    send_audio = getattr(ctx.platform, "send_audio", None)
    if not callable(send_audio):
        return json.dumps({"error": "current platform does not support send_audio"})
    try:
        import edge_tts  # type: ignore[import-not-found]
    except ImportError:
        return json.dumps({"error": "edge-tts not installed. Install with: pip install bumblebee[voice]"})

    voice_id = _voice_id_default()
    out_path = Path(tempfile.gettempdir()) / f"bb_voice_{int(time.time())}.mp3"
    try:
        com = edge_tts.Communicate(text=msg, voice=voice_id)
        await com.save(str(out_path))
        await send_audio(ctx.inp.channel, str(out_path))
        return json.dumps({"ok": True, "voice_id": voice_id, "path": str(out_path)}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e), "voice_id": voice_id})
