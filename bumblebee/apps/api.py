"""Thin HTTP service: health and readiness (webhooks can attach here later)."""

from __future__ import annotations

import os

from fastapi import FastAPI

from bumblebee.config import load_entity_config, load_harness_config
from bumblebee.entity import Entity
from bumblebee.health import check_database, check_inference


def create_api_app() -> FastAPI:
    app = FastAPI(title="Bumblebee API", version="0.1.0")

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "service": "bumblebee-api"}

    @app.get("/ready")
    async def ready() -> dict:
        name = (os.environ.get("BUMBLEBEE_ENTITY") or "").strip()
        if not name:
            return {"ready": True, "detail": "no BUMBLEBEE_ENTITY — shallow ready only"}
        harness = load_harness_config()
        ec = load_entity_config(name, harness)
        ent = Entity(ec)
        try:
            inf = await check_inference(ec, ent.client)
            db = await check_database(ent.store)
            ok = bool(inf.get("ok")) and bool(db.get("ok"))
            return {"ready": ok, "inference": inf, "database": db}
        finally:
            await ent.shutdown()

    return app
