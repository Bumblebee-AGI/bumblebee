"""Narrow FastAPI surface: inference only (no host tools, no shell).

Security posture (non-negotiable):
- This process is a **private inference appliance** on the home machine. It must not expose
  shell, filesystem, Bumblebee tools, browser automation, admin consoles, or any host-control API.
- **Cloudflare Tunnel (or equivalent) must terminate only at this gateway** — forward to
  ``127.0.0.1:<gateway_port>`` — never at a generic reverse proxy that also exposes SSH, home
  dashboards, NAS UI, or other host services. The tunnel must not become a broad path into
  the user's computer; only this HTTP surface is in scope.

Exposed HTTP surface (all require ``Authorization: Bearer <INFERENCE_GATEWAY_TOKEN>`` except
that unauthenticated requests receive 401/403 before route logic):
- ``GET /health`` — liveness and downstream model check
- ``GET /v1/models`` — model list (proxied)
- ``POST /v1/chat/completions`` — chat (optional streaming)
- ``POST /v1/embeddings`` — embeddings

``HEAD`` on ``/health`` and ``/v1/models`` is allowed for simple probes.

OpenAPI/Swagger UI is **disabled** so the tunneled origin does not ship interactive admin docs.
Unknown methods/paths receive **404** with a small JSON body (see ``_reject_unknown_surface``).

Rate limiting is **not** implemented here; use Cloudflare / edge (see docs).
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.responses import Response

from bumblebee.inference_gateway.backends.ollama import OllamaCompatibleBackend

log = logging.getLogger("bumblebee.inference_gateway")

# Authoritative allowlist — must stay in sync with route handlers below.
GATEWAY_ALLOWED_METHOD_PATH: frozenset[tuple[str, str]] = frozenset(
    {
        ("GET", "/health"),
        ("HEAD", "/health"),
        ("GET", "/v1/models"),
        ("HEAD", "/v1/models"),
        ("POST", "/v1/chat/completions"),
        ("POST", "/v1/embeddings"),
    }
)


def _expected_token() -> str:
    return (os.environ.get("INFERENCE_GATEWAY_TOKEN") or "").strip()


def _max_body() -> int:
    raw = (os.environ.get("INFERENCE_GATEWAY_MAX_BODY_BYTES") or "8000000").strip()
    try:
        return max(64_000, int(raw))
    except ValueError:
        return 8_000_000


def _backend_timeout() -> float:
    raw = (os.environ.get("INFERENCE_GATEWAY_BACKEND_TIMEOUT") or "120").strip()
    try:
        return max(5.0, float(raw))
    except ValueError:
        return 120.0


def _normalize_path(path: str) -> str:
    p = path.rstrip("/") or "/"
    return p if p.startswith("/") else f"/{p}"


def _reject_unknown_surface(method: str, raw_path: str) -> JSONResponse | None:
    m = method.upper()
    path = _normalize_path(raw_path)
    if (m, path) in GATEWAY_ALLOWED_METHOD_PATH:
        return None
    log.info(
        "gateway_rejected_route",
        extra={"method": m, "path": raw_path},
    )
    return JSONResponse(
        status_code=404,
        content={
            "error": "not_found",
            "service": "bumblebee-inference-gateway",
            "message": (
                "Only inference routes exist: GET /health, GET /v1/models, "
                "POST /v1/chat/completions, POST /v1/embeddings. "
                "No shell, filesystem, tools, or admin API is exposed."
            ),
        },
    )


def create_app() -> FastAPI:
    ollama_url = (os.environ.get("OLLAMA_BASE_URL") or "http://127.0.0.1:11434").strip().rstrip("/")
    backend = OllamaCompatibleBackend(
        ollama_url,
        timeout=_backend_timeout(),
        max_body_bytes=_max_body(),
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        yield
        await backend.aclose()

    app = FastAPI(
        title="Bumblebee Inference Gateway",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def limit_body(request: Request, call_next):
        max_b = _max_body()
        cl = request.headers.get("content-length")
        if cl:
            try:
                if int(cl) > max_b:
                    return JSONResponse(
                        status_code=413,
                        content={"error": "payload_too_large", "max_bytes": max_b},
                    )
            except ValueError:
                pass
        return await call_next(request)

    @app.middleware("http")
    async def inference_surface_only(request: Request, call_next):
        """Registered after ``limit_body`` so this runs first inbound — reject junk paths early."""
        rej = _reject_unknown_surface(request.method, request.url.path)
        if rej is not None:
            return rej
        return await call_next(request)

    def _auth(authorization: str | None) -> None:
        exp = _expected_token()
        if not exp:
            raise HTTPException(
                status_code=503,
                detail="server_misconfigured: INFERENCE_GATEWAY_TOKEN not set",
            )
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="missing_bearer")
        if authorization[7:].strip() != exp:
            raise HTTPException(status_code=403, detail="invalid_token")

    @app.get("/health")
    async def health(authorization: str | None = Header(default=None)) -> dict[str, Any]:
        _auth(authorization)
        t0 = time.perf_counter()
        try:
            await backend.get_models()
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            return {
                "ok": True,
                "backend": "ollama_compatible",
                "ollama_base": ollama_url,
                "latency_ms": latency_ms,
            }
        except Exception as e:
            log.warning("health_failed", extra={"error": str(e)})
            return {"ok": False, "error": str(e)[:300]}

    @app.head("/health")
    async def health_head(authorization: str | None = Header(default=None)) -> Response:
        """Minimal probe without JSON body (auth still required)."""
        _auth(authorization)
        return Response(status_code=200)

    @app.get("/v1/models")
    async def list_models(authorization: str | None = Header(default=None)) -> Any:
        _auth(authorization)
        data = await backend.get_models()
        return data

    @app.head("/v1/models")
    async def list_models_head(authorization: str | None = Header(default=None)) -> Response:
        _auth(authorization)
        return Response(status_code=200)

    @app.post("/v1/chat/completions")
    async def chat_completions(
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> Any:
        _auth(authorization)
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="invalid_json") from None
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="expected_object")
        if body.get("stream"):
            resp = await backend.open_chat_completions_stream(body)
            if resp.status >= 400:
                err = await resp.text()
                await resp.release()
                raise HTTPException(status_code=resp.status, detail=err[:800])

            async def gen():
                try:
                    async for chunk in resp.content.iter_chunked(8192):
                        if chunk:
                            yield chunk
                finally:
                    await resp.release()

            ct = resp.headers.get("content-type", "text/event-stream")
            return StreamingResponse(gen(), status_code=resp.status, media_type=ct)

        status, result = await backend.post_chat_completions(body)
        if status >= 400:
            return JSONResponse(status_code=status, content=result if isinstance(result, dict) else {"error": str(result)})
        return result

    @app.post("/v1/embeddings")
    async def embeddings(
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> Any:
        _auth(authorization)
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="invalid_json") from None
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="expected_object")
        status, result = await backend.post_embeddings(body)
        if status >= 400:
            return JSONResponse(status_code=status, content=result)
        return result

    return app
