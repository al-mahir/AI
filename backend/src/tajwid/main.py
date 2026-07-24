"""Tajwid AI service: streaming Quran recitation feedback.

audio -> silero VAD endpointing -> Muaalem phonemes+sifat -> locate/track ->
per-word feedback, over one WebSocket. See api/ws.py for the protocol.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.rest import router as rest_router
from .api.ws import router as ws_router
from .asr.engine import build_engines
from .config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build every eagerly-available engine once, in a single thread, before
    # serving — same reasoning as before (avoid a concurrent-first-call race
    # triggering two simultaneous GPU model loads), just now producing a
    # dict instead of one engine so a session can pick per-connection (see
    # api/ws.py). app.state.default_engine_name is what an omitted/unknown
    # per-session choice falls back to.
    settings = get_settings()
    app.state.engines = await asyncio.to_thread(build_engines, settings)
    app.state.default_engine_name = settings.resolved_asr_engine
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Tajwid recitation feedback service",
        description=__doc__,
        version="1.0.0",
        lifespan=lifespan,
    )
    # The Java Spring backend / dev frontend call us from another origin (VPC-internal
    # in production). Tighten with an env-based allowlist when the topology is known.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(rest_router)
    app.include_router(ws_router)

    @app.get("/")
    def index() -> dict:
        s = get_settings()
        return {
            "service": "Tajwid recitation feedback",
            "engine": s.resolved_asr_engine,
            "docs": "/docs",
            "endpoints": {
                "GET /health": "status + loaded engine",
                "POST /transcribe-file": "offline: upload a recitation, get chunk transcripts",
                "WS /ws/session": "live: JSON start config, then 16 kHz mono PCM16-LE "
                "frames; per-waqf-chunk word feedback comes back",
            },
        }

    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8100)


if __name__ == "__main__":
    main()
