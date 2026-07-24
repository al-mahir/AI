"""Live recitation WebSocket: ``WS /ws/session``.

Protocol (all text messages are JSON):

  client -> server
    first message   {"type": "start", "sura": 1, "aya": 1, "word_idx": 0,
                     "strictness": "normal"?, "moshaf": {...}?, "include_units": false?,
                     "engine": "zipformer"?}
                    The start position SEEDS THE CURSOR — required in practice:
                    a cold session cannot identify its own first chunk (the basmalah
                    ambiguity), and the app always knows what the user picked.
                    "engine" picks from whatever main.py's lifespan built into
                    app.state.engines (currently: "real"/"mock"/"zipformer",
                    whichever Settings.resolved_asr_engine resolves to, plus
                    "zipformer" always). Omitted or unrecognized -> falls back to
                    app.state.default_engine_name (documented, not surfaced as an
                    error to the client — the "session" ack's "engine" field is
                    the source of truth for what actually got used).
    binary          a frame of 16 kHz mono PCM16 little-endian audio.
    {"type":"seek", "sura","aya","word_idx"}   user jumped elsewhere; cursor resets.
    {"type":"end"}  (or bare "end")           flush and close.

  server -> client
    {"type":"session", "session_id", "engine"}         on start.
    {"type":"partial", "phonemes": "..."}              streaming engines only: emitted
                                                       on every audio frame that changes
                                                       the partial decode hypothesis.
                                                       Not emitted for ChunkEngine
                                                       (Muaalem/Mock). Superseded by
                                                       the next "feedback" event.
    {"type":"feedback", ...}                           one per finalized waqf chunk
                                                       (see tajwid.session.LiveSession).
    {"type":"done"}                                    after the end-of-stream flush.
"""

from __future__ import annotations

import asyncio
import json
import uuid

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from quran_transcript import MoshafAttributes

from ..config import get_settings
from ..feedback.types import Span
from ..session import LiveSession

router = APIRouter()


def _pcm16_to_float(data: bytes) -> np.ndarray:
    """Interpret raw little-endian PCM16 bytes as float32 in [-1, 1]."""
    if not data:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def _span_of(msg: dict) -> Span | None:
    if "sura" not in msg or "aya" not in msg:
        return None
    return Span(sura=int(msg["sura"]), aya=int(msg["aya"]), word_idx=int(msg.get("word_idx", 0)))


def _is_end(text: str) -> bool:
    if text.strip() == "end":
        return True
    try:
        return json.loads(text).get("type") == "end"
    except (json.JSONDecodeError, AttributeError):
        return False


@router.websocket("/ws/session")
async def ws_session(websocket: WebSocket) -> None:
    await websocket.accept()
    engines: dict = websocket.app.state.engines
    default_engine_name: str = websocket.app.state.default_engine_name
    session_id = str(uuid.uuid4())

    # First message must be the start config (it seeds the cursor).
    try:
        raw = await websocket.receive_text()
        cfg = json.loads(raw)
    except (WebSocketDisconnect, json.JSONDecodeError):
        await websocket.close(code=1002)
        return

    requested_engine_name = cfg.get("engine")
    engine = engines.get(requested_engine_name) or engines[default_engine_name]

    # A moshaf from the client may be an invalid combination (e.g. madd al-leen longer
    # than madd al-aared). Don't let that tear down the whole session on its first
    # message: fall back to the default recitation rather than dropping the connection.
    moshaf = None
    if cfg.get("moshaf"):
        try:
            moshaf = MoshafAttributes(**cfg["moshaf"])
        except Exception:  # noqa: BLE001 — any bad-config shape, not just ValidationError
            moshaf = None

    session = LiveSession(
        engine,
        session_id=session_id,
        moshaf=moshaf,
        start=_span_of(cfg),
        strictness=cfg.get("strictness"),
        include_units=bool(cfg.get("include_units", False)),
    )

    await websocket.send_text(
        json.dumps(
            {
                "type": "session",
                "session_id": session_id,
                "engine": getattr(engine, "name", "unknown"),
                "sample_rate": get_settings().sample_rate,
            }
        )
    )

    async def send_events(events: list[dict]) -> None:
        for e in events:
            await websocket.send_text(json.dumps(e, ensure_ascii=False))

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break

            data = message.get("bytes")
            text = message.get("text")

            if data is not None:
                samples = _pcm16_to_float(data)
                events = await asyncio.to_thread(session.feed, samples)
                await send_events(events)

            elif text is not None:
                if _is_end(text):
                    events = await asyncio.to_thread(session.flush)
                    await send_events(events)
                    await websocket.send_text(json.dumps({"type": "done"}))
                    break
                try:
                    msg = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if msg.get("type") == "seek" and (span := _span_of(msg)):
                    session.seek(span)

    except WebSocketDisconnect:
        pass
    finally:
        try:
            await websocket.close()
        except RuntimeError:
            pass