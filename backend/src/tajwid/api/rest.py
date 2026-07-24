"""REST endpoints: health, offline file transcription, and cold analysis."""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

from typing import get_args

from fastapi import APIRouter, File, Request, UploadFile

from ..config import get_settings

router = APIRouter()


def _literal_values(annotation) -> list:
    """The allowed values of a ``Literal`` (or ``Optional[Literal]``) field, in order."""
    values: list = []
    for arg in get_args(annotation):
        if arg is type(None):
            continue
        inner = get_args(arg)  # a nested Literal inside Optional[...]
        values.extend(inner if inner else [arg])
    return values


@router.get("/moshaf-schema")
def moshaf_schema() -> dict:
    """The recitation (moshaf) attributes the reciter can set, for the settings panel.

    Introspected from ``MoshafAttributes`` so the panel is generated from the one source
    of truth: each field's Arabic name, its allowed values with Arabic labels, and a
    sensible starting value. Sending the whole set keeps required fields satisfied.
    """
    from pydantic_core import PydanticUndefined
    from quran_transcript import MoshafAttributes

    fields = []
    for name, f in MoshafAttributes.model_fields.items():
        values = _literal_values(f.annotation)
        if len(values) < 2:
            continue  # nothing to choose (e.g. rewaya is fixed to hafs)
        extra = f.json_schema_extra or {}
        amap = extra.get("field_arabic_attrs_map") or {}
        default = f.default if f.default not in (None, PydanticUndefined) else values[0]
        fields.append(
            {
                "key": name,
                "name_ar": extra.get("field_arabic_name", name),
                "description": f.description,
                "default": default,
                "options": [
                    {"value": v, "label": amap.get(str(v), amap.get(v, str(v)))}
                    for v in values
                ],
            }
        )
    return {"fields": fields}


@router.get("/health")
def health(request: Request) -> dict:
    s = get_settings()
    engines = request.app.state.engines
    default_name = request.app.state.default_engine_name
    engine = engines[default_name]
    info = {
        "status": "healthy",
        "engine": getattr(engine, "name", "unknown"),
        "available_engines": sorted(engines.keys()),
        "device": s.device,
        "dtype": s.dtype_str,
        "muaalem_model": s.muaalem_model_id,
        "segmenter_model": s.segmenter_model_id,
    }
    if getattr(engine, "name", "") == "real":
        bundle = engine.bundle
        info["muaalem_device"] = str(bundle.muaalem.device)
        info["segmenter_device"] = str(bundle.segmenter_device)
    return info


@router.post("/transcribe-file")
async def transcribe_file_endpoint(file: UploadFile = File(...)) -> dict:
    """Batch: transcribe a complete uploaded recitation (any ffmpeg-decodable format).

    Real-engine only — the W2V-BERT segmenter is the chunker here.
    """
    from ..asr.batch import transcribe_file

    audio_bytes = await file.read()
    filename = file.filename or "upload"

    def _run():
        suffix = Path(filename).suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        try:
            return transcribe_file(tmp_path, session_id=filename)
        finally:
            os.unlink(tmp_path)

    results = await asyncio.to_thread(_run)
    return {"chunks": [r.model_dump() for r in results]}
