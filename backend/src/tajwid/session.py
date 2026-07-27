"""The live-session orchestrator: where the two halves of the pipeline meet.
 
Audio frames come in; per-word feedback events go out. Two engine shapes, one
pipeline:
 
**ChunkEngine (Muaalem, Mock)** — finalized-chunk-only flow:
 
    StreamSession (silero endpointing)          [asr half]
      -> engine.transcribe_chunk                [asr half: real GPU model or mock]
      -> transcript_to_output                   [the in-process adapter]
      -> analyse_session                        [feedback half: track/locate, diff, sifat, score]
      -> one JSON-able event
 
**StreamingEngine (Zipformer)** — continuous-feed flow:
 
    every feed() call:
      raw frames -> engine.feed()               [partial decode, no waqf gate]
      -> {"type": "partial", "phonemes": "..."}  [emitted if text changed]
 
    on waqf boundary (StreamSession finalization):
      engine.finalize(fin.wave, forced=fin.forced) [commit the hypothesis on an
                                                     isolated decode of the exact
                                                     waqf-bounded wave; resets its
                                                     own partial stream internally]
      -> transcript_to_output + analyse_session  [same feedback pipeline]
 
The adapter is the load-bearing piece: the old HTTP handoff dropped per-character
probs and sifat, killing confidence grading and sifat feedback silently. Here the
model output crosses the boundary whole.
 
Each LiveSession owns a fresh silero instance (stateful RNN — sharing one across
concurrent sessions would mix their hidden states; the pre-merge server did share it).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

import numpy as np

from quran_transcript import MoshafAttributes

from .asr.contract import ChunkResult
from .asr.engine import AsrEngine, ChunkContext, is_streaming
from .asr.stream import FinalizedChunk, StreamSession
from .asr.transcribe import ChunkTranscript
from .asr.vad import load_vad
from .config import Settings, get_settings
from .feedback.confidence import STRICTNESS
from .feedback.pipeline import analyse_session
from .feedback.session import SessionState
from .feedback.types import (
    SIFA_ATTRS,
    MuaalemOutput,
    Phonemes,
    PredictedSifa,
    Span,
)


def transcript_to_output(t: ChunkTranscript) -> MuaalemOutput:
    """ChunkTranscript -> the feedback half's input type. A shape change, not a
    conversion: text + per-CHAR probs + all 10 sifat attrs with their own probs."""
    sifat: list[PredictedSifa] = []
    for s in t.sifat:
        attrs: dict[str, str] = {}
        probs: dict[str, float] = {}
        for attr in SIFA_ATTRS:
            unit = getattr(s, attr, None)
            if unit is not None and unit.text is not None:
                attrs[attr] = unit.text
                probs[attr] = float(unit.prob)
        sifat.append(
            PredictedSifa(phonemes_group=s.phonemes_group, attrs=attrs, probs=probs)
        )
    return MuaalemOutput(
        phonemes=Phonemes(text=t.phonemes_text, probs=t.char_probs or None),
        sifat=sifat,
    )


def default_moshaf(settings: Settings | None = None) -> MoshafAttributes:
    s = settings or get_settings()
    return MoshafAttributes(
        rewaya="hafs",
        madd_monfasel_len=s.madd_monfasel_len,
        madd_mottasel_len=s.madd_mottasel_len,
        madd_mottasel_waqf=s.madd_mottasel_waqf,
        madd_aared_len=s.madd_aared_len,
    )


def resolve_moshaf(raw: dict | None, settings: Settings | None = None) -> MoshafAttributes:
    """The client's moshaf choice, filled in over the default rather than validated bare.

    ``/moshaf-schema`` (api/rest.py) only shows fields with 2+ options -- nothing to pick
    when there's only one -- so a field like ``rewaya`` (Literal["hafs"], fixed) never
    appears in what the frontend sends. But ``rewaya`` has no default on MoshafAttributes,
    so validating the client's dict AS THE WHOLE MODEL always raised, and used to fall
    back to `default_moshaf()` outright -- discarding every field the reciter DID set,
    including the one this bug was reported over (madd_monfasel_len). Layering the
    client's dict over the resolved default's dump means an omitted-because-hidden field
    resolves to its default while a provided field still overrides it.

    A genuinely invalid combination (madd al-leen longer than madd al-aared, an
    out-of-range value) still raises after the merge and still falls back to the
    default -- that's the real "don't let a bad config kill the session" case.
    """
    s = settings or get_settings()
    if not raw:
        return default_moshaf(s)
    try:
        return MoshafAttributes(**{**default_moshaf(s).model_dump(), **raw})
    except Exception:  # noqa: BLE001 — any bad-config shape, not just ValidationError
        return default_moshaf(s)


def resolve_strictness(raw: str | None, settings: Settings | None = None) -> str:
    """The client's strictness, or the default — never an unusable value.

    `STRICTNESS[strictness]` is read once per finalized chunk, deep inside the feedback
    pipeline, long after the start message was accepted. An unrecognised value therefore
    did not fail at the boundary where it arrived: it raised KeyError mid-recitation,
    the exception escaped the WebSocket handler, and the socket closed with code 1000 —
    "OK". The reciter's feedback simply stopped, and the client could not tell that from
    a normal end of session. `"Normal"` with a capital N was enough to do it.

    Falling back mirrors what this boundary already does for `engine` (unbuilt name ->
    server default) and `moshaf` (invalid combination -> default). A bad setting costs
    you the setting, not the session.
    """
    s = settings or get_settings()
    if raw in STRICTNESS:
        return raw
    return s.strictness if s.strictness in STRICTNESS else "normal"


class LiveSession:
    """One reciter's live stream: endpointing + ASR + tracking feedback.

    Detects the engine shape at init and routes audio accordingly:
    - ChunkEngine: audio flows only through StreamSession; on finalization, the
      complete wave is handed to ``transcribe_chunk()`` — same as always.
    - StreamingEngine: creates a per-session stream processor. Raw frames are
      pushed to ``stream_processor.feed()`` on every ``feed()`` call (partial
      results emitted immediately), and ``stream_processor.finalize()`` is called
      on waqf boundaries instead of ``transcribe_chunk()``.
    """

    def __init__(
        self,
        engine: AsrEngine,
        *,
        session_id: str,
        moshaf: MoshafAttributes | None = None,
        start: Optional[Span] = None,
        strictness: str | None = None,
        include_units: bool = False,
        rules: frozenset[str] | None = None,
        settings: Settings | None = None,
    ):
        self.s = settings or get_settings()
        self.engine = engine
        self._is_streaming = is_streaming(engine)
        self.stream_processor = engine.create_stream_processor() if self._is_streaming else None
        
        # Zipformer is streaming, Muaalem/Mock are chunk.
        max_chunk_s = self.s.max_chunk_s_zipformer if self._is_streaming else self.s.max_chunk_s_muaalem
        self.stream = StreamSession(load_vad(), self.s, max_chunk_samples=int(max_chunk_s * self.s.sample_rate))
        
        self.state = SessionState(
            moshaf=moshaf or default_moshaf(self.s),
            session_id=session_id,
            cursor=start,
            strictness=resolve_strictness(strictness, self.s),
            rules=rules,
        )
        self.include_units = include_units
        self.seq = 0

    # -- public API -------------------------------------------------------
    def feed(self, samples: np.ndarray) -> list[dict]:
        """Append audio samples; return events (partial + feedback)."""
        events: list[dict] = []

        # StreamingEngine: push raw frames for incremental decode.
        if self._is_streaming:
            partial = self.stream_processor.feed(samples, self.s.sample_rate)  # type: ignore[union-attr]
            if partial is not None:
                from .asr.transcribe import ChunkTranscript
                from .feedback.pipeline import analyse_session

                transcript = ChunkTranscript(
                    phonemes_text=partial, char_probs=[], groups=[], group_probs=[], sifat=[]
                )
                output = transcript_to_output(transcript)
                # NOTE: We intentionally do NOT commit `next_state` back to `self.state`
                # here. Partials are speculative — the frontend uses the cursor hint to
                # paint words, but the backend's authoritative state only advances on
                # finalized waqf chunks (see _process_finalized). This means the backend
                # and frontend cursors temporarily diverge during streaming, which is
                # correct: the finalize will re-derive from the backend's real state.
                feedback, next_state = analyse_session(output, self.state, forced_cut=True)

                event: dict = {
                    "type": "partial",
                    "phonemes": partial,
                    "feedback": feedback.model_dump(),
                }
                if next_state.cursor:
                    event["cursor"] = next_state.cursor.model_dump()
                
                events.append(event)

        # VAD endpointing (both engine shapes need this).
        for fin in self.stream.feed(samples):
            event = self._process_finalized(fin)
            if event is not None:
                events.append(event)

        return events

    def flush(self) -> list[dict]:
        """End of stream: finalize and score any in-progress utterance."""
        events: list[dict] = []

        # If streaming engine has accumulated audio that StreamSession hasn't
        # finalized yet (e.g. user stopped mid-word without a waqf), finalize
        # the streaming hypothesis too — StreamSession.flush() will produce a
        # FinalizedChunk for it, and _process_finalized will call
        # engine.finalize() on that boundary.
        for fin in self.stream.flush():
            event = self._process_finalized(fin)
            if event is not None:
                events.append(event)

        return events

    def seek(self, span: Span) -> None:
        """The user repositioned (picked a different sura/aya). Reset the cursor."""
        self.state = replace(self.state, cursor=span, penalty=0)
        # Discard any partial streaming state — the user jumped, so whatever
        # the engine had accumulated is for the wrong passage.
        if self._is_streaming:
            self.stream_processor.reset()  # type: ignore[union-attr]

    @property
    def cursor(self) -> Optional[Span]:
        return self.state.cursor

    # -- internals --------------------------------------------------------
    def _process_finalized(self, fin: FinalizedChunk) -> dict | None:
        """Process a finalized VAD chunk through the feedback pipeline.

        For ChunkEngine: transcribe the finalized wave.
        For StreamingEngine: finalize the streaming hypothesis (the audio was
        already fed incrementally), then reset for the next utterance.
        """
        sr = self.s.sample_rate
        ctx = ChunkContext(
            duration_s=(fin.end_sample - fin.start_sample) / sr,
            cursor=self.state.cursor,
            moshaf=self.state.moshaf,
        )
        if self._is_streaming:
            transcript = self.stream_processor.finalize(fin.wave, sr, ctx, forced=fin.forced)  # type: ignore[union-attr]
        else:
            transcript = self.engine.transcribe_chunk(fin.wave, sr, ctx)  # type: ignore[union-attr]

        print(f"[LiveSession] chunk {self.seq} start={fin.start_sample} end={fin.end_sample}")
        print(f"[LiveSession] transcript={transcript.phonemes_text or '(empty)'}")
        if not transcript.phonemes_text:
            return None

        output = transcript_to_output(transcript)
        feedback, self.state = analyse_session(output, self.state, forced_cut=fin.forced)

        event: dict = {
            "type": "feedback",
            "chunk_seq": self.seq,
            "audio_span_sec": [fin.start_sample / sr, fin.end_sample / sr],
            "forced_cut": fin.forced,
            "phonemes": transcript.phonemes_text,
            "feedback": feedback.model_dump(),
            "cursor": self.state.cursor.model_dump() if self.state.cursor else None,
        }
        if self.include_units:
            event["units"] = [
                u.model_dump()
                for u in ChunkResult.from_transcript(
                    transcript,
                    session_id=self.state.session_id,
                    chunk_seq=self.seq,
                    audio_span_sec=tuple(event["audio_span_sec"]),
                ).units
            ]
        self.seq += 1
        return event
