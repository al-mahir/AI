"""ASR engines: the one seam between audio and the feedback half.

Two engine *shapes*, one pipeline:

* ``ChunkEngine`` — one call per finalized utterance-sized wave. The engine has no
  cross-chunk memory; ``StreamSession`` decides when an utterance ends and hands the
  complete wave over.

  - ``RealMuaalemEngine`` — GPU path: the Muaalem multi-level CTC model, trained on
    ≤20 s waqf segments, decoded reference-free (see transcribe.py).
  - ``MockEngine`` — no model at all. Fabricates what a perfect reciter would have
    produced from the session cursor.

* ``StreamingEngine`` — raw audio frames arrive continuously; the engine decodes
  incrementally and exposes partial results. ``StreamSession``'s waqf boundary is a
  *commit signal* ("finalize the hypothesis, start fresh") rather than the trigger
  that starts decoding.

  - ``ZipformerAsrEngine`` — streaming phoneme-CTC (Muno459/zipformer_p-quran via
    sherpa-onnx), CPU-only. One persistent sherpa_onnx stream per engine instance,
    fed every incoming frame, decoded incrementally. No per-character confidence or
    sifat detection (see class docstring).

``LiveSession`` (session.py) detects which shape the engine has and routes audio
accordingly — raw frames to ``feed()`` on every call for StreamingEngine, finalized
chunks to ``transcribe_chunk()`` for ChunkEngine. The feedback pipeline downstream
is identical in both cases: it receives a ``ChunkTranscript``.

Engines are built once at startup (see main.py's build_engines) and selected
PER SESSION from the ``start`` message's ``engine`` field (api/ws.py) — falling back
to Settings.resolved_asr_engine's choice for an unknown/omitted name.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, Union

import numpy as np
import torch

from ..config import Settings, get_settings

logger = logging.getLogger(__name__)
from quran_transcript import Aya

from ..feedback.mock import shorten_a_madd
from ..feedback.reference import build_reference
from ..feedback.track import _ordinal_of_word, _uthmani_for_ordinals
from ..feedback.types import SIFA_ATTRS, Span
from .transcribe import ChunkTranscript


@dataclass
class ChunkContext:
    """What an engine may know about a live chunk beyond the audio itself.

    The real engine ignores it entirely. The mock engine — which cannot hear — uses
    the cursor and duration to decide what the reciter "said".
    """

    duration_s: float
    cursor: Optional[Span] = None
    moshaf: object | None = None  # MoshafAttributes; typed loosely to avoid an import cycle


# ---------------------------------------------------------------------------
# Engine protocols
# ---------------------------------------------------------------------------

class ChunkEngine(Protocol):
    """One call per finalized utterance-sized wave (Muaalem, Mock).

    Stateless per call — safe to share across concurrent sessions.
    """
    def transcribe_chunk(
        self, wave: torch.FloatTensor, sample_rate: int, ctx: ChunkContext | None = None
    ) -> ChunkTranscript: ...


class StreamProcessor(Protocol):
    """Per-session streaming decode state (one per concurrent user).

    Created by a ``StreamingEngine``.  Owns the mutable decode state
    (the sherpa_onnx stream) so concurrent sessions are isolated.

    ``feed()`` is called on every incoming audio frame.  ``finalize()`` is called
    when a waqf boundary is detected, producing the committed ``ChunkTranscript``.
    ``reset()`` prepares for the next utterance within the same session.
    """
    def feed(self, samples: np.ndarray, sample_rate: int) -> str | None:
        """Accept new audio, decode available frames.

        Returns the current partial phoneme text if it changed since the last
        call, or None if unchanged (no new decode output).
        """
        ...

    def finalize(self, wave: torch.FloatTensor, sample_rate: int, ctx: ChunkContext | None = None) -> ChunkTranscript:
        """Commit the current hypothesis: pad, flush, decode remaining frames.

        Takes the exact wave of the waqf chunk (from the VAD endpointer) to guarantee
        perfect waqf boundaries without corrupting the continuous stream state.
        Returns a ``ChunkTranscript`` identical in shape to what ChunkEngine produces.
        """
        ...

    def reset(self) -> None:
        """Discard streaming state and start a fresh stream for the next utterance."""
        ...


class StreamingEngine(Protocol):
    """Shared factory: owns the heavy model weights, produces per-session handles.

    Like a connection pool or database session maker — the heavy resource (the
    neural network / ONNX recognizer) is loaded once and shared; the mutable
    per-user state (the decode stream) is isolated into lightweight handles.
    """
    def create_stream_processor(self) -> StreamProcessor:
        """Create a new per-session streaming handle."""
        ...


# Union type for anywhere that stores "an engine" generically.
AsrEngine = Union[ChunkEngine, StreamingEngine]


def is_streaming(engine: AsrEngine) -> bool:
    """Duck-type check: does this engine produce per-session streaming handles?"""
    return hasattr(engine, "create_stream_processor")



class RealMuaalemEngine:
    """The GPU path. Loads models lazily via asr.models.get_models()."""

    name = "real"

    def __init__(self) -> None:
        from .models import get_models  # heavy import kept out of mock-mode startup

        self.bundle = get_models()

    def transcribe_chunk(
        self, wave: torch.FloatTensor, sample_rate: int, ctx: ChunkContext | None = None
    ) -> ChunkTranscript:
        from .transcribe import transcribe_reference_free

        return transcribe_reference_free(self.bundle.muaalem, [wave], sample_rate)[0]


# --- Mock ---------------------------------------------------------------------


@dataclass
class _FakeUnit:
    text: str
    prob: float


class _FakeSifa:
    """Duck-types quran_muaalem.Sifa: .phonemes_group + one _FakeUnit per SIFA attr."""

    def __init__(self, phonemes_group: str, attrs: dict[str, str], prob: float):
        self.phonemes_group = phonemes_group
        for attr in SIFA_ATTRS:
            value = attrs.get(attr)
            setattr(self, attr, _FakeUnit(value, prob) if value is not None else None)


# Reciting pace used to guess how many words a mock chunk covers. Murattal with madd
# runs ~1.5–2.5 words/s; the tracker forgives over- and under-shoot anyway.
_WORDS_PER_SECOND = 2.0
_MAX_MOCK_WORDS = 28


class MockEngine:
    """Fabricates the perfect recitation continuing from the session cursor.

    ``mock_error_rate`` optionally shortens a madd in some chunks so the feedback
    colours can be demonstrated end-to-end without a model.
    """

    name = "mock"

    def __init__(self, settings: Settings | None = None):
        self.s = settings or get_settings()
        self._rng = random.Random(0)

    def transcribe_chunk(
        self, wave: torch.FloatTensor, sample_rate: int, ctx: ChunkContext | None = None
    ) -> ChunkTranscript:
        if ctx is None or ctx.cursor is None or ctx.moshaf is None:
            return ChunkTranscript(
                phonemes_text="", char_probs=[], groups=[], group_probs=[], sifat=[]
            )

        n_words = max(1, min(_MAX_MOCK_WORDS, round(ctx.duration_s * _WORDS_PER_SECOND)))
        cursor_ord = _ordinal_of_word().get(
            (ctx.cursor.sura, ctx.cursor.aya, ctx.cursor.word_idx)
        )
        if cursor_ord is None:
            return ChunkTranscript(
                phonemes_text="", char_probs=[], groups=[], group_probs=[], sifat=[]
            )

        # ponytail: the basmalah seam is the mock's one hazard. Al-Fātiḥa 1:1 IS the
        # basmalah, so a span starting there and spilling into 1:2 gets phonetized as one
        # continuous phrase — which elides the hamzat-waṣl of ٱلْحَمْد mid-phrase. The
        # pipeline then strips a *standalone* basmalah and is left with ٱلْحَمْد minus its
        # opening ءَ, one phoneme short of the phrase-initial reference, and the whole
        # group alignment shifts. A real reciter says the basmalah as its own waqf (its
        # own audio chunk), so the model never hands the feedback half this seam. The mock
        # models that by not crossing it: a chunk that opens on 1:1 stops at the basmalah.
        if (ctx.cursor.sura, ctx.cursor.aya, ctx.cursor.word_idx) == (1, 1, 0):
            n_words = min(n_words, len(Aya(1, 1).get().uthmani_words))

        uthmani = _uthmani_for_ordinals(cursor_ord, n_words)
        ref = build_reference(uthmani, ctx.moshaf)

        text = ref.phonemes
        if self.s.mock_error_rate and self._rng.random() < self.s.mock_error_rate:
            text = _inject_error(text, self._rng)

        from quran_transcript import chunck_phonemes

        groups = chunck_phonemes(text)
        prob = 0.97
        # A correct (or madd-shortened) recitation aligns group-for-group with the
        # reference, so reusing the reference sifat per group is exact, not approximate.
        sifat = [
            _FakeSifa(
                group,
                {
                    attr: getattr(ref.sifat[min(i, len(ref.sifat) - 1)], attr)
                    for attr in SIFA_ATTRS
                },
                prob,
            )
            for i, group in enumerate(groups)
        ]
        return ChunkTranscript(
            phonemes_text=text,
            char_probs=[prob] * len(text),
            groups=groups,
            group_probs=[prob] * len(groups),
            sifat=sifat,
        )


def _inject_error(text: str, rng: random.Random) -> str:
    """Corrupt one phoneme group in the MIDDLE of the chunk, for the demo colours.

    Middle, not the edge: an error on the trailing (or leading) word sits on a chunk
    boundary, and the pipeline correctly trims boundary words as unscored — so a
    trailing corruption would demonstrate nothing. A substituted consonant surfaces as
    a word-level error (red); it is the most legible mark to show the loop is live.

    ponytail: demo aid only. The real engine transcribes real audio and needs none of
    this.
    """
    from quran_transcript import chunck_phonemes

    groups = chunck_phonemes(text)
    if len(groups) < 5:
        return shorten_a_madd(text)  # too short to have a safe middle; fall back
    i = rng.randint(len(groups) // 3, max(len(groups) // 3, 2 * len(groups) // 3))
    g = groups[i]
    # Swap the base consonant for a plausible confusion (ت/ط, س/ص, …) or, failing a
    # known pair, just double a letter — either reads as a recitation slip.
    swaps = {"ت": "ط", "س": "ص", "ذ": "ز", "ك": "ق", "د": "ت", "ه": "ح"}
    base = g[0]
    groups[i] = (swaps.get(base, base) + g[1:]) if base in swaps else g + g[0]
    return "".join(groups)


def make_engine(settings: Settings | None = None) -> ChunkEngine:
    s = settings or get_settings()
    if s.resolved_asr_engine == "real":
        return RealMuaalemEngine()
    return MockEngine(s)


# --- Zipformer (StreamingEngine) -------------------------------------------

# Placeholder confidence for the optional debug `units` field only (ChunkResult.
# UnitResult.prob is non-optional). The real scoring path never sees this —
# finalize() returns char_probs=[], which feedback.confidence already
# treats as "unscored" (see its docstring: "everything stays UNSCORED (None),
# which is not the same as confident") rather than a fabricated confidence
# number. 0.5 here is deliberately neutral, not a claim about accuracy.
_ZIPFORMER_UNSCORED_PROB = 0.5


class ZipformerStreamProcessor:
    """Per-session streaming state for the Zipformer engine.
    
    Holds the mutable sherpa_onnx stream, isolating concurrent sessions.
    """
    def __init__(self, recognizer):
        self._recognizer = recognizer
        self._stream = self._recognizer.create_stream()
        self._last_partial: str = ""
        self._committed_text: str = ""

    def feed(self, samples: np.ndarray, sample_rate: int) -> str | None:
        """Push streaming float32 samples into the continuous session stream."""
        if sample_rate != 16000:
            raise ValueError(f"Zipformer requires 16kHz audio, got {sample_rate}")
        
        audio_data = np.ascontiguousarray(samples, dtype=np.float32).reshape(-1)
        if audio_data.size == 0:
            return None

        self._stream.accept_waveform(sample_rate, audio_data)
        while self._recognizer.is_ready(self._stream):
            self._recognizer.decode_stream(self._stream)

        text = self._recognizer.get_result(self._stream)
        
        if text != self._last_partial:
            self._last_partial = text
            # Return uncommitted partial text for live visual updates
            return text[len(self._committed_text):]
        return None

    def finalize(
        self,
        wave: torch.FloatTensor,
        sample_rate: int,
        ctx: ChunkContext | None = None,
        forced: bool = False,
    ) -> ChunkTranscript:
        """Commit the exact waqf audio chunk and reset the partial stream.

        Prepends 0.30s of leading silence around the VAD speech wave before decoding
        on an isolated stream, so opening phonemes (Hamzat al-Wasl, e.g. ءَ / ٱ) start
        from clean silence instead of being clipped.

        Trailing silence is ONLY added when this is a real waqf (``forced=False``):
        trailing phonemes (Madd, Waqf endings) need the lookahead to flush fully. A
        forced (hard-cap) cut is mid-utterance, not a real pause -- padding it with
        artificial trailing silence would tell the model the reciter stopped right
        when they didn't, corrupting exactly the boundary this cut already damages.
        This mirrors StreamSession._extract's own trail=0-on-forced rule; Zipformer
        shares the same StreamSession, so it hits forced cuts too and must honor it.
        """
        from quran_transcript import chunck_phonemes

        if sample_rate != 16000:
            raise ValueError(f"Zipformer requires 16kHz audio, got {sample_rate}")

        raw_pcm = wave.detach().cpu().numpy().reshape(-1).copy()
        if raw_pcm.size == 0:
            self.reset()
            return ChunkTranscript(phonemes_text="", char_probs=[], groups=[], group_probs=[], sifat=[])

        # Apply a 10ms fade-in to the VAD chunk before prepending digital silence.
        # This prevents a sharp step-function (click) at the junction between the 0s
        # and the ambient background noise. Zipformer often misinterprets this click
        # as a transient consonant (ي or ه).
        fade_len = int(0.01 * sample_rate)
        if raw_pcm.size > fade_len:
            fade = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
            raw_pcm[:fade_len] *= fade

        # Pad wave with 0.30s lead silence (300ms CTC warm-up) and 0.50s trail silence (500ms flush).
        # Digital silence padding for CTC warm-up is handled by ZipformerStreamProcessor.finalize().
        lead_padding = np.zeros(int(0.3 * sample_rate), dtype=np.float32)
        tail_padding = (
            np.zeros(0, dtype=np.float32)
            if forced
            else np.zeros(int(0.66 * sample_rate), dtype=np.float32)
        )
        full_audio = np.concatenate([lead_padding, raw_pcm, tail_padding])

        # Decode on an isolated stream
        chunk_stream = self._recognizer.create_stream()
        chunk_stream.accept_waveform(sample_rate, full_audio)
        chunk_stream.input_finished()

        while self._recognizer.is_ready(chunk_stream):
            self._recognizer.decode_stream(chunk_stream)

        text = self._recognizer.get_result(chunk_stream)

        # Reset partial stream for the next verse
        self.reset()

        print(f"[Zipformer] finalized: {text or '(empty)'}")

        if not text:
            return ChunkTranscript(phonemes_text="", char_probs=[], groups=[], group_probs=[], sifat=[])

        groups = chunck_phonemes(text)
        return ChunkTranscript(
            phonemes_text=text,
            char_probs=[],
            groups=groups,
            group_probs=[_ZIPFORMER_UNSCORED_PROB] * len(groups),
            sifat=[],
        )

    def reset(self) -> None:
        """Discard all streaming state and start a fresh stream for the next utterance."""
        self._stream = self._recognizer.create_stream()
        self._last_partial = ""
        self._committed_text = ""


class ZipformerAsrEngine:
    """Streaming phoneme-CTC (Muno459/zipformer_p-quran, via sherpa-onnx), run
    CPU-only.

    Implements ``StreamingEngine``: loaded once at startup and shared
    across sessions. Yields ``ZipformerStreamProcessor``s (which own the actual
    decode stream) to isolate concurrent users.

    Also retains a ``transcribe_chunk()`` convenience wrapper (create_stream_processor +
    feed + finalize) for backward compatibility with direct-drive tests that
    don't go through the streaming LiveSession path.

    Two things this engine genuinely cannot provide, left honest rather than
    faked:
      - No per-character CTC confidence extracted from this decode path
        (unlike Muaalem's softmax probs) -> char_probs is always [].
      - No tajweed/sifat detection at all -> every sifat attribute is None,
        via the same _FakeSifa shim MockEngine already uses for its own
        different reason (no model at all vs. this: a model with no sifat
        head).

    tokens.txt MUST be the one shipped in the zipformer_p-quran HF repo, not
    one rebuilt from phoneme_units.json — that mapping has blank at the wrong
    id. See the Quran Companion API backend's ai/asr_streaming_session.py for
    the full story; not repeated here since this is a different repo.
    """

    name = "zipformer"

    def __init__(self, model_path: str | None = None, tokens_path: str | None = None, settings: Settings | None = None):
        import sherpa_onnx

        s = settings or get_settings()
        self._recognizer = sherpa_onnx.OnlineRecognizer.from_zipformer2_ctc(
            tokens=tokens_path or s.zipformer_tokens_path,
            model=model_path or s.zipformer_model_path,
            num_threads=2,
            sample_rate=16000,
            feature_dim=80,
            decoding_method="greedy_search",
        )

    # -- StreamingEngine interface ------------------------------------

    def create_stream_processor(self) -> ZipformerStreamProcessor:
        return ZipformerStreamProcessor(self._recognizer)

    # -- Backward-compat convenience wrapper ---------------------------------

    def transcribe_chunk(
        self, wave: torch.FloatTensor, sample_rate: int, ctx: ChunkContext | None = None
    ) -> ChunkTranscript:
        """One-shot batch decode: feed + finalize + reset.

        Kept for backward compatibility with direct-drive tests that bypass
        the streaming LiveSession path. Equivalent to the old implementation.
        """
        samples = wave.detach().to(torch.float32).cpu().numpy()
        stream = self.create_stream_processor()
        stream.feed(samples, sample_rate)
        transcript = stream.finalize(wave, sample_rate, ctx)
        return transcript


# Engines that are cheap enough to always build eagerly at startup alongside
# whatever Settings.resolved_asr_engine picks (main.py's lifespan calls this
# in a thread, same eager-build-to-avoid-a-GPU-race reasoning as before).
# "zipformer" is CPU/ONNX and loads in well under a second, so it's always
# included — a user can switch to it per-session even when the server's
# default engine is "real"/"mock".  If the model files aren't present, skip
# it gracefully rather than crashing the whole startup.
def build_engines(settings: Settings | None = None) -> dict[str, AsrEngine]:
    s = settings or get_settings()
    engines: dict[str, AsrEngine] = {}
    try:
        engines["zipformer"] = ZipformerAsrEngine(settings=s)
    except (AssertionError, FileNotFoundError, OSError) as exc:
        import logging
        logging.getLogger(__name__).warning("Zipformer engine unavailable (model files missing?): %s", exc)
    default_name = s.resolved_asr_engine
    if default_name not in engines:
        engines[default_name] = make_engine(s)
    return engines

