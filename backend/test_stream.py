import sys; from pathlib import Path; sys.path.insert(0, str(Path("backend/src").resolve()))
import soundfile as sf; import librosa, numpy as np
from tajwid.session import LiveSession
from tajwid.asr.engine import build_engines
from tajwid.config import get_settings

settings = get_settings()
engine = build_engines(settings)["zipformer"]
session = LiveSession(engine, session_id="test")

wave, sr = sf.read("tests/assets/fatiha_long_track.wav")
if sr != 16000: wave = librosa.resample(wave, orig_sr=sr, target_sr=16000)

print("=== ZIPFORMER STREAMING ===")
chunk_size = int(16000 * 0.1)
for i in range(0, len(wave), chunk_size):
    samples = wave[i:i+chunk_size]
    events = session.feed(samples)
    for e in events:
        if e["type"] == "feedback":
            print(f"Feedback: {e['phonemes']}")
