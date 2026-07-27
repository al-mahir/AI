import sys
import torch
import numpy as np
from pathlib import Path

# Fix python path
sys.path.insert(0, str(Path("backend/src").resolve()))

from tajwid.asr.engine import build_engines, ChunkContext
from tajwid.asr.stream import StreamSession
from tajwid.asr.vad import load_vad
from tajwid.config import get_settings

def load_audio(path, sr):
    import soundfile as sf
    import librosa
    wave, orig_sr = sf.read(path)
    if orig_sr != sr:
        wave = librosa.resample(wave, orig=orig_sr, target=sr)
    return wave

def main():
    settings = get_settings()
    engines = build_engines(settings)
    engine = engines["zipformer"]
    
    wave = load_audio("backend/tests/assets/fatiha_long_track.wav", 16000)
    vad = load_vad()
    stream = StreamSession(vad, settings)
    
    print("=== ZIPFORMER ===")
    chunks = stream.feed(wave) + stream.flush()
    for i, fin in enumerate(chunks):
        ctx = ChunkContext(duration_s=(fin.end_sample - fin.start_sample) / 16000)
        t = engine.transcribe_chunk(fin.wave, 16000, ctx)
        print(f"chunk {i}: {t.phonemes_text}")

if __name__ == "__main__":
    main()
