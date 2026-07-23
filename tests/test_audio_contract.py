"""
Audio contract tests: mono conversion, resampling, and edge cases
(very short / silent clips) for src/vad.py and src/classifier.py.

These are self-contained — they synthesize their own .wav fixtures rather
than depending on data/ being populated, so they run in any environment.
"""
import numpy as np
import soundfile as sf
import pytest

from src.vad import read_wav
from src.classifier import extract_features, extract_features_from_array


def _write_wav(path, audio, sr):
    sf.write(str(path), audio, sr)


def test_read_wav_converts_stereo_to_mono(tmp_path):
    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    left = 0.5 * np.sin(2 * np.pi * 440 * t)
    right = 0.5 * np.sin(2 * np.pi * 220 * t)
    stereo = np.stack([left, right], axis=1)
    wav_path = tmp_path / "stereo.wav"
    _write_wav(wav_path, stereo, sr)

    pcm_bytes, out_sr = read_wav(str(wav_path))

    assert out_sr == sr
    decoded = np.frombuffer(pcm_bytes, dtype=np.int16)
    expected_len = len(left)
    assert len(decoded) == expected_len, "mono output should have one sample per input frame, not two"


def test_read_wav_resamples_to_16khz(tmp_path):
    sr = 8000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    wav_path = tmp_path / "8k.wav"
    _write_wav(wav_path, audio, sr)

    pcm_bytes, out_sr = read_wav(str(wav_path))

    assert out_sr == 16000
    decoded = np.frombuffer(pcm_bytes, dtype=np.int16)
    # resampled length should be roughly double the original (8kHz -> 16kHz)
    assert abs(len(decoded) - 2 * len(audio)) < 50


def test_extract_features_handles_very_short_clip(tmp_path):
    sr = 16000
    # 0.02s — well under the 0.1s padding threshold in extract_features
    audio = 0.1 * np.random.randn(int(sr * 0.02)).astype(np.float32)
    wav_path = tmp_path / "short.wav"
    _write_wav(wav_path, audio, sr)

    features = extract_features(str(wav_path))

    assert features.shape == (106,)
    assert np.isfinite(features).all(), "short-clip padding should not introduce NaN/Inf"


def test_extract_features_handles_silence(tmp_path):
    sr = 16000
    audio = np.zeros(sr, dtype=np.float32)  # 1s of pure digital silence
    wav_path = tmp_path / "silence.wav"
    _write_wav(wav_path, audio, sr)

    features = extract_features(str(wav_path))

    assert features.shape == (106,)
    assert np.isfinite(features).all(), "all-zero input should not produce NaN/Inf (e.g. via divide-by-zero in ratios)"


def test_extract_features_from_array_matches_shape():
    sr = 16000
    audio = 0.3 * np.random.randn(sr).astype(np.float32)
    features = extract_features_from_array(audio, sr)
    assert features.shape == (106,)
    assert np.isfinite(features).all()
