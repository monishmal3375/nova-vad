"""
Regression test for codec degradation — this project already had one
subtle bug caught only by manual inspection (Opus decodes at 48kHz
internally regardless of requested rate; an earlier version silently
mis-length- and mis-aligned the output). These tests would have caught it.
"""
import numpy as np

from scripts.codec_degrade import apply_g711, apply_opus


def _test_signal(sr=16000, duration_s=1.0):
    t = np.arange(int(sr * duration_s)) / sr
    return (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def test_g711_alaw_preserves_length_and_alignment():
    audio = _test_signal()
    degraded = apply_g711(audio, "alaw")
    assert len(degraded) == len(audio)
    corr = np.correlate(degraded, audio, mode="full")
    best_lag = np.argmax(corr) - (len(audio) - 1)
    assert best_lag == 0


def test_g711_mulaw_preserves_length_and_alignment():
    audio = _test_signal()
    degraded = apply_g711(audio, "mulaw")
    assert len(degraded) == len(audio)
    corr = np.correlate(degraded, audio, mode="full")
    best_lag = np.argmax(corr) - (len(audio) - 1)
    assert best_lag == 0


def test_g711_actually_degrades_not_passthrough():
    audio = _test_signal()
    degraded = apply_g711(audio, "alaw")
    diff_rms = np.sqrt(np.mean((audio - degraded) ** 2))
    assert diff_rms > 1e-4  # real degradation, not a silent no-op


def test_opus_preserves_length_and_alignment():
    audio = _test_signal(duration_s=2.0)
    degraded = apply_opus(audio, bitrate=24000)
    assert len(degraded) == len(audio)
    corr = np.correlate(degraded[:16000], audio[:16000], mode="full")
    best_lag = np.argmax(corr) - (16000 - 1)
    assert abs(best_lag) <= 5  # allow a few samples of resampling-filter slop


def test_opus_actually_degrades_not_passthrough():
    audio = _test_signal(duration_s=2.0)
    degraded = apply_opus(audio, bitrate=24000)
    diff_rms = np.sqrt(np.mean((audio - degraded) ** 2))
    assert diff_rms > 1e-4
