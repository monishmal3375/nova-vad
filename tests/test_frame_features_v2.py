"""
Regression + first-principles sanity tests for the v2 noise-robust features.
These use synthetic signals (pure tones, white noise), not project audio —
the point is verifying the features behave the way acoustic theory predicts,
independent of anything in data/scenes/.
"""
import numpy as np

from scripts.frame_features_v2 import (
    extract_frame_features_v2, FEATURE_COUNT, _periodicity_features,
)


def test_feature_count_matches_constant():
    sr = 16000
    window = 0.3 * np.random.randn(int(0.32 * sr)).astype(np.float32)
    features = extract_frame_features_v2(window, sr)
    assert features.shape == (FEATURE_COUNT,)


def test_no_nan_or_inf_on_silence():
    sr = 16000
    window = np.zeros(int(0.32 * sr), dtype=np.float32)
    features = extract_frame_features_v2(window, sr)
    assert np.isfinite(features).all()


def test_pure_tone_shows_high_periodicity():
    sr = 16000
    t = np.arange(int(0.32 * sr)) / sr
    tone = (0.5 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)
    periodicity, f0 = _periodicity_features(tone, sr)
    assert periodicity > 0.9
    assert abs(f0 - 150) < 5


def test_white_noise_shows_low_periodicity():
    sr = 16000
    rng = np.random.RandomState(0)
    noise = (0.3 * rng.randn(int(0.32 * sr))).astype(np.float32)
    periodicity, f0 = _periodicity_features(noise, sr)
    assert periodicity < 0.2
    assert f0 == 0.0


def test_periodicity_survives_noise_better_than_disappearing():
    """A noisy tone should retain meaningfully more periodicity than pure noise —
    this is the entire premise for why this feature should help under the SNR
    conditions where energy-based features degrade."""
    sr = 16000
    t = np.arange(int(0.32 * sr)) / sr
    rng = np.random.RandomState(1)
    tone = (0.5 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)
    noise = (0.4 * rng.randn(len(t))).astype(np.float32)
    noisy_tone_periodicity, f0 = _periodicity_features(tone + noise, sr)
    noise_only_periodicity, _ = _periodicity_features(noise, sr)
    assert noisy_tone_periodicity > noise_only_periodicity + 0.2
    assert abs(f0 - 150) < 10
