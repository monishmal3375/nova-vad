"""
Regression test for the causal frame-feature extractor's output shape —
this project has already had two "documented count != actual count" bugs
(README said 150+ features, actual was 106), so the count is asserted here
rather than trusted to stay in sync by hand.
"""
import numpy as np

from scripts.frame_features import extract_frame_features, FEATURE_COUNT


def test_feature_count_matches_constant():
    sr = 16000
    window = 0.3 * np.random.randn(int(0.32 * sr)).astype(np.float32)
    features = extract_frame_features(window, sr)
    assert features.shape == (FEATURE_COUNT,)


def test_no_nan_or_inf_on_silence():
    sr = 16000
    window = np.zeros(int(0.32 * sr), dtype=np.float32)
    features = extract_frame_features(window, sr)
    assert np.isfinite(features).all()


def test_no_nan_or_inf_on_very_short_window():
    sr = 16000
    window = 0.1 * np.random.randn(int(0.02 * sr)).astype(np.float32)
    features = extract_frame_features(window, sr)
    assert features.shape == (FEATURE_COUNT,)
    assert np.isfinite(features).all()
