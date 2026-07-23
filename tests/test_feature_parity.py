"""
Feature parity tests: file-based and in-memory feature extraction must
produce equivalent vectors (extract_features vs extract_features_from_array),
and neither should ever emit NaN/Inf on real project data.
"""
import os
import numpy as np
import librosa
import pytest

from src.classifier import extract_features, extract_features_from_array

SPEECH_DIR = "data/speech"
NOISE_DIR = "data/noise"

_have_data = os.path.isdir(SPEECH_DIR) and any(
    f.endswith(".wav") for f in os.listdir(SPEECH_DIR)
) if os.path.isdir(SPEECH_DIR) else False

pytestmark = pytest.mark.skipif(
    not _have_data,
    reason="data/speech is empty or missing — run download_data.py first",
)


def _sample_files(directory, n=5):
    files = sorted(f for f in os.listdir(directory) if f.endswith(".wav"))
    return [os.path.join(directory, f) for f in files[:n]]


def test_file_and_array_extraction_agree():
    for path in _sample_files(SPEECH_DIR, 3) + _sample_files(NOISE_DIR, 3):
        from_file = extract_features(path)
        audio, sr = librosa.load(path, sr=16000, mono=True)
        from_array = extract_features_from_array(audio, sr)

        assert from_file.shape == from_array.shape
        np.testing.assert_allclose(
            from_file, from_array, rtol=1e-5, atol=1e-6,
            err_msg=f"file vs in-memory feature mismatch for {path}",
        )


def test_no_nan_or_inf_on_real_data():
    for path in _sample_files(SPEECH_DIR, 10) + _sample_files(NOISE_DIR, 10):
        features = extract_features(path)
        assert np.isfinite(features).all(), f"non-finite feature value in {path}"
