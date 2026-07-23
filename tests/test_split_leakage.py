"""
Split-leakage tests for src/benchmark.py's split_dataset(): no file should
ever appear in both the train and test partitions, and the seeded split
must be deterministic (protects against someone silently removing the
random.seed(42) call and making the reported benchmark unreproducible).
"""
import os
import pytest

from src.benchmark import split_dataset

SPEECH_DIR = "data/speech"
NOISE_DIR = "data/noise"

_have_data = os.path.isdir(SPEECH_DIR) and os.path.isdir(NOISE_DIR) and any(
    f.endswith(".wav") for f in os.listdir(SPEECH_DIR)
)

pytestmark = pytest.mark.skipif(
    not _have_data,
    reason="data/speech or data/noise is empty or missing — run download_data.py first",
)


def test_train_test_do_not_overlap():
    train_speech, train_noise, test_speech, test_noise = split_dataset(SPEECH_DIR, NOISE_DIR)

    assert set(train_speech).isdisjoint(test_speech), "a speech file appears in both train and test"
    assert set(train_noise).isdisjoint(test_noise), "a noise file appears in both train and test"


def test_split_covers_all_files_exactly_once():
    train_speech, train_noise, test_speech, test_noise = split_dataset(SPEECH_DIR, NOISE_DIR)
    all_speech = sorted(f for f in os.listdir(SPEECH_DIR) if f.endswith(".wav"))
    all_noise = sorted(f for f in os.listdir(NOISE_DIR) if f.endswith(".wav"))

    assert sorted(train_speech + test_speech) == all_speech
    assert sorted(train_noise + test_noise) == all_noise


def test_split_is_deterministic_across_calls():
    result_a = split_dataset(SPEECH_DIR, NOISE_DIR)
    result_b = split_dataset(SPEECH_DIR, NOISE_DIR)

    assert result_a == result_b, (
        "split_dataset() must be deterministic (random.seed(42)) — "
        "otherwise the reported benchmark numbers aren't reproducible"
    )
