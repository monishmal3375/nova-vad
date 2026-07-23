"""
Generates a NEW validation split ('val') for the noise-robustness iteration,
kept strictly separate from:
  - data/scenes/dev/  — already used + reported for v1's hysteresis tuning;
    left untouched so that history stays reproducible.
  - data/scenes/test/ — the locked 40-scene held-out set; NEVER regenerated,
    NEVER used for any tuning decision in this file.

Drawn from the same train-source speech/noise files as data/scenes/train/
(via src.benchmark.split_dataset()'s existing seeded 80/20 split), using a
different rng seed so scene content doesn't duplicate the train scenes.
Same 4 conditions as test (clean, 10dB, 0dB, -5dB) so validation numbers are
comparable in kind (not value) to the eventual test numbers.

Run: python3 -m scripts.generate_val_split
"""
import json
import os
import random

from scripts.generate_scenes import (
    SPEECH_DIR, NOISE_DIR, OUT_DIR, _generate_split,
)
from src.benchmark import split_dataset

VAL_SEED = 43  # deliberately different from generate_scenes.py's SEED=42
SCENES_PER_CONDITION_VAL = 10  # x4 conditions = 40 scenes, same size as test for comparable CIs


def generate_val():
    rng = random.Random(VAL_SEED)
    train_speech, train_noise, test_speech, test_noise = split_dataset(SPEECH_DIR, NOISE_DIR)

    print("Generating VAL scenes (from train-split files, seed=43, "
          "for this iteration's threshold/hyperparameter tuning only)...")
    val_manifest = _generate_split(train_speech, train_noise, "val", SCENES_PER_CONDITION_VAL, rng)
    print(f"  {len(val_manifest)} val scenes written to {OUT_DIR}/val/")

    with open(os.path.join(OUT_DIR, "val_manifest.json"), "w") as f:
        json.dump({"val": val_manifest}, f, indent=2)


if __name__ == "__main__":
    generate_val()
