"""
Additional training scenes for the v2 noise-robustness iteration ('train2'),
generated on top of (not replacing) the original data/scenes/train/ (100
scenes) so v1's training data stays exactly reproducible. v2's training data
is train + train2 combined: 100 + 300 = 400 scenes.

Same 4 conditions as eval (clean, 10dB, 0dB, -5dB), same train-only noise
pool as everything else in this project's train split — confirmed by
construction, since _generate_split only ever receives train_noise (never
test_noise). This is 3x the per-condition scene count of the original train
split (25 -> 75 per condition), addressing "more noisy training volume" per
the noise-robustness plan, NOT a class-imbalance fix — the audit
(reports/decision_v3.md) found the original composition was already
balanced (~24% speech in every condition).

Run: python3 -m scripts.generate_train2_split
"""
import json
import os
import random

from scripts.generate_scenes import SPEECH_DIR, NOISE_DIR, OUT_DIR, _generate_split
from src.benchmark import split_dataset

TRAIN2_SEED = 44  # different from train's 42 and val's 43
SCENES_PER_CONDITION_TRAIN2 = 75  # x4 conditions = 300 additional scenes


def generate_train2():
    rng = random.Random(TRAIN2_SEED)
    train_speech, train_noise, test_speech, test_noise = split_dataset(SPEECH_DIR, NOISE_DIR)

    print("Generating TRAIN2 scenes (additional training volume, seed=44, "
          "train-only noise pool, same conditions as eval)...")
    train2_manifest = _generate_split(train_speech, train_noise, "train2", SCENES_PER_CONDITION_TRAIN2, rng)
    print(f"  {len(train2_manifest)} train2 scenes written to {OUT_DIR}/train2/")

    with open(os.path.join(OUT_DIR, "train2_manifest.json"), "w") as f:
        json.dump({"train2": train2_manifest}, f, indent=2)


if __name__ == "__main__":
    generate_train2()
