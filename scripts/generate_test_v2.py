"""
Builds the EXPANDED, more noise-diverse locked test set ('test_v2'), fixing
the statistical-power problem the round-1 LONO analysis quantified: the
original 40-scene test set only had 8 unique noise files per SNR condition,
letting a single file swing a condition's accuracy by more than the actual
model differences being measured.

Unlike the original _generate_split() (which samples noise files WITH
replacement, causing collisions down to just 8 unique files out of 10
draws), this samples WITHOUT replacement within each condition — shuffle
the available noise pool once per condition, then take scenes sequentially,
cycling back to a re-shuffled pool only if more scenes are requested than
there are unique files. This directly maximizes unique-file coverage
instead of leaving it to chance.

Drawn EXCLUSIVELY from the test-only noise/speech pool (never train-side)
— the same split_dataset() 80/20 division already used everywhere else in
this project, so test_v2 has the same leakage guarantees as the original
test set.

The original data/scenes/test/ (40 scenes) is left completely untouched —
kept as the historical record the LONO analysis was run against.

Run: python3 -m scripts.generate_test_v2
"""
import json
import os
import random

from scripts.generate_scenes import (
    SPEECH_DIR, NOISE_DIR, OUT_DIR, SNR_CONDITIONS_DB, _make_scene, _frame_labels_10ms,
)
from src.benchmark import split_dataset

TEST_V2_SEED = 45  # different from test's 42, val's 43, train2's 44
SCENES_PER_CONDITION_TEST_V2 = 25  # x4 conditions = 100 scenes (vs original 40)


def _generate_split_diverse_noise(speech_files, noise_files, split_name, n_per_condition, rng):
    """Same scene-building logic as generate_scenes.py's _generate_split, but
    samples noise files WITHOUT replacement within each condition (cycling
    through a reshuffled pool only if n_per_condition exceeds the pool size)."""
    os.makedirs(os.path.join(OUT_DIR, split_name), exist_ok=True)
    manifest = []
    scene_idx = 0

    conditions = [("clean", None)] + [(f"snr_{db}db", db) for db in SNR_CONDITIONS_DB]

    for condition_name, snr_db in conditions:
        pool = list(noise_files)
        rng.shuffle(pool)
        pool_iter = iter(pool)

        for _ in range(n_per_condition):
            try:
                noise_file = next(pool_iter)
            except StopIteration:
                pool = list(noise_files)
                rng.shuffle(pool)
                pool_iter = iter(pool)
                noise_file = next(pool_iter)

            audio, speech_intervals_ms, chosen_speech = _make_scene(
                speech_files, noise_file, snr_db, rng
            )
            scene_id = f"{split_name}_scene_{scene_idx:04d}_{condition_name}"
            import soundfile as sf
            wav_path = os.path.join(OUT_DIR, split_name, f"{scene_id}.wav")
            sf.write(wav_path, audio, 16000, subtype="PCM_16")

            duration_ms = round(len(audio) / 16000 * 1000)
            frame_labels = _frame_labels_10ms(duration_ms, speech_intervals_ms)

            meta = {
                "scene_id": scene_id,
                "split": split_name,
                "condition": condition_name,
                "snr_db": snr_db,
                "duration_ms": duration_ms,
                "sample_rate": 16000,
                "speech_intervals_ms": speech_intervals_ms,
                "frame_labels_10ms": frame_labels,
                "source_speech_files": chosen_speech,
                "source_noise_file": noise_file,
            }
            with open(os.path.join(OUT_DIR, split_name, f"{scene_id}.json"), "w") as f:
                json.dump(meta, f, indent=2)

            manifest.append(meta)
            scene_idx += 1

    return manifest


def generate_test_v2():
    rng = random.Random(TEST_V2_SEED)
    train_speech, train_noise, test_speech, test_noise = split_dataset(SPEECH_DIR, NOISE_DIR)

    print(f"Test-only pool: {len(test_speech)} speech files, {len(test_noise)} noise files")
    print("Generating TEST_V2 scenes (expanded noise diversity, seed=45, "
          "test-only pool, without-replacement noise sampling)...")
    manifest = _generate_split_diverse_noise(
        test_speech, test_noise, "test_v2", SCENES_PER_CONDITION_TEST_V2, rng
    )
    print(f"  {len(manifest)} test_v2 scenes written to {OUT_DIR}/test_v2/")

    with open(os.path.join(OUT_DIR, "test_v2_manifest.json"), "w") as f:
        json.dump({"test_v2": manifest}, f, indent=2)

    # report unique noise file count per condition, to confirm the fix worked
    from collections import defaultdict
    by_cond = defaultdict(set)
    for m in manifest:
        by_cond[m["condition"]].add(m["source_noise_file"])
    print("\nUnique noise files per condition (test_v2):")
    for cond, files in by_cond.items():
        print(f"  {cond}: {len(files)} unique files")


if __name__ == "__main__":
    generate_test_v2()
