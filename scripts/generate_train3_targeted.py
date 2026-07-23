"""
Targeted additional training scenes for the SNR bands the now-trustworthy
test_v2 benchmark shows are genuinely weak: 0dB and -5dB specifically
(clean and 10dB are already competitive with or better than Silero/Pyannote
on test_v2 — see reports/decision_v4.md). Clean/10dB are deliberately
skipped here rather than padded further, since Part 1's fix showed they
were not the real gap.

Train-only noise pool (same split_dataset() division as everything else),
seed=46 (different from train=42, val=43, train2=44, test_v2=45).

Run: python3 -m scripts.generate_train3_targeted
"""
import json
import os
import random

from scripts.generate_scenes import SPEECH_DIR, NOISE_DIR, OUT_DIR, _generate_split
from src.benchmark import split_dataset

TRAIN3_SEED = 46
SCENES_PER_TARGETED_CONDITION = 75  # x2 conditions (0dB, -5dB only) = 150 scenes


def _generate_targeted(speech_files, noise_files, split_name, n_per_condition, rng):
    """Same as _generate_split but only for snr_0db and snr_-5db conditions."""
    os.makedirs(os.path.join(OUT_DIR, split_name), exist_ok=True)
    manifest = []
    scene_idx = 0

    from scripts.generate_scenes import _make_scene, _frame_labels_10ms
    import soundfile as sf

    conditions = [("snr_0db", 0), ("snr_-5db", -5)]

    for condition_name, snr_db in conditions:
        for _ in range(n_per_condition):
            noise_file = noise_files[rng.randrange(len(noise_files))]
            audio, speech_intervals_ms, chosen_speech = _make_scene(
                speech_files, noise_file, snr_db, rng
            )
            scene_id = f"{split_name}_scene_{scene_idx:04d}_{condition_name}"
            wav_path = os.path.join(OUT_DIR, split_name, f"{scene_id}.wav")
            sf.write(wav_path, audio, 16000, subtype="PCM_16")

            duration_ms = round(len(audio) / 16000 * 1000)
            frame_labels = _frame_labels_10ms(duration_ms, speech_intervals_ms)

            meta = {
                "scene_id": scene_id, "split": split_name, "condition": condition_name,
                "snr_db": snr_db, "duration_ms": duration_ms, "sample_rate": 16000,
                "speech_intervals_ms": speech_intervals_ms, "frame_labels_10ms": frame_labels,
                "source_speech_files": chosen_speech, "source_noise_file": noise_file,
            }
            with open(os.path.join(OUT_DIR, split_name, f"{scene_id}.json"), "w") as f:
                json.dump(meta, f, indent=2)
            manifest.append(meta)
            scene_idx += 1

    return manifest


def generate_train3():
    rng = random.Random(TRAIN3_SEED)
    train_speech, train_noise, test_speech, test_noise = split_dataset(SPEECH_DIR, NOISE_DIR)

    print("Generating TRAIN3 scenes (targeted: 0dB + -5dB only, seed=46, train-only noise)...")
    manifest = _generate_targeted(train_speech, train_noise, "train3", SCENES_PER_TARGETED_CONDITION, rng)
    print(f"  {len(manifest)} train3 scenes written to {OUT_DIR}/train3/")

    with open(os.path.join(OUT_DIR, "train3_manifest.json"), "w") as f:
        json.dump({"train3": manifest}, f, indent=2)


if __name__ == "__main__":
    generate_train3()
