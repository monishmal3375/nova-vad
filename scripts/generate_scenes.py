"""
Builds mixed speech+noise scenes with millisecond-accurate ground truth,
addressing the plan's Section 3.2/7.1 concern: the original benchmark
compares whole files from two different source datasets (Google Speech
Commands vs. MUSAN), which lets a model learn "which dataset is this"
instead of "is speech present." Here, speech and noise live in the SAME
file, so detecting speech is the only way to do well.

Scope: Layer 1 (clean) + Layer 2 (controlled noise-level degradation) from
the plan's benchmark-layer table. Codec/RTC transmission (Layer 3) and hard
negatives beyond music (Layer 4) are explicitly out of scope for this pass
— see reports/ for what's been covered vs. deferred.

Reuses src.benchmark.split_dataset()'s existing seeded 80/20 file split, so
scenes built from the "test" files never draw on "train" files — leakage is
prevented by construction, not by a second, potentially-inconsistent split.

Run: python3 -m scripts.generate_scenes
"""
import json
import os
import random

import librosa
import numpy as np
import soundfile as sf

from src.benchmark import split_dataset

SPEECH_DIR = "data/speech"
NOISE_DIR = "data/noise"
OUT_DIR = "data/scenes"
SR = 16000
SEED = 42

# SNR conditions in dB. "clean" is handled separately (near-silent bed).
SNR_CONDITIONS_DB = [10, 0, -5]
SCENES_PER_CONDITION_TEST = 10   # x4 conditions (clean + 3 SNR) = 40 test scenes (locked)
SCENES_PER_CONDITION_DEV = 5     # x4 conditions = 20 dev scenes (threshold tuning only)
SCENES_PER_CONDITION_TRAIN = 25  # x4 conditions = 100 train scenes (frame-level model training)

SCENE_DURATION_S = 12.0
MIN_SPEECH_CLIPS_PER_SCENE = 2
MAX_SPEECH_CLIPS_PER_SCENE = 4


def _load_mono_16k(path):
    audio, sr = librosa.load(path, sr=SR, mono=True)
    return audio.astype(np.float32)


def _rms(x):
    return float(np.sqrt(np.mean(x ** 2)) + 1e-10)


def _tile_to_length(audio, length):
    if len(audio) >= length:
        return audio[:length]
    reps = int(np.ceil(length / len(audio)))
    return np.tile(audio, reps)[:length]


def _make_scene(speech_files, noise_file, snr_db, rng):
    """
    snr_db=None means "clean": the background bed is near-silent instead of
    a real noise mix, per the plan's Layer 1 definition.
    """
    n_samples = int(SCENE_DURATION_S * SR)
    noise_audio = _load_mono_16k(os.path.join(NOISE_DIR, noise_file))
    bed = _tile_to_length(noise_audio, n_samples).copy()

    if snr_db is None:
        bed *= 0.01  # near-silent background, not literal digital silence
    # else: bed stays at its natural recorded level; speech gets scaled to it

    n_clips = rng.randint(MIN_SPEECH_CLIPS_PER_SCENE, MAX_SPEECH_CLIPS_PER_SCENE)
    chosen_speech = [speech_files[rng.randrange(len(speech_files))] for _ in range(n_clips)]

    # lay out non-overlapping 1s speech clips at random positions with gaps
    speech_intervals_samples = []
    cursor = int(rng.uniform(0.3, 1.5) * SR)  # lead-in noise-only
    for f in chosen_speech:
        clip = _load_mono_16k(os.path.join(SPEECH_DIR, f))
        clip_len = len(clip)
        end = cursor + clip_len
        if end >= n_samples - int(0.5 * SR):
            break  # ran out of room; scene keeps whatever clips fit

        if snr_db is None:
            scaled_clip = clip  # natural recorded level over the near-silent bed
        else:
            noise_rms = _rms(bed[cursor:end])
            speech_rms = _rms(clip)
            target_speech_rms = noise_rms * (10 ** (snr_db / 20))
            scale = target_speech_rms / speech_rms
            scaled_clip = clip * scale

        bed[cursor:end] += scaled_clip
        speech_intervals_samples.append((cursor, end))

        gap = int(rng.uniform(0.5, 2.5) * SR)
        cursor = end + gap

    peak = np.max(np.abs(bed))
    if peak > 0.98:
        bed = bed * (0.98 / peak)

    speech_intervals_ms = [
        (round(s / SR * 1000), round(e / SR * 1000)) for s, e in speech_intervals_samples
    ]

    return bed, speech_intervals_ms, chosen_speech


def _frame_labels_10ms(duration_ms, speech_intervals_ms):
    n_frames = duration_ms // 10
    labels = [0] * n_frames
    for start_ms, end_ms in speech_intervals_ms:
        start_frame = start_ms // 10
        end_frame = min(n_frames, -(-end_ms // 10))  # ceil
        for i in range(start_frame, end_frame):
            labels[i] = 1
    return labels


def _generate_split(speech_files, noise_files, split_name, n_per_condition, rng):
    os.makedirs(os.path.join(OUT_DIR, split_name), exist_ok=True)
    manifest = []
    scene_idx = 0

    conditions = [("clean", None)] + [(f"snr_{db}db", db) for db in SNR_CONDITIONS_DB]

    for condition_name, snr_db in conditions:
        for _ in range(n_per_condition):
            noise_file = noise_files[rng.randrange(len(noise_files))]
            audio, speech_intervals_ms, chosen_speech = _make_scene(
                speech_files, noise_file, snr_db, rng
            )
            scene_id = f"{split_name}_scene_{scene_idx:04d}_{condition_name}"
            wav_path = os.path.join(OUT_DIR, split_name, f"{scene_id}.wav")
            sf.write(wav_path, audio, SR, subtype="PCM_16")

            duration_ms = round(len(audio) / SR * 1000)
            frame_labels = _frame_labels_10ms(duration_ms, speech_intervals_ms)

            meta = {
                "scene_id": scene_id,
                "split": split_name,
                "condition": condition_name,
                "snr_db": snr_db,
                "duration_ms": duration_ms,
                "sample_rate": SR,
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


def generate_all():
    rng = random.Random(SEED)

    # Reuse the existing seeded split so scenes never cross the train/test
    # boundary the original benchmark already established.
    train_speech, train_noise, test_speech, test_noise = split_dataset(SPEECH_DIR, NOISE_DIR)

    # Order matters for the seeded rng: dev and test are generated first, in
    # the exact same order as the original run, so their scene content is
    # byte-identical to what NOVA-VAD v0 was already scored on
    # (reports/frame_level_benchmark_v1.md's -0.28 MCC result stays valid
    # and comparable). Train scenes are new and go last.
    print("Generating DEV scenes (from train-split files, for threshold tuning only)...")
    dev_manifest = _generate_split(
        train_speech, train_noise, "dev", SCENES_PER_CONDITION_DEV, rng
    )
    print(f"  {len(dev_manifest)} dev scenes written to {OUT_DIR}/dev/")

    print("Generating TEST scenes (from held-out files, locked for final scoring)...")
    test_manifest = _generate_split(
        test_speech, test_noise, "test", SCENES_PER_CONDITION_TEST, rng
    )
    print(f"  {len(test_manifest)} test scenes written to {OUT_DIR}/test/")

    print("Generating TRAIN scenes (from train-split files, for frame-level model training)...")
    train_manifest = _generate_split(
        train_speech, train_noise, "train", SCENES_PER_CONDITION_TRAIN, rng
    )
    print(f"  {len(train_manifest)} train scenes written to {OUT_DIR}/train/")

    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
        json.dump({"train": train_manifest, "dev": dev_manifest, "test": test_manifest}, f, indent=2)

    total_speech_s = sum(
        (e - s) for m in test_manifest for s, e in m["speech_intervals_ms"]
    ) / 1000
    total_s = sum(m["duration_ms"] for m in test_manifest) / 1000
    print(f"\nTest set: {total_s:.0f}s total audio, {total_speech_s:.0f}s labeled speech "
          f"({total_speech_s/total_s*100:.0f}%)")


if __name__ == "__main__":
    generate_all()
