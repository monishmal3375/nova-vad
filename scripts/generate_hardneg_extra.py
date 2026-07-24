"""
Three additional hard-negative categories (Plan Section 7.2 Layer 4),
chosen specifically because they can be sourced with zero licensing risk
-- either synthesized from first principles or built from audio this
project already has a documented license for. The other categories in
the plan's list (laughter, coughing, crying, singing) need authentic
recorded audio to be a meaningful test -- faking them with crude
synthesis would produce a result that doesn't actually test what it
claims to, so they are explicitly NOT attempted here (see
reports/decision_v7.md for the full reasoning). Television is separately
deferred for licensing risk, as flagged.

1. Overlapping speech: two real speech clips from the test-only speech
   pool (Google Speech Commands, CC BY 4.0 -- same license already
   documented for every other use of this corpus in this project),
   mixed together. This is the highest-fidelity category here, since
   it's genuinely real speech, not a synthetic proxy.

2. Breathing: filtered broadband noise with a slow rhythmic amplitude
   envelope (inhale/exhale), matching breathing's actual acoustic
   signature (turbulent airflow = broadband noise, not periodic/tonal --
   a genuine hard negative for energy-based features, though NOT for the
   periodicity feature, which correctly should NOT fire on this). Fully
   synthesized, zero licensing risk.

3. Hold music: a simple original synthesized melodic loop (sine-wave
   tones with a fixed note sequence and gentle envelope). Representative
   of the "tonal, periodic, non-speech audio" character of real hold
   music without using any actual copyrighted composition. Fully
   original, zero licensing risk.

Ground truth for all: no speech anywhere in the scene (these are pure
hard-negative scenes, like the DTMF scenes in round 4), EXCEPT
overlapping speech, where ground truth is speech-present for the full
duration both clips overlap (it IS real speech, just two speakers at
once -- the "hard" part is whether VAD/downstream systems handle
overlap correctly, not whether speech is present).

Run: python3 -m scripts.generate_hardneg_extra
"""
import json
import os
import random

import librosa
import numpy as np
import soundfile as sf

from src.benchmark import split_dataset

SR = 16000
OUT_DIR = "data/scenes/test_v2_hardneg_extra"
SPEECH_DIR = "data/speech"
NOISE_DIR = "data/noise"
SEED = 47  # different from every prior seed in this project
N_SCENES_PER_CATEGORY = 8
SCENE_DURATION_S = 6.0


def _frame_labels_10ms(duration_ms, speech_intervals_ms):
    n_frames = duration_ms // 10
    labels = [0] * n_frames
    for start_ms, end_ms in speech_intervals_ms:
        start_frame = start_ms // 10
        end_frame = min(n_frames, -(-end_ms // 10))
        for i in range(start_frame, end_frame):
            labels[i] = 1
    return labels


def gen_breathing(rng, n_samples):
    """Filtered broadband noise with a slow rhythmic amplitude envelope."""
    noise = rng.normal(0, 1, n_samples).astype(np.float32)
    # low-pass filter to make it "breath-like" (not harsh white noise) --
    # simple moving-average filter, no scipy filter design needed
    kernel_size = 15
    kernel = np.ones(kernel_size) / kernel_size
    filtered = np.convolve(noise, kernel, mode="same")

    # breathing envelope: ~4-second full cycle (inhale+exhale), smooth
    t = np.arange(n_samples) / SR
    breath_cycle_s = 4.0
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * t / breath_cycle_s - np.pi / 2)
    envelope = envelope ** 1.5  # sharpen peaks slightly, like real breath attack

    audio = filtered * envelope * 0.15
    return audio.astype(np.float32)


def gen_hold_music(rng, n_samples):
    """Simple original synthesized melodic loop -- sine tones, fixed note
    sequence, gentle envelope. Not based on any existing composition."""
    notes_hz = [261.63, 293.66, 329.63, 349.23, 392.00, 349.23, 329.63, 293.66]  # C major scale snippet
    note_dur_s = SCENE_DURATION_S / len(notes_hz)
    note_samples = int(note_dur_s * SR)

    audio = np.zeros(n_samples, dtype=np.float32)
    for i, freq in enumerate(notes_hz):
        start = i * note_samples
        end = min(n_samples, start + note_samples)
        t = np.arange(end - start) / SR
        tone = 0.2 * np.sin(2 * np.pi * freq * t)
        # gentle attack/decay envelope per note
        env = np.ones(len(t))
        attack = min(len(t) // 8, 400)
        if attack > 0:
            env[:attack] = np.linspace(0, 1, attack)
            env[-attack:] = np.linspace(1, 0, attack)
        audio[start:end] = tone * env
    return audio


def gen_overlapping_speech(rng, speech_files, n_samples):
    """Mix two real speech clips from the test-only pool, overlapping."""
    f1, f2 = rng.sample(speech_files, 2)
    clip1, _ = librosa.load(os.path.join(SPEECH_DIR, f1), sr=SR, mono=True)
    clip2, _ = librosa.load(os.path.join(SPEECH_DIR, f2), sr=SR, mono=True)

    bed = np.zeros(n_samples, dtype=np.float32)
    pos1 = int(rng.uniform(0.5, 1.5) * SR)
    pos2 = int(rng.uniform(0.8, 2.0) * SR)  # overlapping start, not identical

    def _add(bed, clip, pos):
        end = min(len(bed), pos + len(clip))
        bed[pos:end] += clip[:end - pos]
        return pos, end

    s1, e1 = _add(bed, clip1, pos1)
    s2, e2 = _add(bed, clip2, pos2)

    peak = np.max(np.abs(bed))
    if peak > 0.95:
        bed = bed * (0.95 / peak)

    overlap_start_ms = round(min(s1, s2) / SR * 1000)
    overlap_end_ms = round(max(e1, e2) / SR * 1000)
    return bed, [(overlap_start_ms, overlap_end_ms)], [f1, f2]


def generate():
    rng_np = np.random.RandomState(SEED)
    rng_py = random.Random(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    n_samples = int(SCENE_DURATION_S * SR)

    train_speech, train_noise, test_speech, test_noise = split_dataset(SPEECH_DIR, NOISE_DIR)

    manifest = []
    categories = {
        "breathing": lambda: (gen_breathing(rng_np, n_samples), [], []),
        "hold_music": lambda: (gen_hold_music(rng_np, n_samples), [], []),
        "overlapping_speech": lambda: gen_overlapping_speech(rng_py, test_speech, n_samples),
    }

    for cat_name, gen_fn in categories.items():
        print(f"\nGenerating {cat_name}...")
        for i in range(N_SCENES_PER_CATEGORY):
            audio, speech_intervals_ms, source_files = gen_fn()
            scene_id = f"hardneg_{cat_name}_{i:03d}"
            wav_path = os.path.join(OUT_DIR, f"{scene_id}.wav")
            sf.write(wav_path, audio, SR, subtype="PCM_16")

            duration_ms = round(len(audio) / SR * 1000)
            frame_labels = _frame_labels_10ms(duration_ms, speech_intervals_ms)

            meta = {
                "scene_id": scene_id,
                "condition": f"hardneg_{cat_name}",
                "duration_ms": duration_ms,
                "sample_rate": SR,
                "speech_intervals_ms": speech_intervals_ms,
                "frame_labels_10ms": frame_labels,
                "source_files": source_files,
                "license": "Google Speech Commands v0.02, CC BY 4.0" if source_files else "Synthesized, no external source, no licensing constraint",
            }
            with open(os.path.join(OUT_DIR, f"{scene_id}.json"), "w") as f:
                json.dump(meta, f, indent=2)
            manifest.append(meta)
        print(f"  {N_SCENES_PER_CATEGORY} {cat_name} scenes written")

    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
        json.dump({"hardneg_extra": manifest}, f, indent=2)
    print(f"\n{len(manifest)} total hard-negative scenes written to {OUT_DIR}/")


if __name__ == "__main__":
    generate()
