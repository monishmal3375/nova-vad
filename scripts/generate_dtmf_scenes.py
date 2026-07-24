"""
Hard-negative test: DTMF tones (plan Section 7.2 Layer 4). Synthesized
programmatically — no new external audio source needed, so no new leakage
surface to check (pure synthesis, doesn't draw on train or test speech/
noise pools at all).

Scope, disclosed explicitly: this is ONE of the ~8 hard-negative
categories the plan lists (laughter, coughing, crying, singing, TV, DTMF,
hold music, breathing, overlapping speech). DTMF was chosen because it's
the only one synthesizable with zero data-sourcing/licensing risk within
this round's time budget. The other ~7 categories need real audio this
project doesn't currently have and are explicitly DEFERRED, not silently
skipped — see reports/decision_v6.md.

DTMF is a meaningful test for this project specifically: it's strongly
tonal/periodic (dual-tone sine pairs), which is exactly the acoustic
property NOVA-VAD-frame-v2's periodicity_strength feature was designed to
detect in *speech* — a real risk is that a periodicity-leaning VAD
mistakes DTMF for speech. This scene set directly probes that.

Ground truth: NO speech anywhere in these scenes (frame_labels_10ms is
all zeros) — DTMF tones are not speech, full stop. A well-behaved VAD's
false-positive rate on this set is the metric that matters.

Run: python3 -m scripts.generate_dtmf_scenes
"""
import json
import os
import random

import numpy as np
import soundfile as sf

OUT_DIR = "data/scenes/test_v2_hardneg"
SR = 16000
SEED = 47
N_SCENES = 10
SCENE_DURATION_S = 12.0

DTMF_FREQS = {
    "1": (697, 1209), "2": (697, 1336), "3": (697, 1477),
    "4": (770, 1209), "5": (770, 1336), "6": (770, 1477),
    "7": (852, 1209), "8": (852, 1336), "9": (852, 1477),
    "*": (941, 1209), "0": (941, 1336), "#": (941, 1477),
}


def _dtmf_tone(digit, duration_s, sr=SR):
    f1, f2 = DTMF_FREQS[digit]
    t = np.arange(int(duration_s * sr)) / sr
    tone = 0.5 * (np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t))
    # short fade in/out to avoid clicks (standard DTMF generation practice)
    fade = min(len(tone) // 10, int(0.005 * sr))
    if fade > 0:
        tone[:fade] *= np.linspace(0, 1, fade)
        tone[-fade:] *= np.linspace(1, 0, fade)
    return tone.astype(np.float32)


def generate():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = random.Random(SEED)
    manifest = []

    digits = list(DTMF_FREQS.keys())
    for scene_idx in range(N_SCENES):
        n_samples = int(SCENE_DURATION_S * SR)
        bed = np.zeros(n_samples, dtype=np.float32)

        n_tones = rng.randint(3, 6)
        cursor = int(rng.uniform(0.3, 1.5) * SR)
        for _ in range(n_tones):
            digit = digits[rng.randrange(len(digits))]
            tone_dur = rng.uniform(0.15, 0.35)  # realistic DTMF press duration
            tone = _dtmf_tone(digit, tone_dur)
            end = cursor + len(tone)
            if end >= n_samples - int(0.3 * SR):
                break
            bed[cursor:end] += tone
            cursor = end + int(rng.uniform(0.3, 1.2) * SR)

        peak = np.max(np.abs(bed))
        if peak > 0.98:
            bed = bed * (0.98 / peak)

        scene_id = f"test_v2_hardneg_dtmf_{scene_idx:04d}"
        wav_path = os.path.join(OUT_DIR, f"{scene_id}.wav")
        sf.write(wav_path, bed, SR, subtype="PCM_16")

        duration_ms = round(len(bed) / SR * 1000)
        n_frames = duration_ms // 10
        meta = {
            "scene_id": scene_id,
            "split": "test_v2_hardneg",
            "condition": "hardneg_dtmf",
            "duration_ms": duration_ms,
            "sample_rate": SR,
            "speech_intervals_ms": [],  # DTMF is never speech
            "frame_labels_10ms": [0] * n_frames,
            "source_speech_files": [],
            "source_noise_file": None,
            "note": "synthesized DTMF tones, no real audio source, no leakage surface",
        }
        with open(os.path.join(OUT_DIR, f"{scene_id}.json"), "w") as f:
            json.dump(meta, f, indent=2)
        manifest.append(meta)

    with open("data/scenes/test_v2_hardneg_manifest.json", "w") as f:
        json.dump({"test_v2_hardneg": manifest}, f, indent=2)
    print(f"{len(manifest)} DTMF hard-negative scenes written to {OUT_DIR}/")


if __name__ == "__main__":
    generate()
