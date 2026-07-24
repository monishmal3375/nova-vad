"""
Applies codec degradation (G.711 A-law, G.711 mu-law, Opus @24kbps) to the
25 CLEAN test_v2 scenes, producing 3 new codec conditions closing plan
Section 7.3's "raw/degraded/actual-transmission" gap for the minimum
required codecs.

Scope decision, disclosed explicitly: codec degradation is applied to
CLEAN scenes only, not crossed with the SNR conditions (would be a 3
codec x 4 SNR = 12-condition matrix) — this isolates "does codec alone
hurt performance" the same way the original SNR conditions isolated
"does noise alone hurt performance" from codec effects. Codec x noise
interaction is NOT tested and is an explicit, disclosed gap for a future
round, not a claim of completeness beyond what's actually done here.

No new leakage risk: this transforms already-test-only-pool audio
(test_v2's clean scenes, already leakage-verified against every
train-side split) — no new source files are drawn, so no new leakage
check is needed beyond what test_v2 already has.

Ground truth (frame_labels_10ms, speech_intervals_ms) is copied unchanged
from the source clean scene — codec degradation alters audio quality, not
speech timing (Opus's small algorithmic delay is corrected for length/
alignment inside scripts/codec_degrade.py, verified in
tests/test_codec_degrade.py).

Run: python3 -m scripts.generate_codec_scenes
"""
import glob
import json
import os

import librosa
import numpy as np
import soundfile as sf

from scripts.codec_degrade import apply_g711, apply_opus

SOURCE_DIR = "data/scenes/test_v2"
OUT_DIR = "data/scenes/test_v2_codec"
SR = 16000

CODECS = {
    "codec_g711_alaw": lambda audio: apply_g711(audio, "alaw"),
    "codec_g711_mulaw": lambda audio: apply_g711(audio, "mulaw"),
    "codec_opus24k": lambda audio: apply_opus(audio, bitrate=24000),
}


def generate():
    os.makedirs(OUT_DIR, exist_ok=True)
    clean_jsons = sorted(glob.glob(os.path.join(SOURCE_DIR, "*_clean.json")))
    print(f"Found {len(clean_jsons)} clean test_v2 scenes to degrade")

    manifest = []
    for codec_name, codec_fn in CODECS.items():
        print(f"\nApplying {codec_name}...")
        for json_path in clean_jsons:
            with open(json_path) as f:
                meta = json.load(f)
            wav_path = json_path.replace(".json", ".wav")
            audio, sr = librosa.load(wav_path, sr=SR, mono=True)

            degraded = codec_fn(audio)
            n = min(len(audio), len(degraded))
            degraded = degraded[:n]

            new_scene_id = meta["scene_id"].replace("_clean", f"_{codec_name}")
            out_wav = os.path.join(OUT_DIR, f"{new_scene_id}.wav")
            sf.write(out_wav, degraded, SR, subtype="PCM_16")

            new_meta = dict(meta)
            new_meta["scene_id"] = new_scene_id
            new_meta["condition"] = codec_name
            new_meta["source_clean_scene"] = meta["scene_id"]
            with open(os.path.join(OUT_DIR, f"{new_scene_id}.json"), "w") as f:
                json.dump(new_meta, f, indent=2)
            manifest.append(new_meta)

    with open("data/scenes/test_v2_codec_manifest.json", "w") as f:
        json.dump({"test_v2_codec": manifest}, f, indent=2)
    print(f"\n{len(manifest)} codec-degraded scenes written to {OUT_DIR}/")


if __name__ == "__main__":
    generate()
