"""
Builds a typed manifest of every current speech and noise file — the
foundation for grouped splits and the mixed-scene generator.
Run: python3 -m scripts.build_manifest
"""
import csv
import os
import soundfile as sf

SPEECH_DIR = "data/speech"
NOISE_DIR = "data/noise"
OUT_PATH = "data/manifests/current_files.csv"


def build_manifest():
    rows = []

    for f in sorted(os.listdir(SPEECH_DIR)):
        if not f.endswith(".wav"):
            continue
        path = os.path.join(SPEECH_DIR, f)
        info = sf.info(path)
        rows.append({
            "sample_id": f"speech_{f}",
            "filename": f,
            "path": path,
            "label": "speech",
            "source_dataset": "google_speech_commands_v0.02",
            "original_sample_rate": info.samplerate,
            "channels": info.channels,
            "duration_ms": round(info.duration * 1000),
        })

    for f in sorted(os.listdir(NOISE_DIR)):
        if not f.endswith(".wav"):
            continue
        path = os.path.join(NOISE_DIR, f)
        info = sf.info(path)
        rows.append({
            "sample_id": f"noise_{f}",
            "filename": f,
            "path": path,
            "label": "noise",
            "source_dataset": "musan_noise_and_music",
            "original_sample_rate": info.samplerate,
            "channels": info.channels,
            "duration_ms": round(info.duration * 1000),
        })

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    n_speech = sum(1 for r in rows if r["label"] == "speech")
    n_noise = sum(1 for r in rows if r["label"] == "noise")
    print(f"Wrote {OUT_PATH}: {n_speech} speech + {n_noise} noise = {len(rows)} files")


if __name__ == "__main__":
    build_manifest()
