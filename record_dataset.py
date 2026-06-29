import sounddevice as sd
import soundfile as sf
import numpy as np
import os
import time

SAMPLE_RATE = 16000
DURATION    = 2  # seconds per clip

def record_clip(filename: str, duration: int = DURATION):
    print(f"  Recording {filename} ({duration}s)...")
    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype=np.float32
    )
    sd.wait()
    sf.write(filename, audio.flatten(), SAMPLE_RATE)
    print(f"  Saved.")

def record_dataset():
    print("=" * 50)
    print("  NOVA-VAD MIC DATASET RECORDER")
    print("=" * 50)

    # create folders
    os.makedirs("data/mic_speech", exist_ok=True)
    os.makedirs("data/mic_noise",  exist_ok=True)

    # ── Record Speech ──────────────────────────────
    print("\n[ PART 1 ] Recording YOUR VOICE — 30 clips")
    print("  Say anything naturally — count, read, talk")
    print("  Each clip is 2 seconds\n")

    for i in range(1, 31):
        input(f"  Press ENTER to record speech clip {i}/30...")
        record_clip(f"data/mic_speech/mic_speech_{i:03d}.wav")
        time.sleep(0.3)

    # ── Record Silence/Noise ───────────────────────
    print("\n[ PART 2 ] Recording ROOM NOISE — 30 clips")
    print("  Stay quiet — let it capture your room")
    print("  Move around, type, do normal things\n")

    for i in range(1, 31):
        input(f"  Press ENTER to record noise clip {i}/30...")
        record_clip(f"data/mic_noise/mic_noise_{i:03d}.wav")
        time.sleep(0.3)

    print("\n" + "=" * 50)
    print("  RECORDING COMPLETE")
    print("=" * 50)
    print(f"  mic_speech/ → {len(os.listdir('data/mic_speech'))} files")
    print(f"  mic_noise/  → {len(os.listdir('data/mic_noise'))} files")
    print("\n  Now run: python3 retrain_streaming.py")

if __name__ == "__main__":
    record_dataset()