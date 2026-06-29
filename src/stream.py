import os
import time
import warnings
import numpy as np
import sounddevice as sd
import joblib
import queue
import threading

warnings.filterwarnings("ignore")

from src.classifier import extract_features_from_array

# ── Settings ───────────────────────────────────────────────────────────────
SAMPLE_RATE    = 16000
CHUNK_DURATION = 1.0
CHUNK_SAMPLES  = int(SAMPLE_RATE * CHUNK_DURATION)

# ── Audio Queue ────────────────────────────────────────────────────────────
audio_queue = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"  Audio status: {status}")
    audio_queue.put(indata.copy())

# ── Process Audio ──────────────────────────────────────────────────────────
def process_stream_dynamic(rf, scaler, threshold=55):
    buffer = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
    ptr    = 0

    while True:
        try:
            chunk = audio_queue.get(timeout=2.0)
            chunk = chunk[:, 0]

            space = CHUNK_SAMPLES - ptr
            if len(chunk) >= space:
                buffer[ptr:] = chunk[:space]

                features    = extract_features_from_array(buffer, SAMPLE_RATE)
                X_scaled    = scaler.transform([features])
                probs       = rf.predict_proba(X_scaled)[0]
                speech_prob = probs[1] * 100
                noise_prob  = probs[0] * 100

                if speech_prob >= threshold:
                    label      = "SPEECH"
                    confidence = speech_prob
                    color      = "\033[92m"
                elif noise_prob >= threshold:
                    label      = "NO SPEECH"
                    confidence = noise_prob
                    color      = "\033[91m"
                else:
                    label      = "UNCERTAIN"
                    confidence = max(speech_prob, noise_prob)
                    color      = "\033[93m"

                reset     = "\033[0m"
                timestamp = time.strftime("%H:%M:%S")
                print(f"  [{timestamp}] {color}{label}{reset} — {confidence:.1f}% confidence")

                buffer = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
                ptr    = 0
            else:
                buffer[ptr:ptr + len(chunk)] = chunk
                ptr += len(chunk)

        except queue.Empty:
            continue
        except KeyboardInterrupt:
            break

# ── Main ───────────────────────────────────────────────────────────────────
def calibrate_silence(seconds=3) -> np.ndarray:
    """
    Records room silence to establish a baseline noise floor.
    """
    print(f"\n  Calibrating — stay quiet for {seconds} seconds...")
    recording = sd.rec(
        int(seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype=np.float32
    )
    for i in range(seconds, 0, -1):
        print(f"  {i}...")
        time.sleep(1)
    sd.wait()
    print("  Calibration complete.\n")
    return recording.flatten()


def run_stream():
    print("=" * 50)
    print("  NOVA-VAD — REAL TIME STREAMING")
    print("=" * 50)

    if os.path.exists("models/stream_rf.pkl"):
        rf_path     = "models/stream_rf.pkl"
        scaler_path = "models/stream_scaler.pkl"
        print("  Using streaming-optimized model")
    else:
        rf_path     = "models/nova_vad_rf.pkl"
        scaler_path = "models/nova_vad_scaler.pkl"
        print("  Using general model (run retrain_streaming.py for better streaming)")

    if not os.path.exists(rf_path):
        print("Model not found. Run python3 -m src.pipeline first.")
        return

    rf     = joblib.load(rf_path)
    scaler = joblib.load(scaler_path)
    print("Model loaded")

    # calibrate on your actual room silence
    silence_sample  = calibrate_silence(seconds=3)
    silence_features = extract_features_from_array(silence_sample, SAMPLE_RATE)
    silence_prob     = rf.predict_proba(scaler.transform([silence_features]))[0][1] * 100
    print(f"  Room noise speech probability: {silence_prob:.1f}%")
    print(f"  Using {silence_prob + 10:.1f}% as dynamic threshold\n")

    # dynamic threshold based on your actual room
    dynamic_threshold = min(silence_prob + 10, 60)

    print(f"  Sample rate:    {SAMPLE_RATE}Hz")
    print(f"  Chunk duration: {CHUNK_DURATION}s")
    print(f"  Speech threshold: {dynamic_threshold:.1f}%")
    print("\n  Listening... (Press Ctrl+C to stop)\n")
    print("  " + "-" * 44)

    processor = threading.Thread(
        target=process_stream_dynamic,
        args=(rf, scaler, dynamic_threshold),
        daemon=True
    )
    processor.start()

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype=np.float32,
        blocksize=int(SAMPLE_RATE * 0.1),
        callback=audio_callback
    ):
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n\n  Stopped.")
            print("=" * 50)

if __name__ == "__main__":
    run_stream()
