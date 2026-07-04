import os
import sys
import time
import argparse
import warnings
import numpy as np
import sounddevice as sd
import joblib
import queue
import threading
from collections import deque

warnings.filterwarnings("ignore")

from src.classifier import extract_features_from_array

# ── Settings ───────────────────────────────────────────────────────────────
SAMPLE_RATE    = 16000
CHUNK_DURATION = 1.0
CHUNK_SAMPLES  = int(SAMPLE_RATE * CHUNK_DURATION)

# Hysteresis / smoothing settings — prevents SPEECH/NO SPEECH flicker on
# borderline frames. A state flip only takes effect once the new label has
# been the majority decision across the last SMOOTHING_WINDOW raw chunk
# decisions, requiring at least SMOOTHING_MIN_AGREE of them to agree.
SMOOTHING_WINDOW    = 5   # number of most recent 1s chunks considered
SMOOTHING_MIN_AGREE = 3   # how many of those must agree to flip displayed state

# ── Audio Queue ────────────────────────────────────────────────────────────
audio_queue = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"  Audio status: {status}")
    audio_queue.put(indata.copy())


# ── Device selection ───────────────────────────────────────────────────────
def list_input_devices() -> list:
    """
    Returns a list of (index, device_info) tuples for devices with at
    least one input channel.
    """
    devices = sd.query_devices()
    return [(i, d) for i, d in enumerate(devices) if d.get("max_input_channels", 0) > 0]


def print_input_devices(input_devices: list):
    print("\n  Available input devices:")
    print("  " + "-" * 44)
    try:
        default_in, _ = sd.default.device
    except Exception:
        default_in = None

    for i, d in input_devices:
        marker = " (default)" if i == default_in else ""
        print(f"    [{i}] {d['name']} — {d['max_input_channels']}ch{marker}")
    print("  " + "-" * 44)


def choose_input_device(preferred: int = None) -> int:
    """
    Resolves which input device index to use.

    - If `preferred` is given (via --device flag) and valid, use it.
    - Otherwise, if running interactively, list devices and prompt.
    - Otherwise, fall back to the system default input device.
    """
    input_devices = list_input_devices()

    if not input_devices:
        print("  No input devices found — using system default.")
        return None

    if preferred is not None:
        valid_indices = [i for i, _ in input_devices]
        if preferred in valid_indices:
            name = next(d["name"] for i, d in input_devices if i == preferred)
            print(f"  Using selected device [{preferred}] {name}")
            return preferred
        print(f"  Device index {preferred} is not a valid input device. Falling back to prompt.")

    print_input_devices(input_devices)

    if not sys.stdin.isatty():
        # non-interactive (e.g. piped/CI) — use system default
        print("  Non-interactive session — using system default input device.")
        return None

    try:
        raw = input("\n  Select a device index (Enter for default): ").strip()
    except EOFError:
        raw = ""

    if raw == "":
        print("  Using system default input device.")
        return None

    try:
        choice = int(raw)
    except ValueError:
        print("  Invalid input — using system default input device.")
        return None

    valid_indices = [i for i, _ in input_devices]
    if choice not in valid_indices:
        print("  Invalid device index — using system default input device.")
        return None

    return choice


# ── Smoothing / Hysteresis ─────────────────────────────────────────────────
class StateSmoother:
    """
    Debounces the raw per-chunk SPEECH/NO SPEECH/UNCERTAIN decision so the
    displayed state doesn't flicker on borderline frames. Keeps a rolling
    window of the last N raw labels and only flips the displayed state once
    the new label has at least `min_agree` votes in that window.
    """
    def __init__(self, window: int = SMOOTHING_WINDOW, min_agree: int = SMOOTHING_MIN_AGREE):
        self.window    = window
        self.min_agree = min_agree
        self.history   = deque(maxlen=window)
        self.state     = None  # currently displayed label

    def update(self, raw_label: str) -> str:
        self.history.append(raw_label)

        if self.state is None:
            self.state = raw_label
            return self.state

        if raw_label == self.state:
            return self.state

        # candidate flip — only accept if it has majority support
        # in the recent window
        votes = sum(1 for h in self.history if h == raw_label)
        if votes >= self.min_agree:
            self.state = raw_label

        return self.state


# ── Process Audio ──────────────────────────────────────────────────────────
def process_stream_dynamic(rf, scaler, threshold=55):
    buffer   = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
    ptr      = 0
    smoother = StateSmoother()

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
                    raw_label = "SPEECH"
                    confidence = speech_prob
                elif noise_prob >= threshold:
                    raw_label = "NO SPEECH"
                    confidence = noise_prob
                else:
                    raw_label = "UNCERTAIN"
                    confidence = max(speech_prob, noise_prob)

                # apply hysteresis so the displayed state only flips once
                # the recent chunk history agrees
                smoothed_label = smoother.update(raw_label)

                color = {
                    "SPEECH":    "\033[92m",
                    "NO SPEECH": "\033[91m",
                    "UNCERTAIN": "\033[93m",
                }.get(smoothed_label, "\033[93m")

                reset     = "\033[0m"
                timestamp = time.strftime("%H:%M:%S")
                flip_note = "" if smoothed_label == raw_label else f" (raw: {raw_label})"
                print(f"  [{timestamp}] {color}{smoothed_label}{reset} — {confidence:.1f}% confidence{flip_note}")

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
def calibrate_silence(seconds=3, device=None) -> np.ndarray:
    """
    Records room silence to establish a baseline noise floor.
    """
    print(f"\n  Calibrating — stay quiet for {seconds} seconds...")
    recording = sd.rec(
        int(seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype=np.float32,
        device=device
    )
    for i in range(seconds, 0, -1):
        print(f"  {i}...")
        time.sleep(1)
    sd.wait()
    print("  Calibration complete.\n")
    return recording.flatten()


def run_stream(device: int = None, list_devices_only: bool = False):
    print("=" * 50)
    print("  NOVA-VAD — REAL TIME STREAMING")
    print("=" * 50)

    if list_devices_only:
        print_input_devices(list_input_devices())
        return

    device = choose_input_device(preferred=device)

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
    silence_sample  = calibrate_silence(seconds=3, device=device)
    silence_features = extract_features_from_array(silence_sample, SAMPLE_RATE)
    silence_prob     = rf.predict_proba(scaler.transform([silence_features]))[0][1] * 100
    print(f"  Room noise speech probability: {silence_prob:.1f}%")
    print(f"  Using {silence_prob + 10:.1f}% as dynamic threshold\n")

    # dynamic threshold based on your actual room
    dynamic_threshold = min(silence_prob + 10, 60)

    print(f"  Sample rate:    {SAMPLE_RATE}Hz")
    print(f"  Chunk duration: {CHUNK_DURATION}s")
    print(f"  Speech threshold: {dynamic_threshold:.1f}%")
    print(f"  Smoothing:      {SMOOTHING_MIN_AGREE}/{SMOOTHING_WINDOW} chunk hysteresis")
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
        device=device,
        callback=audio_callback
    ):
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n\n  Stopped.")
            print("=" * 50)


def parse_args():
    parser = argparse.ArgumentParser(description="NOVA-VAD realtime streaming")
    parser.add_argument(
        "--device", "-d", type=int, default=None,
        help="Input device index to use (see --list-devices). Skips the interactive prompt."
    )
    parser.add_argument(
        "--list-devices", action="store_true",
        help="List available input devices and exit."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_stream(device=args.device, list_devices_only=args.list_devices)
