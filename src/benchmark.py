import os
import csv
import json
import time
import random
import shutil
import platform
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from src.classifier import extract_features, build_dataset
from src.vad import detect_speech
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

RESULTS_DIR = "results"

# ── Train/Test Split ───────────────────────────────────────────────────────
def split_dataset(speech_dir: str, noise_dir: str, test_ratio: float = 0.2):
    """
    Splits files into 80% train and 20% test.
    Returns train and test file lists.
    """
    speech_files = sorted([f for f in os.listdir(speech_dir) if f.endswith(".wav")])
    noise_files  = sorted([f for f in os.listdir(noise_dir)  if f.endswith(".wav")])

    random.seed(42)
    random.shuffle(speech_files)
    random.shuffle(noise_files)

    n_speech_test = int(len(speech_files) * test_ratio)
    n_noise_test  = int(len(noise_files)  * test_ratio)

    test_speech  = speech_files[:n_speech_test]
    train_speech = speech_files[n_speech_test:]
    test_noise   = noise_files[:n_noise_test]
    train_noise  = noise_files[n_noise_test:]

    print(f"  Train: {len(train_speech)} speech + {len(train_noise)} noise = {len(train_speech)+len(train_noise)} files")
    print(f"  Test:  {len(test_speech)} speech + {len(test_noise)} noise = {len(test_speech)+len(test_noise)} files")

    return train_speech, train_noise, test_speech, test_noise


# ── Noise category manifest (from download_noise.py) ──────────────────────
def load_noise_category_manifest(noise_dir: str) -> dict:
    """
    Loads the filename -> UrbanSound8K category mapping written by
    download_noise.py (data/noise/_category_manifest.csv), if present.
    Returns {} if no manifest exists (e.g. noise came from MUSAN via
    download_data.py instead).
    """
    manifest_path = os.path.join(noise_dir, "_category_manifest.csv")
    mapping = {}
    if not os.path.exists(manifest_path):
        return mapping
    with open(manifest_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[row["noise_filename"]] = row["category"]
    return mapping


# ── Train NOVA-VAD on training set only ───────────────────────────────────
def train_nova_vad(train_speech: list, train_noise: list,
                   speech_dir: str, noise_dir: str):
    """
    Trains NOVA-VAD on training files only.
    Returns trained rf, gbt, scaler.
    """
    X, y = [], []

    print("  Extracting features from training set...")
    for f in train_speech:
        path = os.path.join(speech_dir, f)
        X.append(extract_features(path))
        y.append(1)

    for f in train_noise:
        path = os.path.join(noise_dir, f)
        X.append(extract_features(path))
        y.append(0)

    X = np.array(X)
    y = np.array(y)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rf  = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    gbt = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

    rf.fit(X_scaled, y)
    gbt.fit(X_scaled, y)

    return rf, gbt, scaler


# ── Run NOVA-VAD on test set ───────────────────────────────────────────────
def run_nova_vad(test_speech: list, test_noise: list,
                 speech_dir: str, noise_dir: str,
                 rf, gbt, scaler) -> dict:
    results = []
    start   = time.time()

    for f in test_speech:
        path        = os.path.join(speech_dir, f)
        t0          = time.time()
        features    = extract_features(path)
        X_scaled    = scaler.transform([features])
        rf_prob     = rf.predict_proba(X_scaled)[0][1]
        gbt_prob    = gbt.predict_proba(X_scaled)[0][1]
        avg_prob    = (rf_prob + gbt_prob) / 2
        pred        = 1 if avg_prob > 0.5 else 0
        latency_ms  = (time.time() - t0) * 1000
        results.append({
            "true": 1, "pred": pred, "file": f,
            "confidence": round(float(avg_prob if pred == 1 else 1 - avg_prob) * 100, 2),
            "latency_ms": round(latency_ms, 2)
        })

    for f in test_noise:
        path        = os.path.join(noise_dir, f)
        t0          = time.time()
        features    = extract_features(path)
        X_scaled    = scaler.transform([features])
        rf_prob     = rf.predict_proba(X_scaled)[0][1]
        gbt_prob    = gbt.predict_proba(X_scaled)[0][1]
        avg_prob    = (rf_prob + gbt_prob) / 2
        pred        = 1 if avg_prob > 0.5 else 0
        latency_ms  = (time.time() - t0) * 1000
        results.append({
            "true": 0, "pred": pred, "file": f,
            "confidence": round(float(avg_prob if pred == 1 else 1 - avg_prob) * 100, 2),
            "latency_ms": round(latency_ms, 2)
        })

    elapsed    = time.time() - start
    model_size = model_size_bytes([
        "models/nova_vad_rf.pkl", "models/nova_vad_gbt.pkl", "models/nova_vad_scaler.pkl"
    ])
    return compute_metrics(results, elapsed, "NOVA-VAD", model_size_bytes_val=model_size)


# ── Run WebRTC on test set ─────────────────────────────────────────────────
def run_webrtc(test_speech: list, test_noise: list,
               speech_dir: str, noise_dir: str) -> dict:
    results = []
    start   = time.time()

    for f in test_speech:
        path       = os.path.join(speech_dir, f)
        t0         = time.time()
        r          = detect_speech(path)
        latency_ms = (time.time() - t0) * 1000
        results.append({"true": 1, "pred": r["prediction"], "file": f,
                         "confidence": round(r["speech_ratio"] * 100, 2),
                         "latency_ms": round(latency_ms, 2)})

    for f in test_noise:
        path       = os.path.join(noise_dir, f)
        t0         = time.time()
        r          = detect_speech(path)
        latency_ms = (time.time() - t0) * 1000
        results.append({"true": 0, "pred": r["prediction"], "file": f,
                         "confidence": round(r["speech_ratio"] * 100, 2),
                         "latency_ms": round(latency_ms, 2)})

    elapsed = time.time() - start
    # webrtcvad is a thin C-extension wheel — no local model file to size
    return compute_metrics(results, elapsed, "WebRTC VAD", model_size_bytes_val=None)


# ── Run Silero on test set ─────────────────────────────────────────────────
def run_silero(test_speech: list, test_noise: list,
               speech_dir: str, noise_dir: str) -> dict:
    from silero_vad import load_silero_vad, get_speech_timestamps
    import soundfile as sf
    import torch
    import librosa

    model   = load_silero_vad()
    results = []
    start   = time.time()

    def predict(path):
        try:
            audio, sr = sf.read(path)
            # convert to mono
            if audio.ndim > 1:
                audio = audio[:, 0]
            # resample to 16kHz
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            wav      = torch.FloatTensor(audio)
            speeches = get_speech_timestamps(wav, model, sampling_rate=16000)
            return 1 if len(speeches) > 0 else 0
        except Exception as e:
            print(f"    Error on {path}: {e}")
            return 0

    for f in test_speech:
        t0         = time.time()
        pred       = predict(os.path.join(speech_dir, f))
        latency_ms = (time.time() - t0) * 1000
        results.append({"true": 1, "pred": pred, "file": f, "latency_ms": round(latency_ms, 2)})

    for f in test_noise:
        t0         = time.time()
        pred       = predict(os.path.join(noise_dir, f))
        latency_ms = (time.time() - t0) * 1000
        results.append({"true": 0, "pred": pred, "file": f, "latency_ms": round(latency_ms, 2)})

    elapsed = time.time() - start
    return compute_metrics(results, elapsed, "Silero VAD", model_size_bytes_val=None)

# ── Pyannote VAD ───────────────────────────────────────────────────────────
def run_pyannote(test_speech: list, test_noise: list,
                 speech_dir: str, noise_dir: str) -> dict:
    import os
    import torch
    import soundfile as sf
    import librosa
    from pyannote.audio import Model
    from pyannote.audio.pipelines import VoiceActivityDetection

    token    = os.environ.get("HF_TOKEN")
    model    = Model.from_pretrained("pyannote/segmentation-3.0", use_auth_token=token)
    pipeline = VoiceActivityDetection(segmentation=model)
    pipeline.instantiate({"min_duration_on": 0.0, "min_duration_off": 0.0})

    results = []
    start   = time.time()

    def predict(path):
        try:
            audio, sr = sf.read(path)
            if audio.ndim > 1:
                audio = audio[:, 0]
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                sr = 16000
            # pass as preloaded tensor dict — avoids torchcodec issue
            waveform = torch.FloatTensor(audio).unsqueeze(0)
            file_dict = {"waveform": waveform, "sample_rate": sr}
            output    = pipeline(file_dict)
            segments  = list(output.get_timeline())
            return 1 if len(segments) > 0 else 0
        except Exception as e:
            print(f"    Error on {path}: {e}")
            return 0

    for f in test_speech:
        t0         = time.time()
        pred       = predict(os.path.join(speech_dir, f))
        latency_ms = (time.time() - t0) * 1000
        results.append({"true": 1, "pred": pred, "file": f, "latency_ms": round(latency_ms, 2)})

    for f in test_noise:
        t0         = time.time()
        pred       = predict(os.path.join(noise_dir, f))
        latency_ms = (time.time() - t0) * 1000
        results.append({"true": 0, "pred": pred, "file": f, "latency_ms": round(latency_ms, 2)})

    elapsed = time.time() - start
    return compute_metrics(results, elapsed, "Pyannote VAD", model_size_bytes_val=None)

# ── SpeechBrain VAD ────────────────────────────────────────────────────────
def run_speechbrain(test_speech: list, test_noise: list,
                    speech_dir: str, noise_dir: str) -> dict:
    import torch
    import soundfile as sf
    import librosa
    import numpy as np
    from speechbrain.inference.VAD import VAD

    # load pretrained SpeechBrain VAD
    vad_model = VAD.from_hparams(
        source="speechbrain/vad-crdnn-libriparty",
        savedir="models/speechbrain_vad"
    )

    results = []
    start   = time.time()

    def predict(path):
        try:
            # load and resample to 16kHz
            audio, sr = sf.read(path)
            if audio.ndim > 1:
                audio = audio[:, 0]
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                sr = 16000

            # save temp file for speechbrain
            tmp_path = "data/tmp_sb.wav"
            import soundfile as sf2
            sf2.write(tmp_path, audio, sr)

            # run VAD
            boundaries = vad_model.get_speech_segments(tmp_path)
            pred = 1 if len(boundaries) > 0 else 0

            # cleanup
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

            return pred
        except Exception as e:
            print(f"    Error on {path}: {e}")
            return 0

    for f in test_speech:
        t0         = time.time()
        pred       = predict(os.path.join(speech_dir, f))
        latency_ms = (time.time() - t0) * 1000
        results.append({"true": 1, "pred": pred, "file": f, "latency_ms": round(latency_ms, 2)})

    for f in test_noise:
        t0         = time.time()
        pred       = predict(os.path.join(noise_dir, f))
        latency_ms = (time.time() - t0) * 1000
        results.append({"true": 0, "pred": pred, "file": f, "latency_ms": round(latency_ms, 2)})

    elapsed = time.time() - start
    return compute_metrics(results, elapsed, "SpeechBrain VAD", model_size_bytes_val=None)


# ── TEN-VAD ─────────────────────────────────────────────────────────────────
# https://github.com/TEN-framework/ten-vad — open source (Apache 2.0 + extra
# conditions), no API key, lightweight (numpy-only Python side, small native
# lib per-platform). Popular in real-time voice-agent stacks, so it's a
# directly relevant comparison for that audience. Install with:
#   pip install ten-vad
def run_ten_vad(test_speech: list, test_noise: list,
                speech_dir: str, noise_dir: str) -> dict:
    from ten_vad import TenVad
    import soundfile as sf
    import librosa

    HOP_SIZE  = 256   # 16ms frames at 16kHz, per TEN-VAD's recommended config
    THRESHOLD = 0.5    # TEN-VAD's own internal speech/non-speech probability cutoff

    def predict(path):
        try:
            audio, sr = sf.read(path, dtype="int16")
            if audio.ndim > 1:
                audio = audio[:, 0]
            if sr != 16000:
                # resample in float space, then back to int16 (TEN-VAD requires int16 frames)
                audio_f = audio.astype(np.float32) / 32768.0
                audio_f = librosa.resample(audio_f, orig_sr=sr, target_sr=16000)
                audio   = (audio_f * 32768.0).astype(np.int16)

            vad = TenVad(HOP_SIZE, THRESHOLD)
            frames = [
                audio[i:i + HOP_SIZE]
                for i in range(0, len(audio) - HOP_SIZE + 1, HOP_SIZE)
            ]
            if not frames:
                return 0

            speech_frames = sum(vad.process(f)[1] for f in frames)
            speech_ratio  = speech_frames / len(frames)
            # same 40% frame-ratio convention used for the WebRTC baseline,
            # so the two frame-based baselines are compared on equal footing
            return 1 if speech_ratio > 0.40 else 0
        except Exception as e:
            print(f"    Error on {path}: {e}")
            return 0

    results = []
    start   = time.time()

    for f in test_speech:
        path       = os.path.join(speech_dir, f)
        t0         = time.time()
        pred       = predict(path)
        latency_ms = (time.time() - t0) * 1000
        results.append({"true": 1, "pred": pred, "file": f, "latency_ms": round(latency_ms, 2)})

    for f in test_noise:
        path       = os.path.join(noise_dir, f)
        t0         = time.time()
        pred       = predict(path)
        latency_ms = (time.time() - t0) * 1000
        results.append({"true": 0, "pred": pred, "file": f, "latency_ms": round(latency_ms, 2)})

    elapsed = time.time() - start
    # TEN-VAD ships a tiny per-platform native library (a few hundred KB) —
    # no local model file path to size the way NOVA-VAD's .pkl files are sized
    return compute_metrics(results, elapsed, "TEN-VAD", model_size_bytes_val=None)


# ── Picovoice Cobra (stub — needs a commercial AccessKey) ──────────────────
# https://picovoice.ai/platform/cobra/ — requires signing up for a Picovoice
# Console account and generating a free/paid AccessKey. Not wired up yet
# because no key has been provided. Once you have one:
#   1. pip install pvcobra
#   2. export PICOVOICE_ACCESS_KEY="your-key-here"
#   3. this function will pick it up automatically — no other code changes needed
def run_picovoice_cobra(test_speech: list, test_noise: list,
                        speech_dir: str, noise_dir: str):
    access_key = os.environ.get("PICOVOICE_ACCESS_KEY")
    if not access_key:
        print("    Skipping Picovoice Cobra — set PICOVOICE_ACCESS_KEY to enable "
              "(requires a free AccessKey from https://console.picovoice.ai/).")
        return None

    try:
        import pvcobra
    except ImportError:
        print("    Skipping Picovoice Cobra — pvcobra not installed. Run: pip install pvcobra")
        return None

    import soundfile as sf
    import librosa

    cobra = pvcobra.create(access_key=access_key)
    frame_length = cobra.frame_length  # fixed by the SDK, 16kHz mono int16

    def predict(path):
        try:
            audio, sr = sf.read(path, dtype="int16")
            if audio.ndim > 1:
                audio = audio[:, 0]
            if sr != 16000:
                audio_f = audio.astype(np.float32) / 32768.0
                audio_f = librosa.resample(audio_f, orig_sr=sr, target_sr=16000)
                audio   = (audio_f * 32768.0).astype(np.int16)

            frames = [
                audio[i:i + frame_length]
                for i in range(0, len(audio) - frame_length + 1, frame_length)
            ]
            if not frames:
                return 0
            probs        = [cobra.process(f) for f in frames]
            speech_ratio = sum(1 for p in probs if p > 0.5) / len(probs)
            return 1 if speech_ratio > 0.40 else 0
        except Exception as e:
            print(f"    Error on {path}: {e}")
            return 0

    results = []
    start   = time.time()

    for f in test_speech:
        path       = os.path.join(speech_dir, f)
        t0         = time.time()
        pred       = predict(path)
        latency_ms = (time.time() - t0) * 1000
        results.append({"true": 1, "pred": pred, "file": f, "latency_ms": round(latency_ms, 2)})

    for f in test_noise:
        path       = os.path.join(noise_dir, f)
        t0         = time.time()
        pred       = predict(path)
        latency_ms = (time.time() - t0) * 1000
        results.append({"true": 0, "pred": pred, "file": f, "latency_ms": round(latency_ms, 2)})

    elapsed = time.time() - start
    cobra.delete()
    return compute_metrics(results, elapsed, "Picovoice Cobra", model_size_bytes_val=None)


# ── Energy Threshold VAD (naive, dependency-free baseline) ────────────────
def run_energy_threshold(test_speech: list, test_noise: list,
                         speech_dir: str, noise_dir: str) -> dict:
    """
    A trivial RMS-energy-threshold baseline. No ML, no extra dependencies
    beyond what's already installed (soundfile/numpy) — added as a lower
    bound so accuracy gains from NOVA-VAD's feature+ensemble approach are
    contextualized against the simplest possible reproducible baseline.
    """
    import soundfile as sf

    def predict(path):
        audio, sr = sf.read(path)
        if audio.ndim > 1:
            audio = audio[:, 0]
        audio = audio.astype(np.float32)
        if np.max(np.abs(audio)) > 1.5:
            audio = audio / 32768.0  # int16-range fallback
        rms = np.sqrt(np.mean(audio ** 2))
        # fixed empirical threshold — deliberately simple, not tuned per-dataset
        return 1 if rms > 0.02 else 0

    results = []
    start   = time.time()

    for f in test_speech:
        path       = os.path.join(speech_dir, f)
        t0         = time.time()
        pred       = predict(path)
        latency_ms = (time.time() - t0) * 1000
        results.append({"true": 1, "pred": pred, "file": f, "latency_ms": round(latency_ms, 2)})

    for f in test_noise:
        path       = os.path.join(noise_dir, f)
        t0         = time.time()
        pred       = predict(path)
        latency_ms = (time.time() - t0) * 1000
        results.append({"true": 0, "pred": pred, "file": f, "latency_ms": round(latency_ms, 2)})

    elapsed = time.time() - start
    return compute_metrics(results, elapsed, "Energy Threshold", model_size_bytes_val=0)


# ── Model size helper ───────────────────────────────────────────────────────
def model_size_bytes(paths: list) -> int:
    total = 0
    for p in paths:
        if os.path.exists(p):
            total += os.path.getsize(p)
    return total


def human_size(num_bytes) -> str:
    if num_bytes is None:
        return "N/A"
    if num_bytes == 0:
        return "0 B"
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.1f}{unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f}TB"


# ── Compute Metrics ────────────────────────────────────────────────────────
def compute_metrics(results: list, elapsed: float, name: str, model_size_bytes_val=None) -> dict:
    total   = len(results)
    correct = sum(1 for r in results if r["true"] == r["pred"])

    tp = sum(1 for r in results if r["true"] == 1 and r["pred"] == 1)
    tn = sum(1 for r in results if r["true"] == 0 and r["pred"] == 0)
    fp = sum(1 for r in results if r["true"] == 0 and r["pred"] == 1)
    fn = sum(1 for r in results if r["true"] == 1 and r["pred"] == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy  = correct / total * 100

    latencies    = [r["latency_ms"] for r in results if "latency_ms" in r]
    mean_latency = round(float(np.mean(latencies)), 2) if latencies else None
    p95_latency  = round(float(np.percentile(latencies, 95)), 2) if latencies else None

    false_positives = [r for r in results if r["true"] == 0 and r["pred"] == 1]
    false_negatives = [r for r in results if r["true"] == 1 and r["pred"] == 0]

    return {
        "name":      name,
        "total":     total,
        "accuracy":  round(accuracy, 2),
        "precision": round(precision * 100, 2),
        "recall":    round(recall * 100, 2),
        "f1":        round(f1 * 100, 2),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "time":      round(elapsed, 2),
        "mean_latency_ms": mean_latency,
        "p95_latency_ms":  p95_latency,
        "model_size_bytes": model_size_bytes_val,
        "model_size_human": human_size(model_size_bytes_val),
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


# ── Per-category breakdown (noise side only) ───────────────────────────────
def build_category_breakdown(raw_results: list, category_map: dict) -> dict:
    """
    raw_results: list of {"true": 0/1, "pred": 0/1, "file": ...} for noise
    files only. Returns {category: {"total": n, "correct": n, "accuracy": pct}}.
    """
    buckets = {}
    for r in raw_results:
        category = category_map.get(r["file"], "unknown")
        b = buckets.setdefault(category, {"total": 0, "correct": 0})
        b["total"]   += 1
        b["correct"] += 1 if r["true"] == r["pred"] else 0

    for category, b in buckets.items():
        b["accuracy"] = round(b["correct"] / b["total"] * 100, 2) if b["total"] else 0.0

    return buckets


# ── Print Results ──────────────────────────────────────────────────────────
def print_benchmark(results: list, n_test: int):
    print("\n" + "=" * 90)
    print("   NOVA-VAD HONEST BENCHMARK (held-out test set)")
    print("=" * 90)
    print(f"  Test files: {n_test} (never seen during training)\n")
    print(f"  {'Model':<20} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Time':>7} {'AvgLatency':>11} {'ModelSize':>10}")
    print("  " + "-" * 86)

    for r in results:
        latency_str = f"{r['mean_latency_ms']}ms" if r['mean_latency_ms'] is not None else "N/A"
        print(f"  {r['name']:<20} {r['accuracy']:>8}% {r['precision']:>9}% {r['recall']:>7}% {r['f1']:>7}% {r['time']:>5}s {latency_str:>11} {r['model_size_human']:>10}")

    print("=" * 90)

    nova     = next(r for r in results if r["name"] == "NOVA-VAD")
    webrtc   = next(r for r in results if r["name"] == "WebRTC VAD")
    silero   = next(r for r in results if r["name"] == "Silero VAD")
    pyannote = next((r for r in results if r["name"] == "Pyannote VAD"), None)

    print(f"\n  NOVA-VAD vs WebRTC:   {'+' if nova['accuracy'] >= webrtc['accuracy'] else ''}{round(nova['accuracy'] - webrtc['accuracy'], 2)}%")
    print(f"  NOVA-VAD vs Silero:   {'+' if nova['accuracy'] >= silero['accuracy'] else ''}{round(nova['accuracy'] - silero['accuracy'], 2)}%")
    if pyannote:
        print(f"  NOVA-VAD vs Pyannote:     {'+' if nova['accuracy'] >= pyannote['accuracy'] else ''}{round(nova['accuracy'] - pyannote['accuracy'], 2)}%")

    speechbrain = next((r for r in results if r["name"] == "SpeechBrain VAD"), None)
    if speechbrain:
        print(f"  NOVA-VAD vs SpeechBrain:  {'+' if nova['accuracy'] >= speechbrain['accuracy'] else ''}{round(nova['accuracy'] - speechbrain['accuracy'], 2)}%")

    energy = next((r for r in results if r["name"] == "Energy Threshold"), None)
    if energy:
        print(f"  NOVA-VAD vs EnergyThresh: {'+' if nova['accuracy'] >= energy['accuracy'] else ''}{round(nova['accuracy'] - energy['accuracy'], 2)}%")

    ten_vad = next((r for r in results if r["name"] == "TEN-VAD"), None)
    if ten_vad:
        print(f"  NOVA-VAD vs TEN-VAD:      {'+' if nova['accuracy'] >= ten_vad['accuracy'] else ''}{round(nova['accuracy'] - ten_vad['accuracy'], 2)}%")

    cobra = next((r for r in results if r["name"] == "Picovoice Cobra"), None)
    if cobra:
        print(f"  NOVA-VAD vs Picovoice Cobra: {'+' if nova['accuracy'] >= cobra['accuracy'] else ''}{round(nova['accuracy'] - cobra['accuracy'], 2)}%")

    print(f"\n  Explainability:")
    print(f"    WebRTC:           ❌ black box")
    print(f"    Silero:           ❌ black box")
    print(f"    Pyannote:         ❌ black box")
    print(f"    SpeechBrain:      ❌ black box")
    print(f"    TEN-VAD:          ❌ black box (probability score, no feature attribution)")
    if cobra:
        print(f"    Picovoice Cobra:  ❌ black box")
    print(f"    Energy Threshold: ⚠️  trivially explainable (single RMS number), not accurate")
    print(f"    NOVA-VAD:         ✅ confidence + feature breakdown")
    print(f"\n  Dependencies:")
    print(f"    WebRTC:           lightweight (C extension)")
    print(f"    Silero:           PyTorch required (200MB+)")
    print(f"    Pyannote:         PyTorch + pyannote.audio required (heavy)")
    print(f"    SpeechBrain:      PyTorch + speechbrain required (heavy)")
    print(f"    TEN-VAD:          lightweight (numpy + small native lib, no PyTorch)")
    if cobra:
        print(f"    Picovoice Cobra:  lightweight SDK, requires commercial AccessKey")
    print(f"    Energy Threshold: none beyond numpy/soundfile")
    print(f"    NOVA-VAD:         scikit-learn only (lightweight)")
    print("=" * 90)


def save_artifacts(results: list, n_test: int, category_breakdowns: dict = None):
    """
    Saves a JSON summary (metrics + false positive/negative file lists) and
    a plain-text FP/FN report to results/ so benchmark runs are reproducible
    and reviewable without re-running the benchmark.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    summary = {
        "timestamp": timestamp,
        "n_test_files": n_test,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "models": [],
    }

    for r in results:
        entry = {k: v for k, v in r.items() if k not in ("false_positives", "false_negatives")}
        entry["false_positive_files"] = [f["file"] for f in r.get("false_positives", [])]
        entry["false_negative_files"] = [f["file"] for f in r.get("false_negatives", [])]
        summary["models"].append(entry)

    if category_breakdowns:
        summary["nova_vad_noise_category_breakdown"] = category_breakdowns

    summary_path = os.path.join(RESULTS_DIR, f"benchmark_{timestamp}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    latest_path = os.path.join(RESULTS_DIR, "benchmark_latest.json")
    with open(latest_path, "w") as f:
        json.dump(summary, f, indent=2)

    # human-readable FP/FN report, focused on NOVA-VAD since that's the
    # model this repo ships and iterates on
    nova = next((r for r in results if r["name"] == "NOVA-VAD"), None)
    if nova:
        fpfn_path = os.path.join(RESULTS_DIR, "false_positives_negatives.txt")
        with open(fpfn_path, "w") as f:
            f.write(f"NOVA-VAD false positive / false negative report\n")
            f.write(f"Generated: {timestamp}\n")
            f.write(f"Test set size: {n_test}\n\n")

            f.write(f"FALSE POSITIVES ({len(nova['false_positives'])}) — noise misclassified as SPEECH\n")
            f.write("-" * 60 + "\n")
            for item in nova["false_positives"]:
                conf = item.get("confidence", "N/A")
                f.write(f"  {item['file']:<30} predicted=SPEECH actual=NO SPEECH confidence={conf}\n")

            f.write(f"\nFALSE NEGATIVES ({len(nova['false_negatives'])}) — speech misclassified as NO SPEECH\n")
            f.write("-" * 60 + "\n")
            for item in nova["false_negatives"]:
                conf = item.get("confidence", "N/A")
                f.write(f"  {item['file']:<30} predicted=NO SPEECH actual=SPEECH confidence={conf}\n")

        print(f"\n  False positive/negative report: {fpfn_path}")

    print(f"  Full benchmark JSON:            {summary_path}")
    print(f"  Latest benchmark JSON:          {latest_path}")


# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # NOTE: the benchmark deliberately evaluates every model (including
    # NOVA-VAD) on RAW, undenoised audio (data/speech, data/noise) rather
    # than the noisereduce-cleaned data/clean_speech, data/clean_noise
    # directories that src/pipeline.py produces.
    #
    # Root cause of a prior bug this fixes: denoiser.py builds each clip's
    # noise profile from *that same clip's* first 0.5s
    # (noise_sample = audio[:sr*0.5]) and runs noisereduce against it. For a
    # roughly-stationary UrbanSound8K noise clip (drilling, siren, AC hum,
    # engine idling, ...) the first 0.5s is representative of the whole
    # clip, so this profile ends up subtracting out most of the clip's own
    # energy — RMS drops ~67% on average across the noise set. For speech
    # clips the first 0.5s is often a quiet lead-in, not representative of
    # the louder voiced segments, so speech RMS only drops ~18% on average.
    # That asymmetry manufactures an artificial energy gap between the
    # classes that does not exist in the raw source audio (raw noise is
    # actually louder than raw speech on average in this dataset — 8.85%
    # mean RMS vs 6.91%). It's what let the naive Energy-Threshold baseline
    # hit 74% accuracy: measured on data/speech vs data/noise directly
    # (same 80/20 split), the same fixed 0.02 RMS threshold gets 52% — a
    # coin flip, as expected for a single-number heuristic on real-world
    # noise recordings.
    #
    # It's also the more honest choice architecturally: src/explainer.py and
    # src/stream.py — the actual inference entry points a user or
    # downstream integration calls — never run the denoiser. It only ever
    # ran as an offline data-prep step before training/eval. Benchmarking
    # against clean_speech/clean_noise was measuring performance on audio
    # nothing in real deployment ever sees.
    SPEECH_DIR = "data/speech"
    NOISE_DIR  = "data/noise"

    print("=" * 65)
    print("  NOVA-VAD BENCHMARK")
    print("=" * 65)

    # split dataset
    print("\n[ STEP 1 ] Splitting dataset 80/20...")
    train_speech, train_noise, test_speech, test_noise = split_dataset(
        SPEECH_DIR, NOISE_DIR
    )

    # train NOVA-VAD on training set only
    print("\n[ STEP 2 ] Training NOVA-VAD on training set...")
    rf, gbt, scaler = train_nova_vad(
        train_speech, train_noise, SPEECH_DIR, NOISE_DIR
    )
    print("  Training complete.")

    n_test = len(test_speech) + len(test_noise)

    print("\n[ STEP 3 ] Testing all models on held-out test set...")

    print("\n  Running WebRTC VAD...")
    webrtc_r = run_webrtc(test_speech, test_noise, SPEECH_DIR, NOISE_DIR)
    print(f"  Done — {webrtc_r['accuracy']}%")

    print("\n  Running Energy Threshold (naive baseline)...")
    energy_r = run_energy_threshold(test_speech, test_noise, SPEECH_DIR, NOISE_DIR)
    print(f"  Done — {energy_r['accuracy']}%")

    print("\n  Running NOVA-VAD...")
    nova_r = run_nova_vad(test_speech, test_noise, SPEECH_DIR, NOISE_DIR, rf, gbt, scaler)
    print(f"  Done — {nova_r['accuracy']}%")

    print("\n  Running Silero VAD...")
    silero_r = run_silero(test_speech, test_noise, SPEECH_DIR, NOISE_DIR)
    print(f"  Done — {silero_r['accuracy']}%")

    print("\n  Running Pyannote VAD...")
    pyannote_r = run_pyannote(test_speech, test_noise, SPEECH_DIR, NOISE_DIR)
    print(f"  Done — {pyannote_r['accuracy']}%")

    print("\n  Running SpeechBrain VAD...")
    speechbrain_r = run_speechbrain(test_speech, test_noise, SPEECH_DIR, NOISE_DIR)
    print(f"  Done — {speechbrain_r['accuracy']}%")

    print("\n  Running TEN-VAD...")
    ten_vad_r = run_ten_vad(test_speech, test_noise, SPEECH_DIR, NOISE_DIR)
    print(f"  Done — {ten_vad_r['accuracy']}%")

    print("\n  Running Picovoice Cobra (skipped unless PICOVOICE_ACCESS_KEY is set)...")
    cobra_r = run_picovoice_cobra(test_speech, test_noise, SPEECH_DIR, NOISE_DIR)
    if cobra_r:
        print(f"  Done — {cobra_r['accuracy']}%")

    all_results = [webrtc_r, energy_r, nova_r, silero_r, pyannote_r, speechbrain_r, ten_vad_r]
    if cobra_r:
        all_results.append(cobra_r)
    print_benchmark(all_results, n_test)

    # per-noise-category breakdown, if download_noise.py's manifest exists
    category_map = load_noise_category_manifest("data/noise")
    category_breakdowns = {}
    if category_map:
        # re-derive raw per-file noise results for NOVA-VAD from the fp/fn
        # lists plus correct predictions — rebuild full noise-only result list
        noise_results = []
        nova_fp_files = {f["file"] for f in nova_r["false_positives"]}
        for f in test_noise:
            pred = 1 if f in nova_fp_files else 0
            noise_results.append({"true": 0, "pred": pred, "file": f})
        category_breakdowns = build_category_breakdown(noise_results, category_map)

        print("\n  NOVA-VAD accuracy by noise category:")
        for cat, stats in sorted(category_breakdowns.items()):
            print(f"    {cat:<20} {stats['correct']}/{stats['total']} ({stats['accuracy']}%)")

    save_artifacts(all_results, n_test, category_breakdowns)
