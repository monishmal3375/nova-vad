import os
import time
import random
import shutil
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from src.classifier import extract_features, build_dataset
from src.vad import detect_speech
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

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
        path     = os.path.join(speech_dir, f)
        features = extract_features(path)
        X_scaled = scaler.transform([features])
        rf_prob  = rf.predict_proba(X_scaled)[0][1]
        gbt_prob = gbt.predict_proba(X_scaled)[0][1]
        pred     = 1 if (rf_prob + gbt_prob) / 2 > 0.5 else 0
        results.append({"true": 1, "pred": pred, "file": f})

    for f in test_noise:
        path     = os.path.join(noise_dir, f)
        features = extract_features(path)
        X_scaled = scaler.transform([features])
        rf_prob  = rf.predict_proba(X_scaled)[0][1]
        gbt_prob = gbt.predict_proba(X_scaled)[0][1]
        pred     = 1 if (rf_prob + gbt_prob) / 2 > 0.5 else 0
        results.append({"true": 0, "pred": pred, "file": f})

    elapsed = time.time() - start
    return compute_metrics(results, elapsed, "NOVA-VAD")


# ── Run WebRTC on test set ─────────────────────────────────────────────────
def run_webrtc(test_speech: list, test_noise: list,
               speech_dir: str, noise_dir: str) -> dict:
    results = []
    start   = time.time()

    for f in test_speech:
        path = os.path.join(speech_dir, f)
        r    = detect_speech(path)
        results.append({"true": 1, "pred": r["prediction"], "file": f})

    for f in test_noise:
        path = os.path.join(noise_dir, f)
        r    = detect_speech(path)
        results.append({"true": 0, "pred": r["prediction"], "file": f})

    elapsed = time.time() - start
    return compute_metrics(results, elapsed, "WebRTC VAD")


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
        pred = predict(os.path.join(speech_dir, f))
        results.append({"true": 1, "pred": pred, "file": f})

    for f in test_noise:
        pred = predict(os.path.join(noise_dir, f))
        results.append({"true": 0, "pred": pred, "file": f})

    elapsed = time.time() - start
    return compute_metrics(results, elapsed, "Silero VAD")

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
        pred = predict(os.path.join(speech_dir, f))
        results.append({"true": 1, "pred": pred, "file": f})

    for f in test_noise:
        pred = predict(os.path.join(noise_dir, f))
        results.append({"true": 0, "pred": pred, "file": f})

    elapsed = time.time() - start
    return compute_metrics(results, elapsed, "Pyannote VAD")


# ── Compute Metrics ────────────────────────────────────────────────────────
def compute_metrics(results: list, elapsed: float, name: str) -> dict:
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

    return {
        "name":      name,
        "total":     total,
        "accuracy":  round(accuracy, 2),
        "precision": round(precision * 100, 2),
        "recall":    round(recall * 100, 2),
        "f1":        round(f1 * 100, 2),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "time":      round(elapsed, 2)
    }


# ── Print Results ──────────────────────────────────────────────────────────
def print_benchmark(results: list, n_test: int):
    print("\n" + "=" * 65)
    print("   NOVA-VAD HONEST BENCHMARK (held-out test set)")
    print("=" * 65)
    print(f"  Test files: {n_test} (never seen during training)\n")
    print(f"  {'Model':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Time':>7}")
    print("  " + "-" * 61)

    for r in results:
        print(f"  {r['name']:<20} {r['accuracy']:>9}% {r['precision']:>9}% {r['recall']:>7}% {r['f1']:>7}% {r['time']:>5}s")

    print("=" * 65)

    nova    = next(r for r in results if r["name"] == "NOVA-VAD")
    webrtc  = next(r for r in results if r["name"] == "WebRTC VAD")
    silero  = next(r for r in results if r["name"] == "Silero VAD")
    pyannote = next((r for r in results if r["name"] == "Pyannote VAD"), None)

    print(f"\n  NOVA-VAD vs WebRTC:   {'+' if nova['accuracy'] >= webrtc['accuracy'] else ''}{round(nova['accuracy'] - webrtc['accuracy'], 2)}%")
    print(f"  NOVA-VAD vs Silero:   {'+' if nova['accuracy'] >= silero['accuracy'] else ''}{round(nova['accuracy'] - silero['accuracy'], 2)}%")
    if pyannote:
        print(f"  NOVA-VAD vs Pyannote: {'+' if nova['accuracy'] >= pyannote['accuracy'] else ''}{round(nova['accuracy'] - pyannote['accuracy'], 2)}%")
    print(f"\n  Explainability:")
    print(f"    WebRTC:   ❌ black box")
    print(f"    Silero:   ❌ black box")
    print(f"    NOVA-VAD: ✅ confidence + feature breakdown")
    print(f"\n  Dependencies:")
    print(f"    WebRTC:   lightweight")
    print(f"    Silero:   PyTorch required (200MB+)")
    print(f"    NOVA-VAD: scikit-learn only (lightweight)")
    print("=" * 65)


# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    CLEAN_SPEECH = "data/clean_speech"
    CLEAN_NOISE  = "data/clean_noise"

    print("=" * 65)
    print("  NOVA-VAD BENCHMARK")
    print("=" * 65)

    # split dataset
    print("\n[ STEP 1 ] Splitting dataset 80/20...")
    train_speech, train_noise, test_speech, test_noise = split_dataset(
        CLEAN_SPEECH, CLEAN_NOISE
    )

    # train NOVA-VAD on training set only
    print("\n[ STEP 2 ] Training NOVA-VAD on training set...")
    rf, gbt, scaler = train_nova_vad(
        train_speech, train_noise, CLEAN_SPEECH, CLEAN_NOISE
    )
    print("  Training complete.")

    n_test = len(test_speech) + len(test_noise)

    print("\n[ STEP 3 ] Testing all models on held-out test set...")

    print("\n  Running WebRTC VAD...")
    webrtc_r = run_webrtc(test_speech, test_noise, CLEAN_SPEECH, CLEAN_NOISE)
    print(f"  Done — {webrtc_r['accuracy']}%")

    print("\n  Running NOVA-VAD...")
    nova_r = run_nova_vad(test_speech, test_noise, CLEAN_SPEECH, CLEAN_NOISE, rf, gbt, scaler)
    print(f"  Done — {nova_r['accuracy']}%")

    print("\n  Running Silero VAD...")
    silero_r = run_silero(test_speech, test_noise, CLEAN_SPEECH, CLEAN_NOISE)
    print(f"  Done — {silero_r['accuracy']}%")

    print("\n  Running Pyannote VAD...")
    pyannote_r = run_pyannote(test_speech, test_noise, CLEAN_SPEECH, CLEAN_NOISE)
    print(f"  Done — {pyannote_r['accuracy']}%")

    print_benchmark([webrtc_r, nova_r, silero_r, pyannote_r], n_test)