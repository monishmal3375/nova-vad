import os
import numpy as np
import librosa
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib

def extract_features(file_path: str) -> np.ndarray:
    """
    Extracts 106 temporal and spectral features from a .wav file.
    Captures both WHAT audio sounds like and HOW it changes over time.
    """
    audio, sr = librosa.load(file_path, sr=16000, mono=True)

    # guard against very short clips
    if len(audio) < sr * 0.1:
        audio = np.pad(audio, (0, int(sr * 0.1) - len(audio)))

    features = []

    # ── 1. MFCCs (13 coefficients + deltas) ───────────────────────────────
    mfcc        = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    # width must be odd and less than number of frames
    delta_width = min(9, mfcc.shape[1] if mfcc.shape[1] % 2 != 0 else mfcc.shape[1] - 1)
    delta_width = max(3, delta_width)
    mfcc_delta  = librosa.feature.delta(mfcc, width=delta_width)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2, width=delta_width)

    for coef in [mfcc, mfcc_delta, mfcc_delta2]:
        features.extend([
            np.mean(coef, axis=1),
            np.std(coef, axis=1),
        ])

    # ── 2. Zero Crossing Rate ──────────────────────────────────────────────
    # speech has low ZCR, noise is chaotic
    zcr = librosa.feature.zero_crossing_rate(audio)[0]
    features.append([np.mean(zcr), np.std(zcr), np.max(zcr), np.min(zcr)])

    # ── 3. RMS Energy ─────────────────────────────────────────────────────
    # speech rises and falls rhythmically
    rms = librosa.feature.rms(y=audio)[0]
    features.append([
        np.mean(rms), np.std(rms),
        np.max(rms),  np.min(rms),
        np.mean(np.diff(rms))  # energy change rate
    ])

    # ── 4. Spectral Centroid ───────────────────────────────────────────────
    # where the "center of mass" of the spectrum sits
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    features.append([np.mean(centroid), np.std(centroid)])

    # ── 5. Spectral Rolloff ────────────────────────────────────────────────
    # frequency below which 85% of energy sits
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
    features.append([np.mean(rolloff), np.std(rolloff)])

    # ── 6. Spectral Flux ──────────────────────────────────────────────────
    # how fast spectrum changes frame to frame
    # speech changes smoothly, noise changes randomly
    stft = np.abs(librosa.stft(audio))
    flux = np.sqrt(np.sum(np.diff(stft, axis=1) ** 2, axis=0))
    features.append([np.mean(flux), np.std(flux)])

    # ── 7. Spectral Bandwidth ─────────────────────────────────────────────
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
    features.append([np.mean(bandwidth), np.std(bandwidth)])

    # ── 8. Chroma Features ────────────────────────────────────────────────
    # pitch class information — speech has consistent pitch patterns
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    features.append([np.mean(chroma), np.std(chroma)])

    # ── 9. Mel Spectrogram Stats ──────────────────────────────────────────
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=40)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    features.append([
        np.mean(mel_db), np.std(mel_db),
        np.max(mel_db),  np.min(mel_db)
    ])

    # ── 10. Tempo and Rhythm ──────────────────────────────────────────────
    # speech has syllable rhythm (~4-8 Hz) that noise doesn't
    tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
    tempo_val = float(np.atleast_1d(tempo)[0])
    features.append([tempo_val])
    # ── 11. Harmonic vs Percussive ratio ──────────────────────────────────
    # speech is mostly harmonic, noise is percussive
    harmonic, percussive = librosa.effects.hpss(audio)
    h_energy = np.mean(harmonic ** 2)
    p_energy = np.mean(percussive ** 2)
    ratio = h_energy / (p_energy + 1e-10)
    features.append([h_energy, p_energy, ratio])

    # ── 12. Silence ratio ─────────────────────────────────────────────────
    # proportion of frames below energy threshold
    threshold  = 0.01 * np.max(np.abs(audio))
    silence_ratio = np.mean(np.abs(audio) < threshold)
    features.append([silence_ratio])

    # flatten everything into one vector
    flat = []
    for f in features:
        if isinstance(f, np.ndarray):
            flat.extend(f.flatten().tolist())
        else:
            flat.extend(f)

    return np.array(flat)


def build_dataset(speech_dir: str, noise_dir: str):
    """
    Builds feature matrix X and label vector y.
    """
    X, y, filenames = [], [], []

    print("Extracting features from speech files...")
    speech_files = sorted([f for f in os.listdir(speech_dir) if f.endswith(".wav")])
    for i, f in enumerate(speech_files):
        path     = os.path.join(speech_dir, f)
        features = extract_features(path)
        X.append(features)
        y.append(1)
        filenames.append(f)
        if (i + 1) % 50 == 0:
            print(f"  ✓ {i+1}/{len(speech_files)} speech files processed")

    print("\nExtracting features from noise files...")
    noise_files = sorted([f for f in os.listdir(noise_dir) if f.endswith(".wav")])
    for i, f in enumerate(noise_files):
        path     = os.path.join(noise_dir, f)
        features = extract_features(path)
        X.append(features)
        y.append(0)
        filenames.append(f)
        if (i + 1) % 50 == 0:
            print(f"  ✓ {i+1}/{len(noise_files)} noise files processed")

    return np.array(X), np.array(y), filenames


def train_and_evaluate(speech_dir: str, noise_dir: str) -> dict:
    """
    Trains an ensemble classifier using 5-fold cross validation.
    Works well for both small AND large datasets.
    """
    print("\n[ NOVA-VAD CLASSIFIER ]\n")
    X, y, filenames = build_dataset(speech_dir, noise_dir)

    print(f"\nFeature vector size: {X.shape[1]} features per file")
    print(f"Dataset size: {X.shape[0]} files\n")

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # use StratifiedKFold for larger datasets, LOO for small
    if len(y) <= 100:
        print("Small dataset detected — using Leave-One-Out CV")
        cv = LeaveOneOut()
        splits = list(cv.split(X_scaled))
    else:
        print("Large dataset detected — using 5-Fold Stratified CV")
        cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        splits = list(cv.split(X_scaled, y))

    # ensemble: Random Forest + Gradient Boosting
    rf  = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    gbt = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

    predictions  = np.zeros(len(y), dtype=int)
    true_labels  = np.zeros(len(y), dtype=int)

    print("Running cross validation...\n")
    for fold, (train_idx, test_idx) in enumerate(splits):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # train both models
        rf.fit(X_train, y_train)
        gbt.fit(X_train, y_train)

        # ensemble prediction — average probabilities
        rf_probs  = rf.predict_proba(X_test)[:, 1]
        gbt_probs = gbt.predict_proba(X_test)[:, 1]
        avg_probs = (rf_probs + gbt_probs) / 2
        preds     = (avg_probs > 0.5).astype(int)

        predictions[test_idx] = preds
        true_labels[test_idx] = y_test

        if len(splits) <= 100:
            filename   = filenames[test_idx[0]]
            true_label = "SPEECH" if y_test[0] == 1 else "NO SPEECH"
            pred_label = "SPEECH" if preds[0] == 1 else "NO SPEECH"
            status     = "✓" if preds[0] == y_test[0] else "✗"
            print(f"  {status} {filename} → {pred_label} | actual: {true_label}")
        else:
            correct_in_fold = np.sum(preds == y_test)
            fold_acc        = correct_in_fold / len(y_test) * 100
            print(f"  Fold {fold+1}: {correct_in_fold}/{len(y_test)} correct ({fold_acc:.1f}%)")

    # metrics
    total    = len(y)
    correct  = np.sum(predictions == true_labels)
    accuracy = correct / total * 100

    tp = np.sum((predictions == 1) & (true_labels == 1))
    tn = np.sum((predictions == 0) & (true_labels == 0))
    fp = np.sum((predictions == 1) & (true_labels == 0))
    fn = np.sum((predictions == 0) & (true_labels == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # save final model trained on ALL data
    rf.fit(X_scaled, y)
    gbt.fit(X_scaled, y)
    os.makedirs("models", exist_ok=True)
    joblib.dump(rf,     "models/nova_vad_rf.pkl")
    joblib.dump(gbt,    "models/nova_vad_gbt.pkl")
    joblib.dump(scaler, "models/nova_vad_scaler.pkl")
    print("\n✅ NOVA-VAD models saved to models/")

    return {
        "total":     int(total),
        "correct":   int(correct),
        "accuracy":  round(accuracy, 2),
        "precision": round(precision * 100, 2),
        "recall":    round(recall * 100, 2),
        "f1_score":  round(f1 * 100, 2),
        "tp": int(tp), "tn": int(tn),
        "fp": int(fp), "fn": int(fn)
    }

def extract_features_from_array(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Same as extract_features but takes a numpy array directly
    instead of a file path. Used for real-time streaming.
    """
    if len(audio) < sr * 0.1:
        audio = np.pad(audio, (0, int(sr * 0.1) - len(audio)))

    features = []

    mfcc        = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    delta_width = min(9, mfcc.shape[1] if mfcc.shape[1] % 2 != 0 else mfcc.shape[1] - 1)
    delta_width = max(3, delta_width)
    mfcc_delta  = librosa.feature.delta(mfcc, width=delta_width)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2, width=delta_width)

    for coef in [mfcc, mfcc_delta, mfcc_delta2]:
        features.extend([
            np.mean(coef, axis=1),
            np.std(coef, axis=1),
        ])

    zcr = librosa.feature.zero_crossing_rate(audio)[0]
    features.append([np.mean(zcr), np.std(zcr), np.max(zcr), np.min(zcr)])

    rms = librosa.feature.rms(y=audio)[0]
    features.append([
        np.mean(rms), np.std(rms),
        np.max(rms),  np.min(rms),
        np.mean(np.diff(rms))
    ])

    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    features.append([np.mean(centroid), np.std(centroid)])

    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
    features.append([np.mean(rolloff), np.std(rolloff)])

    stft = np.abs(librosa.stft(audio))
    flux = np.sqrt(np.sum(np.diff(stft, axis=1) ** 2, axis=0))
    features.append([np.mean(flux), np.std(flux)])

    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
    features.append([np.mean(bandwidth), np.std(bandwidth)])

    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    features.append([np.mean(chroma), np.std(chroma)])

    mel    = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=40)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    features.append([
        np.mean(mel_db), np.std(mel_db),
        np.max(mel_db),  np.min(mel_db)
    ])

    tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
    tempo_val = float(np.atleast_1d(tempo)[0])
    features.append([tempo_val])

    harmonic, percussive = librosa.effects.hpss(audio)
    h_energy = np.mean(harmonic ** 2)
    p_energy = np.mean(percussive ** 2)
    ratio    = h_energy / (p_energy + 1e-10)
    features.append([h_energy, p_energy, ratio])

    threshold     = 0.01 * np.max(np.abs(audio))
    silence_ratio = np.mean(np.abs(audio) < threshold)
    features.append([silence_ratio])

    flat = []
    for f in features:
        if isinstance(f, np.ndarray):
            flat.extend(f.flatten().tolist())
        else:
            flat.extend(f)

    return np.array(flat)

