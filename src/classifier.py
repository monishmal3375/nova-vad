import os
from pyexpat import features
import numpy as np
import librosa
from scipy.signal import find_peaks
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib


def _spectral_entropy_stats(power_spec: np.ndarray) -> tuple:
    """
    Per-frame spectral entropy (Shannon entropy of the normalized power
    spectrum). Voiced speech concentrates energy in a few formant/harmonic
    bins (low entropy); most environmental noise spreads energy more
    uniformly across the spectrum (high entropy). Cheap (no filterbank/
    fitting), and reported in the noise-robust-VAD literature as holding up
    better than raw energy features at low SNR.
    """
    norm = power_spec / (np.sum(power_spec, axis=0, keepdims=True) + 1e-12)
    entropy = -np.sum(norm * np.log2(norm + 1e-12), axis=0)
    return float(np.mean(entropy)), float(np.std(entropy))


def _harmonic_peak_prominence_stats(power_spec: np.ndarray) -> tuple:
    """
    Per-frame mean peak prominence in the dB spectrum — how far harmonic
    peaks stand above their local surrounding floor. Distinct from the
    existing time-domain harmonic/percussive (HPSS) ratio: this measures
    peak-vs-local-noise-floor energy gap directly in the spectrum, which
    tends to survive additive background noise better than a global
    harmonic/percussive energy split.
    """
    prominences = []
    for frame in power_spec.T:
        frame_db = librosa.power_to_db(frame + 1e-12)
        peaks, props = find_peaks(frame_db, prominence=1.0)
        prominences.append(float(np.mean(props["prominences"])) if len(peaks) else 0.0)
    prominences = np.array(prominences)
    return float(np.mean(prominences)), float(np.std(prominences))


def _voiced_fraction(audio: np.ndarray, sr: int, fmin: float = 65, fmax: float = 400,
                      frame_length: int = 1024, hop_length: int = 256,
                      strength_threshold: float = 0.35) -> float:
    """
    Fraction of frames with strong periodicity in the human pitch range, via
    normalized autocorrelation peak strength per frame (NOT librosa.yin's
    raw F0 estimate — yin always returns *some* frequency even for pure
    noise, so a naive "is yin's output in range" check is always true and
    carries no signal). This is a cheap, real voiced/unvoiced proxy: high
    autocorrelation at a plausible pitch-period lag means the frame is
    periodic (voiced speech); low means it's aperiodic (most noise).
    """
    if len(audio) < frame_length:
        return 0.0
    frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
    lag_min = max(1, int(sr / fmax))
    lag_max = min(frame_length - 1, int(sr / fmin))
    if lag_max <= lag_min:
        return 0.0

    voiced_count = 0
    for i in range(frames.shape[1]):
        frame = frames[:, i] - np.mean(frames[:, i])
        energy = np.dot(frame, frame)
        if energy < 1e-9:
            continue
        ac = np.correlate(frame, frame, mode="full")[frame_length - 1:]
        ac_norm = ac / (ac[0] + 1e-12)
        if np.max(ac_norm[lag_min:lag_max]) > strength_threshold:
            voiced_count += 1
    return float(voiced_count) / frames.shape[1]

def extract_features(file_path: str) -> np.ndarray:
    """
    Extracts 150+ temporal and spectral features from a .wav file.
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
    # NOTE: mel_db's max is always exactly 0dB by construction (ref=np.max
    # normalizes relative to the clip's own peak) — it carries zero signal
    # and was previously a constant, dead feature. log1p(linear power max)
    # instead captures real absolute-energy-level information.
    features.append([
        np.mean(mel_db), np.std(mel_db),
        float(np.log1p(np.max(mel))), np.min(mel_db)
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

    # ── 13. Pitch / voicing (F0 via YIN + autocorrelation voicing) ─────────
    # Human speech has a fundamental frequency in a fairly narrow band
    # (~70-400Hz) and long voiced runs; most environmental noise (engines,
    # drilling, traffic, wind) either has no clear periodicity or sits
    # outside that band. YIN (not the much slower probabilistic pYIN) keeps
    # F0 estimation a ~15-20ms operation so it doesn't blow up inference
    # latency. Note: yin always returns *some* frequency estimate even for
    # pure noise (unlike pyin, it has no built-in "unvoiced" output), so
    # voiced fraction is computed separately below via autocorrelation
    # periodicity strength rather than trusting yin's raw output range.
    f0 = librosa.yin(audio, fmin=65, fmax=400, sr=sr, frame_length=1024)
    f0_mean  = float(np.mean(f0))
    f0_std   = float(np.std(f0))
    f0_range = float(np.max(f0) - np.min(f0))
    voiced_frac = _voiced_fraction(audio, sr)
    features.append([f0_mean, f0_std, f0_range, voiced_frac])

    # ── 14. Spectral Contrast ──────────────────────────────────────────────
    # difference between peaks and valleys across sub-bands — voiced speech
    # (harmonics standing out of the noise floor) tends to show higher
    # contrast than diffuse environmental noise.
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    features.append([np.mean(contrast), np.std(contrast)])

    # ── 15. Spectral Flatness ──────────────────────────────────────────────
    # ~1.0 for noise-like/white-noise spectra, much lower for tonal/
    # harmonic signals like voiced speech. Cheap and complementary to
    # harmonic/percussive ratio.
    flatness = librosa.feature.spectral_flatness(y=audio)[0]
    features.append([np.mean(flatness), np.std(flatness)])

    # ── 16. Spectral Entropy ────────────────────────────────────────────────
    # low for voiced speech (energy concentrated in formants/harmonics),
    # high for diffuse environmental noise. Reuses the STFT already
    # computed above for spectral flux.
    power_spec = stft ** 2
    entropy_mean, entropy_std = _spectral_entropy_stats(power_spec)
    features.append([entropy_mean, entropy_std])

    # ── 17. Harmonic Peak Prominence ────────────────────────────────────────
    # how far harmonic peaks stand above the local spectral floor —
    # survives additive background noise better than a global harmonic/
    # percussive energy split (feature #11 above).
    peak_prom_mean, peak_prom_std = _harmonic_peak_prominence_stats(power_spec)
    features.append([peak_prom_mean, peak_prom_std])

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

    f0 = librosa.yin(audio, fmin=65, fmax=400, sr=sr, frame_length=1024)
    f0_mean  = float(np.mean(f0))
    f0_std   = float(np.std(f0))
    f0_range = float(np.max(f0) - np.min(f0))
    voiced_frac = _voiced_fraction(audio, sr)
    features.append([f0_mean, f0_std, f0_range, voiced_frac])

    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    features.append([np.mean(contrast), np.std(contrast)])

    flatness = librosa.feature.spectral_flatness(y=audio)[0]
    features.append([np.mean(flatness), np.std(flatness)])

    power_spec = stft ** 2
    entropy_mean, entropy_std = _spectral_entropy_stats(power_spec)
    features.append([entropy_mean, entropy_std])

    peak_prom_mean, peak_prom_std = _harmonic_peak_prominence_stats(power_spec)
    features.append([peak_prom_mean, peak_prom_std])

    flat = []
    for f in features:
        if isinstance(f, np.ndarray):
            flat.extend(f.flatten().tolist())
        else:
            flat.extend(f)

    return np.array(flat)

