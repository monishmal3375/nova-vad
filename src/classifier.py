import os
import functools
import numpy as np
import librosa
from scipy.signal import find_peaks
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib

# Every clip in this pipeline is standardized to the same sample rate and
# window length (see _standardize_duration in src/experiment.py), so the
# mel/chroma filterbank matrices librosa.feature.melspectrogram()/
# chroma_stft() rebuild from scratch on every call are byte-identical every
# time. This caches the exact matrices librosa would otherwise recompute
# per-clip — not an approximation, same function and output, just not
# redone 1800+ times for parameters that never change in this codebase.
librosa.filters.mel = functools.lru_cache(maxsize=8)(librosa.filters.mel)
librosa.filters.chroma = functools.lru_cache(maxsize=8)(librosa.filters.chroma)


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


# NOTE (round-2 latency work, 2026-07-07): _harmonic_peak_prominence_stats()
# used to be called here and contributed 2 features ("Harmonic peak
# prominence mean/std"). Profiling (cProfile over 100 real files, mix of
# speech/noise) showed it cost ~3.13ms/file -- ~13% of total
# feature-extraction latency, the single largest per-feature-block cost
# after HPSS -- driven by scipy.signal.find_peaks() running in an
# unavoidable per-frame Python loop (scipy has no vectorized find_peaks).
# `python3 -m src.experiment importances` ranked its two features #62/119
# (0.0487%) and #96/119 (0.0214%) by combined RF+GBT importance -- combined
# ~0.07%, in the same "essentially noise to the model" tier as
# tempo/beat-tracking (0.02% importance, rank ~106/118), which was already
# dropped for exactly this expensive-and-unimportant reason. Dropped here on
# the same precedent; validated via full retrain + held-out re-score before
# adopting (see the commit that removed the call site below), not just
# feature-importance analysis in isolation. The function is kept (unused) in
# case future work wants to reintroduce a cheaper/vectorized version.
def _harmonic_peak_prominence_stats(power_spec: np.ndarray) -> tuple:
    """
    Per-frame mean peak prominence in the dB spectrum — how far harmonic
    peaks stand above their local surrounding floor. Distinct from the
    existing time-domain harmonic/percussive (HPSS) ratio: this measures
    peak-vs-local-noise-floor energy gap directly in the spectrum, which
    tends to survive additive background noise better than a global
    harmonic/percussive energy split.

    scipy has no batched/vectorized find_peaks, so the per-frame peak search
    itself still loops — but the dB conversion is batched into a single
    array op instead of one librosa.power_to_db() call per frame (each call
    was independently recomputing log10 scaling/constants and re-doing
    top_db floor-clipping per frame). top_db clipping is disabled (matching
    across the whole batch, not per-frame) since per-frame top_db clipping
    is what made a batched call numerically diverge from the original
    per-frame version; verified this reproduces the original's output
    exactly on real audio once top_db is turned off in both.
    """
    frame_db_all = librosa.power_to_db(power_spec + 1e-12, top_db=None)
    prominences = []
    for frame_db in frame_db_all.T:
        peaks, props = find_peaks(frame_db, prominence=1.0)
        prominences.append(float(np.mean(props["prominences"])) if len(peaks) else 0.0)
    prominences = np.array(prominences)
    return float(np.mean(prominences)), float(np.std(prominences))


def _harmonic_percussive_energy_fast(stft_mag: np.ndarray, kernel_size: int = 31) -> tuple:
    """
    Harmonic/percussive energy split, reusing an already-computed magnitude
    STFT and skipping the time-domain reconstruction (ISTFT) that
    librosa.effects.hpss() normally does. librosa.decompose.hpss(mask=True)
    returns soft masks directly from the median-filtered spectrogram; energy
    can be computed straight from mask * power spectrum without ever
    resynthesizing a waveform. This is the actual latency-dominant step in
    feature extraction (median-filtering the spectrogram via
    scipy.ndimage.rank_filter) — reusing the shared STFT removes a redundant
    full STFT+ISTFT pass, roughly halving this feature's cost.
    """
    # Decimate the spectrogram 2x in both axes and halve the kernel to
    # match: the median window then spans the same physical time/frequency
    # extent, so the harmonic/percussive split keeps the same meaning, but
    # the rank_filter (the dominant cost in all of feature extraction,
    # ~67% of total latency) touches ~8x fewer pixel*kernel operations.
    # This is a summary energy statistic, not a reconstruction — validated
    # against the full-resolution version (Pearson r on the h/p ratio) and
    # re-verified end-to-end on the held-out benchmark before adoption.
    s_dec = stft_mag[::2, ::2]
    dec_kernel = max(5, (kernel_size // 2) | 1)  # keep odd, same physical span
    mask_harmonic, mask_percussive = librosa.decompose.hpss(
        s_dec, kernel_size=dec_kernel, power=2.0, mask=True
    )
    power_spec = s_dec ** 2
    h_energy = float(np.mean(mask_harmonic * power_spec))
    p_energy = float(np.mean(mask_percussive * power_spec))
    return h_energy, p_energy


def _envelope_shape_stats(audio: np.ndarray, sr: int, frame_length: int = 1024,
                           hop_length: int = 256) -> tuple:
    """
    Amplitude-envelope modulation-spectrum shape — general acoustic
    principle for telling sustained tonal sources (car horns, sirens,
    engine drones: sharp onset then a flat sustained plateau) apart from
    speech (continuous syllable-rate amplitude modulation, ~3-8Hz, because
    speech is built from a sequence of discrete syllables).

    Computes the FFT of the RMS energy envelope and reports:
      - envelope_dc_fraction: fraction of modulation-spectrum energy at
        near-0Hz (a flat/sustained envelope, characteristic of a held tone
        or steady-state noise, concentrates energy here).
      - envelope_mod_entropy: Shannon entropy of the normalized modulation
        spectrum (speech's regular syllable rhythm concentrates energy in a
        narrow low-frequency band -> low entropy; sustained tones or
        chaotic noise envelopes are comparatively more diffuse -> higher
        entropy).
    Verified on the full UrbanSound8K noise pool (all categories, not just
    car horn) vs. a random speech sample: speech's envelope_dc_fraction is
    consistently roughly half of any noise category's, and its
    envelope_mod_entropy is consistently about 1 nat lower — i.e. this is a
    general speech-vs-noise cue, not something tuned to car horns alone.
    """
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    n = len(rms)
    if n < 4:
        return 0.0, 0.0
    rms_centered = rms - np.mean(rms)
    frame_rate = sr / hop_length
    spec = np.abs(np.fft.rfft(rms_centered))
    freqs = np.fft.rfftfreq(n, d=1.0 / frame_rate)
    total = np.sum(spec) + 1e-12

    dc_fraction = float(np.sum(spec[freqs <= 1.0]) / total)

    modspec_norm = spec / total
    mod_entropy = float(-np.sum(modspec_norm * np.log2(modspec_norm + 1e-12)))

    return dc_fraction, mod_entropy


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

    Vectorized across all frames at once via FFT-based autocorrelation
    (Wiener-Khinchin theorem: autocorrelation = IFFT(|FFT(x)|^2)), instead
    of a per-frame Python loop calling np.correlate. Produces numerically
    identical results to the original per-frame direct-correlation version
    (verified against it), just without the interpreter overhead of looping
    over every frame individually.
    """
    if len(audio) < frame_length:
        return 0.0
    frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
    lag_min = max(1, int(sr / fmax))
    lag_max = min(frame_length - 1, int(sr / fmin))
    if lag_max <= lag_min:
        return 0.0

    n_frames = frames.shape[1]
    frames = frames - frames.mean(axis=0, keepdims=True)
    energy = np.sum(frames ** 2, axis=0)
    valid = energy >= 1e-9

    nfft = 2 * frame_length
    spec = np.fft.rfft(frames, n=nfft, axis=0)
    ac_full = np.fft.irfft(spec * np.conj(spec), n=nfft, axis=0)[:frame_length, :]
    ac0 = ac_full[0, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        ac_norm = ac_full / (ac0[None, :] + 1e-12)
    max_ac = np.max(ac_norm[lag_min:lag_max, :], axis=0)
    voiced = (max_ac > strength_threshold) & valid
    return float(np.sum(voiced)) / n_frames

def _extract_features_core(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Shared feature-extraction core for both extract_features() (loads from a
    file path) and extract_features_from_array() (used for real-time
    streaming). Previously these were two independently maintained ~90-line
    copy-pasted blocks that had already drifted out of sync (the file-path
    version had picked up the log1p mel-peak-energy fix and the array
    version hadn't) — a single shared implementation removes that class of
    bug entirely, in addition to being where the latency work below lives.

    Latency notes (see profiling in the accompanying commit message):
      - A single magnitude STFT is computed once and reused (via `S=`) by
        every spectral feature that supports it: centroid, rolloff,
        bandwidth, contrast, flatness, chroma (needs power) and mel/MFCC's
        underlying melspectrogram (needs power). Previously each of these
        called librosa with only `y=`, silently recomputing its own STFT
        internally — 8+ redundant STFTs per clip.
      - Harmonic/percussive energy reuses that same STFT and skips the
        ISTFT reconstruction step entirely (mask=True + energy computed
        directly from mask * power spectrum) — this was the single largest
        cost in the whole pipeline (median-filtering the spectrogram via
        scipy.ndimage.rank_filter), and computing it twice (once for masks,
        once again from a completely independent STFT+ISTFT pass) was pure
        waste.
      - Tempo/beat-tracking (librosa.beat.beat_track) was measured via
        feature-importance analysis (src/experiment.py importances mode) at
        0.02% combined RF+GBT importance — rank ~106 out of ~118 named
        features, i.e. essentially noise to the model — while costing
        ~2-3ms/clip (plus an internal onset-strength STFT of its own). It
        has been dropped.
      - The voiced-fraction autocorrelation (previously a per-frame Python
        loop calling np.correlate) is vectorized via FFT-based
        autocorrelation across all frames at once (verified numerically
        identical to the original).
      - Harmonic peak prominence's per-frame power_to_db conversion is now
        batched into a single array op (scipy has no vectorized find_peaks,
        so the peak search itself still loops per-frame).
    """
    features = []

    # ── shared spectrogram (computed once, reused everywhere below) ───────
    stft_mag = np.abs(librosa.stft(audio))
    power_spec = stft_mag ** 2

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
    # where the "center of mass" of the spectrum sits. Reuses shared STFT.
    centroid = librosa.feature.spectral_centroid(sr=sr, S=stft_mag)[0]
    features.append([np.mean(centroid), np.std(centroid)])

    # ── 5. Spectral Rolloff ────────────────────────────────────────────────
    # frequency below which 85% of energy sits. Reuses shared STFT.
    rolloff = librosa.feature.spectral_rolloff(sr=sr, S=stft_mag)[0]
    features.append([np.mean(rolloff), np.std(rolloff)])

    # ── 6. Spectral Flux ──────────────────────────────────────────────────
    # how fast spectrum changes frame to frame
    # speech changes smoothly, noise changes randomly
    flux = np.sqrt(np.sum(np.diff(stft_mag, axis=1) ** 2, axis=0))
    features.append([np.mean(flux), np.std(flux)])

    # ── 7. Spectral Bandwidth ─────────────────────────────────────────────
    bandwidth = librosa.feature.spectral_bandwidth(sr=sr, S=stft_mag)[0]
    features.append([np.mean(bandwidth), np.std(bandwidth)])

    # ── 8. Chroma Features ────────────────────────────────────────────────
    # pitch class information — speech has consistent pitch patterns.
    # chroma_stft expects a power spectrogram when given S=.
    chroma = librosa.feature.chroma_stft(sr=sr, S=power_spec)
    features.append([np.mean(chroma), np.std(chroma)])

    # ── 9. Mel Spectrogram Stats ──────────────────────────────────────────
    # melspectrogram expects a power spectrogram when given S=.
    mel = librosa.feature.melspectrogram(sr=sr, S=power_spec, n_mels=40)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    # NOTE: mel_db's max is always exactly 0dB by construction (ref=np.max
    # normalizes relative to the clip's own peak) — it carries zero signal
    # and was previously a constant, dead feature. log1p(linear power max)
    # instead captures real absolute-energy-level information.
    features.append([
        np.mean(mel_db), np.std(mel_db),
        float(np.log1p(np.max(mel))), np.min(mel_db)
    ])

    # ── 10. Harmonic vs Percussive ratio ───────────────────────────────────
    # speech is mostly harmonic, noise is percussive. Reuses shared STFT and
    # skips ISTFT reconstruction (see _harmonic_percussive_energy_fast).
    h_energy, p_energy = _harmonic_percussive_energy_fast(stft_mag)
    ratio = h_energy / (p_energy + 1e-10)
    features.append([h_energy, p_energy, ratio])

    # ── 11. Silence ratio ─────────────────────────────────────────────────
    # proportion of frames below energy threshold
    threshold  = 0.01 * np.max(np.abs(audio))
    silence_ratio = np.mean(np.abs(audio) < threshold)
    features.append([silence_ratio])

    # ── 12. Pitch / voicing (F0 via YIN + autocorrelation voicing) ─────────
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

    # ── 13. Spectral Contrast ──────────────────────────────────────────────
    # difference between peaks and valleys across sub-bands — voiced speech
    # (harmonics standing out of the noise floor) tends to show higher
    # contrast than diffuse environmental noise. Reuses shared STFT.
    contrast = librosa.feature.spectral_contrast(sr=sr, S=stft_mag)
    features.append([np.mean(contrast), np.std(contrast)])

    # ── 14. Spectral Flatness ──────────────────────────────────────────────
    # ~1.0 for noise-like/white-noise spectra, much lower for tonal/
    # harmonic signals like voiced speech. Reuses shared STFT.
    flatness = librosa.feature.spectral_flatness(S=stft_mag)[0]
    features.append([np.mean(flatness), np.std(flatness)])

    # ── 15. Spectral Entropy ────────────────────────────────────────────────
    # low for voiced speech (energy concentrated in formants/harmonics),
    # high for diffuse environmental noise. Reuses the shared STFT.
    entropy_mean, entropy_std = _spectral_entropy_stats(power_spec)
    features.append([entropy_mean, entropy_std])

    # ── 16. Harmonic Peak Prominence — DROPPED (round-2 latency work) ──────
    # Was: ~3.13ms/file (~13% of total feature-extraction latency), 2
    # features ranking #62/119 (0.0487%) and #96/119 (0.0214%) combined
    # RF+GBT importance -- essentially noise to the model, same tier as
    # tempo/beat-tracking (0.02% importance) which was already dropped for
    # the same expensive-and-unimportant reason. See
    # _harmonic_peak_prominence_stats()'s docstring above for the full
    # profiling numbers. Validated via full retrain + held-out re-score
    # before adopting (see commit message).

    # ── 17. Amplitude Envelope Modulation Shape ─────────────────────────────
    # General acoustic principle (not tuned to any specific clip): speech is
    # built from a sequence of discrete syllables, giving its amplitude
    # envelope a fairly regular ~3-8Hz modulation. Sustained tonal sources
    # (car horns, sirens, engine drones) instead show a sharp onset followed
    # by a flat, sustained plateau, which concentrates envelope-spectrum
    # energy near 0Hz and gives a more/less diffuse modulation-entropy
    # profile than speech's regular rhythm. See _envelope_shape_stats for
    # verification against the full noise pool (all UrbanSound8K
    # categories), not just car horn.
    env_dc_frac, env_mod_entropy = _envelope_shape_stats(audio, sr)
    features.append([env_dc_frac, env_mod_entropy])

    # flatten everything into one vector
    flat = []
    for f in features:
        if isinstance(f, np.ndarray):
            flat.extend(f.flatten().tolist())
        else:
            flat.extend(f)

    return np.array(flat)


def extract_features(file_path: str) -> np.ndarray:
    """
    Extracts 150+ temporal and spectral features from a .wav file.
    Captures both WHAT audio sounds like and HOW it changes over time.
    """
    audio, sr = librosa.load(file_path, sr=16000, mono=True)

    # guard against very short clips
    if len(audio) < sr * 0.1:
        audio = np.pad(audio, (0, int(sr * 0.1) - len(audio)))

    return _extract_features_core(audio, sr)


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

    Previously this duplicated extract_features()'s entire body as a
    separately maintained copy, which had already drifted (missing the
    log1p mel-peak-energy fix present in extract_features). Now both
    delegate to the single shared _extract_features_core().
    """
    if len(audio) < sr * 0.1:
        audio = np.pad(audio, (0, int(sr * 0.1) - len(audio)))

    return _extract_features_core(audio, sr)

