"""
NOVA-VAD-frame-v2 feature set: v1's 58 causal features (unchanged, imported
directly — see scripts/frame_features.py) plus 4 new structural/harmonic
features chosen from acoustic first principles, not from inspecting test-
scene failures:

  - periodicity_strength: normalized autocorrelation peak in the human
    pitch range (80-400 Hz). Voiced speech is quasi-periodic; additive
    stationary noise is not, so this ratio degrades far more gracefully
    under noise than energy/magnitude-based features do — the autocorrelation
    of noise decays sharply away from lag 0, while a real pitch period keeps
    producing a secondary peak even when the signal is noisy, just a smaller
    one. This is the standard voiced/unvoiced detection principle used in
    classical pitch trackers (e.g. YIN), predating and independent of any
    result from this project's test scenes.
  - estimated_f0_hz: the pitch implied by that autocorrelation peak's lag,
    zeroed out when periodicity_strength is below a fixed confidence
    threshold (0.3) rather than reporting a meaningless "pitch" on noise.
  - spectral_flatness (mean, std): geometric-mean/arithmetic-mean of the
    power spectrum (librosa's standard Wiener-entropy definition). This is a
    ratio, not an absolute magnitude, so it is comparatively scale- and
    noise-invariant: flat (noise-like) spectra approach 1.0, peaky
    (tonal/harmonic, voiced-speech-like) spectra approach 0.0.

v1's 58 features are NOT modified, reweighted, or removed here — this keeps
the v1 model's already-registered result exactly reproducible and isolates
what changed. See reports/decision_v3.md for whether adding these features
(plus more training data) actually closed the noise gap.
"""
import numpy as np
import librosa

from scripts.frame_features import extract_frame_features as _v1_extract_frame_features

FEATURE_COUNT = 62  # 58 (v1) + 4 new (periodicity, f0, flatness mean/std)

_MIN_LAG_400HZ = None  # computed per-call since it depends on sr
_PERIODICITY_CONFIDENCE_THRESHOLD = 0.3


def _periodicity_features(audio_window: np.ndarray, sr: int):
    x = audio_window - np.mean(audio_window)
    energy = np.sum(x ** 2)
    if energy < 1e-10:
        return 0.0, 0.0

    min_lag = max(1, int(sr / 400))  # 400 Hz upper bound on speech F0
    max_lag = int(sr / 80)           # 80 Hz lower bound on speech F0
    if len(x) <= max_lag:
        return 0.0, 0.0

    # FFT-based autocorrelation (O(n log n), not O(n^2)) — needed since this
    # runs per-window over thousands of training/eval windows.
    size = 1
    while size < 2 * len(x):
        size *= 2
    spec = np.fft.rfft(x, size)
    autocorr = np.fft.irfft(spec * np.conj(spec))[:len(x)]

    zero_lag = autocorr[0]
    if zero_lag <= 0:
        return 0.0, 0.0
    normalized = autocorr / zero_lag

    search_range = normalized[min_lag:max_lag + 1]
    if len(search_range) == 0:
        return 0.0, 0.0

    peak_idx = int(np.argmax(search_range))
    peak_val = float(search_range[peak_idx])
    lag = min_lag + peak_idx

    periodicity_strength = max(0.0, peak_val)
    f0_hz = (sr / lag) if peak_val >= _PERIODICITY_CONFIDENCE_THRESHOLD else 0.0
    return periodicity_strength, float(f0_hz)


def extract_frame_features_v2(audio_window: np.ndarray, sr: int) -> np.ndarray:
    v1_features = _v1_extract_frame_features(audio_window, sr)

    min_len = int(0.05 * sr)
    window = audio_window
    if len(window) < min_len:
        window = np.pad(window, (0, min_len - len(window)))

    periodicity_strength, f0_hz = _periodicity_features(window, sr)

    flatness = librosa.feature.spectral_flatness(y=window, n_fft=512, hop_length=128)[0]
    flatness_mean = float(np.mean(flatness))
    flatness_std = float(np.std(flatness))

    new_features = np.array(
        [periodicity_strength, f0_hz, flatness_mean, flatness_std], dtype=np.float64
    )
    new_features = np.nan_to_num(new_features, nan=0.0, posinf=0.0, neginf=0.0)

    return np.concatenate([v1_features, new_features])
