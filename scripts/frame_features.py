"""
Causal, reduced feature set for frame-level VAD (plan Section 11.2 / 7.8 V1):
"a compact mobile baseline might use log energy, zero-crossing rate, spectral
centroid/bandwidth/rolloff, spectral flux, MFCC statistics, and a small mel
representation." Deliberately drops tempo/beat-tracking, chroma, and
harmonic/percussive separation from the original 106-feature set — those are
non-causal (need forward context or the whole clip), expensive, and were
part of what failed to generalize in the v0 frame-level test
(reports/decision_v1.md).

Each window is causal: features for the decision at time t are computed only
from audio in [t - window_s, t] — no look-ahead.
"""
import numpy as np
import librosa

FEATURE_COUNT = 58  # verified by direct measurement, not hand-counted (see tests/test_frame_features.py)


def extract_frame_features(audio_window: np.ndarray, sr: int) -> np.ndarray:
    # guard against windows too short for a stable STFT
    min_len = int(0.05 * sr)
    if len(audio_window) < min_len:
        audio_window = np.pad(audio_window, (0, min_len - len(audio_window)))

    features = []

    # log energy (RMS in dB)
    rms = librosa.feature.rms(y=audio_window)[0]
    log_rms = np.log10(rms + 1e-10)
    features.extend([np.mean(log_rms), np.std(log_rms)])

    # zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(audio_window)[0]
    features.extend([np.mean(zcr), np.std(zcr)])

    # spectral centroid
    centroid = librosa.feature.spectral_centroid(y=audio_window, sr=sr, n_fft=512, hop_length=128)[0]
    features.extend([np.mean(centroid), np.std(centroid)])

    # spectral bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(y=audio_window, sr=sr, n_fft=512, hop_length=128)[0]
    features.extend([np.mean(bandwidth), np.std(bandwidth)])

    # spectral rolloff
    rolloff = librosa.feature.spectral_rolloff(y=audio_window, sr=sr, n_fft=512, hop_length=128)[0]
    features.extend([np.mean(rolloff), np.std(rolloff)])

    # spectral flux
    stft = np.abs(librosa.stft(audio_window, n_fft=512, hop_length=128))
    if stft.shape[1] > 1:
        flux = np.sqrt(np.sum(np.diff(stft, axis=1) ** 2, axis=0))
        features.append(np.mean(flux))
    else:
        features.append(0.0)

    # MFCC statistics (13 coefficients, mean + std)
    mfcc = librosa.feature.mfcc(y=audio_window, sr=sr, n_mfcc=13, n_fft=512, hop_length=128)
    features.extend(np.mean(mfcc, axis=1).tolist())
    features.extend(np.std(mfcc, axis=1).tolist())

    # small mel representation (20 bands, mean only)
    mel = librosa.feature.melspectrogram(y=audio_window, sr=sr, n_mels=20, n_fft=512, hop_length=128)
    mel_db = librosa.power_to_db(mel, ref=1.0)
    features.extend(np.mean(mel_db, axis=1).tolist())

    # silence indicator
    threshold = 0.02
    silence_flag = 1.0 if np.max(np.abs(audio_window)) < threshold else 0.0
    features.append(silence_flag)

    vec = np.array(features, dtype=np.float64)
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    return vec
