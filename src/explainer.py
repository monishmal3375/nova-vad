import os
import numpy as np
import joblib
import shap
import librosa
from src.classifier import extract_features

# ── Feature Names ──────────────────────────────────────────────────────────
def get_feature_names() -> list:
    """
    Returns human readable names for all 106 features.
    Matches exactly what extract_features() produces.
    """
    names = []

    # MFCCs (13 x mean, std x 3 sets = 78)
    for kind in ["MFCC", "MFCC Delta", "MFCC Delta2"]:
        for i in range(13):
            names.append(f"{kind} {i+1} mean")
        for i in range(13):
            names.append(f"{kind} {i+1} std")

    # ZCR (4)
    names += ["ZCR mean", "ZCR std", "ZCR max", "ZCR min"]

    # RMS Energy (5)
    names += ["RMS mean", "RMS std", "RMS max", "RMS min", "RMS change rate"]

    # Spectral Centroid (2)
    names += ["Spectral centroid mean", "Spectral centroid std"]

    # Spectral Rolloff (2)
    names += ["Spectral rolloff mean", "Spectral rolloff std"]

    # Spectral Flux (2)
    names += ["Spectral flux mean", "Spectral flux std"]

    # Spectral Bandwidth (2)
    names += ["Spectral bandwidth mean", "Spectral bandwidth std"]

    # Chroma (2)
    names += ["Chroma mean", "Chroma std"]

    # Mel Spectrogram (4)
    names += ["Mel mean", "Mel std", "Mel max", "Mel min"]

    # Tempo (1)
    names += ["Tempo"]

    # Harmonic/Percussive (3)
    names += ["Harmonic energy", "Percussive energy", "Harmonic ratio"]

    # Silence ratio (1)
    names += ["Silence ratio"]

    return names


# ── Human Readable Interpretation ─────────────────────────────────────────
def interpret_feature(name: str, value: float, importance: float) -> str:
    """
    Converts a feature name and value into a human readable explanation.
    """
    if "Harmonic ratio" in name:
        if value > 2.0:
            return "VERY HIGH — strong tonal content, consistent with speech"
        elif value > 1.0:
            return "HIGH — tonal content present, speech-like"
        else:
            return "LOW — noise dominated, not speech-like"

    elif "ZCR std" in name:
        if value > 0.15:
            return "HIGH variability — irregular signal, noise-like"
        else:
            return "LOW variability — consistent pattern, speech-like"

    elif "ZCR mean" in name:
        if value < 0.05:
            return "LOW — smooth signal, consistent with voiced speech"
        elif value < 0.15:
            return "MODERATE — mixed voiced/unvoiced content"
        else:
            return "HIGH — chaotic signal, noise-like"

    elif "RMS change rate" in name:
        if abs(value) < 0.005:
            return "STEADY energy — consistent with background noise"
        elif abs(value) < 0.02:
            return "RHYTHMIC changes — consistent with speech syllables"
        else:
            return "IRREGULAR changes — unpredictable energy pattern"

    elif "RMS mean" in name:
        if value < 0.005:
            return "VERY LOW — near silence"
        elif value < 0.05:
            return "LOW — quiet audio"
        else:
            return "PRESENT — active audio signal"

    elif "Spectral flux mean" in name:
        if value < 50:
            return "SMOOTH transitions — consistent with speech"
        elif value < 150:
            return "MODERATE transitions — mixed content"
        else:
            return "RAPID transitions — chaotic, noise-like"

    elif "Spectral centroid" in name and "std" in name:
        if value < 800:
            return "STABLE frequency center — consistent with speech"
        elif value < 1500:
            return "MODERATE variation — mixed content"
        else:
            return "HIGH variation — shifting frequency center, noise-like"

    elif "Spectral centroid" in name:
        if value < 2000:
            return "LOW center — bass-heavy or muffled audio"
        elif value < 4000:
            return "MID range — typical speech frequency range"
        else:
            return "HIGH center — bright or noisy audio"

    elif "Tempo" in name:
        if value > 100:
            return f"{value:.0f} BPM — fast rhythm detected"
        elif value > 60:
            return f"{value:.0f} BPM — consistent with speech syllable rate"
        else:
            return f"{value:.0f} BPM — slow or no clear rhythm"

    elif "Silence ratio" in name:
        pct = value * 100
        if pct > 70:
            return f"{pct:.0f}% silence — mostly quiet, little speech activity"
        elif pct > 40:
            return f"{pct:.0f}% silence — mix of speech and pauses"
        else:
            return f"{pct:.0f}% silence — highly active audio"

    elif "MFCC Delta 1 std" in name:
        if value > 25:
            return "HIGH spectral change rate — dynamic audio like speech"
        else:
            return "LOW spectral change rate — static or repetitive noise"

    elif "MFCC Delta 2 std" in name:
        if value > 20:
            return "HIGH acceleration — rapidly changing audio, speech-like"
        else:
            return "LOW acceleration — slowly changing or static audio"

    elif "Mel mean" in name:
        if value < -60:
            return "LOW energy across mel bands — quiet or distant audio"
        elif value < -40:
            return "MODERATE energy — normal speech level"
        else:
            return "HIGH energy — loud or close-range audio"

    elif "MFCC" in name and "std" in name:
        if value > 100:
            return "HIGH variability — dynamic spectral content"
        else:
            return "LOW variability — stable spectral content"

    elif "MFCC" in name:
        return f"Spectral shape coefficient: {value:.3f}"
    elif "Spectral rolloff" in name and "std" in name:
        if value < 1000:
            return "LOW variation — stable high-frequency cutoff, speech-like"
        elif value < 2000:
            return "MODERATE variation — mixed frequency content"
        else:
            return "HIGH variation — shifting frequency cutoff, noise-like"

    else:
        return f"value: {value:.4f}"
# ── Main Explainer ─────────────────────────────────────────────────────────
def explain(audio_path: str, models_dir: str = "models") -> dict:
    """
    Takes an audio file and returns a full explanation of the VAD decision.
    
    Returns:
        dict with label, confidence, and feature importance breakdown
    """
    # load saved model and scaler
    rf_path     = os.path.join(models_dir, "nova_vad_rf.pkl")
    scaler_path = os.path.join(models_dir, "nova_vad_scaler.pkl")

    if not os.path.exists(rf_path):
        raise FileNotFoundError("Model not found. Run pipeline.py first to train the model.")

    rf     = joblib.load(rf_path)
    scaler = joblib.load(scaler_path)

    # extract features
    features     = extract_features(audio_path)
    features_scaled = scaler.transform([features])

    # get prediction and confidence
    prediction   = rf.predict(features_scaled)[0]
    probabilities = rf.predict_proba(features_scaled)[0]
    confidence   = probabilities[prediction] * 100

    label = "SPEECH" if prediction == 1 else "NO SPEECH"

    # get feature importances from Random Forest
    importances  = rf.feature_importances_
    feature_names = get_feature_names()

    # pad or trim feature names to match
    n_features = len(features)
    if len(feature_names) < n_features:
        feature_names += [f"Feature {i}" for i in range(len(feature_names), n_features)]
    feature_names = feature_names[:n_features]

    # rank features by importance
    ranked_idx = np.argsort(importances)[::-1]

    top_features = []
    for idx in ranked_idx[:10]:
        top_features.append({
            "feature":    feature_names[idx],
            "importance": round(float(importances[idx]) * 100, 2),
            "value":      round(float(features[idx]), 4),
            "meaning":    interpret_feature(feature_names[idx], features[idx], importances[idx])
        })

    result = {
        "file":       os.path.basename(audio_path),
        "label":      label,
        "confidence": round(confidence, 2),
        "top_features": top_features
    }

    return result


def print_explanation(result: dict):
    """
    Prints a human readable explanation of a VAD decision.
    """
    print("\n" + "=" * 55)
    print(f"  NOVA-VAD EXPLANATION")
    print("=" * 55)
    print(f"  File:        {result['file']}")
    print(f"  Prediction:  {result['label']}")
    print(f"  Confidence:  {result['confidence']}%")
    print("\n  Why this decision was made:")
    print("  " + "-" * 51)

    for i, f in enumerate(result['top_features']):
        print(f"  {i+1:2}. {f['feature'][:28]:<28} ({f['importance']:5.2f}%)")
        print(f"      → {f['meaning']}")

    print("=" * 55)


# ── CLI Usage ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python3 -m src.explainer <audio_file.wav>")
        print("Example: python3 -m src.explainer data/speech/speech_001.wav")
        sys.exit(1)

    audio_path = sys.argv[1]
    if not os.path.exists(audio_path):
        print(f"Error: File not found: {audio_path}")
        sys.exit(1)

    result = explain(audio_path)
    print_explanation(result)