import os
import numpy as np
import joblib
import shap
import librosa
from src.classifier import extract_features

# ── Feature Names ──────────────────────────────────────────────────────────
def get_feature_names() -> list:
    """
    Returns human readable names for all 150+ features.
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
    names += ["Mel mean", "Mel std", "Mel peak energy (log)", "Mel min"]

    # NOTE: Tempo/beat-tracking was dropped from extract_features() — measured
    # at 0.02% combined RF+GBT feature importance (rank ~106/118) while
    # costing several ms/clip in latency (see src/classifier.py's
    # _extract_features_core docstring and the commit that removed it).

    # Harmonic/Percussive (3)
    names += ["Harmonic energy", "Percussive energy", "Harmonic ratio"]

    # Silence ratio (1)
    names += ["Silence ratio"]

    # Pitch / voicing via YIN (4)
    names += ["F0 mean", "F0 std", "F0 range", "Voiced fraction"]

    # Spectral Contrast (2)
    names += ["Spectral contrast mean", "Spectral contrast std"]

    # Spectral Flatness (2)
    names += ["Spectral flatness mean", "Spectral flatness std"]

    # Spectral Entropy (2)
    names += ["Spectral entropy mean", "Spectral entropy std"]

    # Harmonic Peak Prominence (2)
    names += ["Harmonic peak prominence mean", "Harmonic peak prominence std"]

    # Amplitude Envelope Modulation Shape (2) — distinguishes speech's
    # syllable-rate amplitude modulation from sustained tonal sources
    # (car horns, sirens, engine drones) whose envelope is mostly a flat
    # sustained plateau after a sharp onset.
    names += ["Envelope DC fraction", "Envelope modulation entropy"]

    return names


# ── Human Readable Interpretation ─────────────────────────────────────────
def interpret_feature(name: str, value: float, importance: float) -> str:
    """
    Converts a feature name and value into a human readable explanation.
    """
    if "Spectral entropy" in name and "std" in name:
        if value > 1.0:
            return "HIGH entropy variation — mixed tonal/noisy segments over time"
        else:
            return "LOW entropy variation — consistent spectral concentration"

    elif "Spectral entropy" in name:
        if value < 3.5:
            return "LOW entropy — energy concentrated in formants/harmonics, speech-like"
        elif value < 5.0:
            return "MODERATE entropy — mixed tonal/diffuse spectrum"
        else:
            return "HIGH entropy — energy spread across spectrum, noise-like"

    elif "Harmonic peak prominence" in name and "std" in name:
        if value > 3:
            return "HIGH variation in peak clarity over time — dynamic, speech-like"
        else:
            return "LOW variation — consistently clear or consistently flat spectrum"

    elif "Harmonic peak prominence" in name:
        if value > 8:
            return "HIGH prominence — clear harmonic peaks above noise floor, speech-like"
        else:
            return "LOW prominence — no clear peaks standing out, noise-like"

    elif "Voiced fraction" in name:
        pct = value * 100
        if pct > 40:
            return f"{pct:.0f}% voiced frames — clear pitch track, strongly speech-like"
        elif pct > 10:
            return f"{pct:.0f}% voiced frames — some pitched content present"
        else:
            return f"{pct:.0f}% voiced frames — little to no pitch, noise-like"

    elif "F0 mean" in name:
        if value == 0:
            return "No reliable pitch detected — noise-like"
        elif 80 <= value <= 300:
            return f"{value:.0f}Hz — within typical human speech pitch range"
        else:
            return f"{value:.0f}Hz — outside typical speech pitch range"

    elif "F0 std" in name or "F0 range" in name:
        if value == 0:
            return "No pitch variation — no voiced content detected"
        elif value < 40:
            return "LOW pitch variation — steady voiced tone, speech-like"
        else:
            return "HIGH pitch variation — sweeping/unstable pitch"

    elif "Spectral contrast" in name and "std" in name:
        if value > 5:
            return "HIGH variation in peak/valley contrast — dynamic, speech-like"
        else:
            return "LOW variation — flat spectral contrast over time"

    elif "Spectral contrast" in name:
        if value > 20:
            return "HIGH contrast — clear harmonic peaks over noise floor, speech-like"
        else:
            return "LOW contrast — diffuse spectrum, noise-like"

    elif "Spectral flatness" in name and "std" in name:
        if value > 0.05:
            return "HIGH variation in tonality over time — mixed content"
        else:
            return "LOW variation — consistently tonal or consistently noisy"

    elif "Spectral flatness" in name:
        if value < 0.1:
            return "LOW flatness — tonal/harmonic spectrum, speech-like"
        elif value < 0.3:
            return "MODERATE flatness — mixed tonal/noisy content"
        else:
            return "HIGH flatness — noise-like, closer to white noise"

    elif "Harmonic ratio" in name:
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

    elif "Envelope DC fraction" in name:
        if value > 0.25:
            return "HIGH near-DC envelope energy — regular syllable-like amplitude modulation, speech-like"
        else:
            return "LOW near-DC envelope energy — flat sustained plateau, consistent with a held tone or steady noise"

    elif "Envelope modulation entropy" in name:
        if value < 3.6:
            return "LOW modulation entropy — energy concentrated in a narrow rhythm band, speech-like"
        else:
            return "HIGH modulation entropy — diffuse envelope shape, noise-like"

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

    elif "Mel peak energy" in name:
        if value < 2:
            return "LOW peak energy — quiet clip overall"
        elif value < 5:
            return "MODERATE peak energy — typical speech loudness"
        else:
            return "HIGH peak energy — loud clip overall"

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
    import json

    args = [a for a in sys.argv[1:] if a != "--json"]
    as_json = "--json" in sys.argv[1:]

    if len(args) < 1:
        print("Usage: python3 -m src.explainer <audio_file.wav> [--json]")
        print("Example: python3 -m src.explainer data/speech/speech_001.wav")
        print("         python3 -m src.explainer data/speech/speech_001.wav --json > explanation.json")
        sys.exit(1)

    audio_path = args[0]
    if not os.path.exists(audio_path):
        print(f"Error: File not found: {audio_path}")
        sys.exit(1)

    result = explain(audio_path)

    if as_json:
        print(json.dumps(result, indent=2))
    else:
        print_explanation(result)