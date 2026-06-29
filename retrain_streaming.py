import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from src.classifier import extract_features

def retrain():
    print("=" * 50)
    print("  NOVA-VAD STREAMING RETRAINER")
    print("=" * 50)

    MIC_SPEECH = "data/mic_speech"
    MIC_NOISE  = "data/mic_noise"

    # check data exists
    if not os.path.exists(MIC_SPEECH):
        print("No mic data found. Run record_dataset.py first.")
        return

    X, y = [], []

    print("\nExtracting features from mic speech...")
    for f in sorted(os.listdir(MIC_SPEECH)):
        if f.endswith(".wav"):
            path = os.path.join(MIC_SPEECH, f)
            X.append(extract_features(path))
            y.append(1)
            print(f"  ✓ {f}")

    print("\nExtracting features from mic noise...")
    for f in sorted(os.listdir(MIC_NOISE)):
        if f.endswith(".wav"):
            path = os.path.join(MIC_NOISE, f)
            X.append(extract_features(path))
            y.append(0)
            print(f"  ✓ {f}")

    X = np.array(X)
    y = np.array(y)

    print(f"\nTraining on {len(y)} mic-captured files...")

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rf  = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    gbt = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

    rf.fit(X_scaled, y)
    gbt.fit(X_scaled, y)

    # save as streaming-specific models
    os.makedirs("models", exist_ok=True)
    joblib.dump(rf,     "models/stream_rf.pkl")
    joblib.dump(gbt,    "models/stream_gbt.pkl")
    joblib.dump(scaler, "models/stream_scaler.pkl")

    print("\n✅ Streaming models saved to models/")
    print("  Now run: python3 -m src.stream")

if __name__ == "__main__":
    retrain()