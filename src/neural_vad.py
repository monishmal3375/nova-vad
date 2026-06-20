import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
import joblib
from src.classifier import extract_features, build_dataset


# ── Model Architecture ─────────────────────────────────────────────────────
class VADNet(nn.Module):
    """
    Small neural network for Voice Activity Detection.
    Input:  78 MFCC features
    Output: 1 probability (speech vs no speech)
    """
    def __init__(self, input_size=78):
        super(VADNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


# ── Training Function ──────────────────────────────────────────────────────
def train_model(X_train, y_train, input_size=78, epochs=300):
    """
    Trains VADNet on given features and labels.
    Returns trained model.
    """
    X_tensor = torch.FloatTensor(X_train)
    y_tensor = torch.FloatTensor(y_train).unsqueeze(1)

    dataset    = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    model     = VADNet(input_size=input_size)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss   = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()

        if (epoch + 1) % 100 == 0:
            print(f"    Epoch [{epoch+1}/{epochs}] Loss: {epoch_loss:.4f}")

    return model


# ── Evaluation ─────────────────────────────────────────────────────────────
def evaluate_neural_vad(speech_dir: str, noise_dir: str) -> dict:
    """
    Trains and evaluates VADNet using Leave-One-Out cross validation.
    Returns accuracy metrics.
    """
    print("\n[ NEURAL NETWORK VAD ]\n")
    X, y, filenames = build_dataset(speech_dir, noise_dir)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    loo         = LeaveOneOut()
    predictions = []

    print("\nRunning Leave-One-Out cross validation...")
    for i, (train_idx, test_idx) in enumerate(loo.split(X_scaled)):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = train_model(X_train, y_train)
        model.eval()

        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test)
            output        = model(X_test_tensor)
            pred          = 1 if output.item() > 0.5 else 0

        predictions.append(pred)

        filename   = filenames[test_idx[0]]
        true_label = "SPEECH" if y_test[0] == 1 else "NO SPEECH"
        pred_label = "SPEECH" if pred == 1 else "NO SPEECH"
        status     = "✓" if pred == y_test[0] else "✗"
        print(f"  {status} {filename} → predicted: {pred_label} | actual: {true_label}")

    predictions = np.array(predictions)

    # metrics
    total    = len(y)
    correct  = np.sum(predictions == y)
    accuracy = correct / total * 100

    tp = np.sum((predictions == 1) & (y == 1))
    tn = np.sum((predictions == 0) & (y == 0))
    fp = np.sum((predictions == 1) & (y == 0))
    fn = np.sum((predictions == 0) & (y == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # save final model trained on all data
    final_model = train_model(X_scaled, y)
    os.makedirs("models", exist_ok=True)
    torch.save(final_model.state_dict(), "models/vad_neural_net.pt")
    joblib.dump(scaler, "models/neural_scaler.pkl")
    print("\n✅ Neural network saved to models/")

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