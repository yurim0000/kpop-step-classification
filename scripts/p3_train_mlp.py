
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "dataset"
MODEL_DIR = ROOT / "checkpoints"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 0.001

# TODO (Optional):
# Design a domain-specific model for your motion classification task.
class ActionMLP(nn.Module):
    def __init__(self, input_size=30 * 99, num_classes=2):
        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.network(x)


class ActionCNN1D(nn.Module):
    """
    1D CNN over time: each frame's values are channels, convolution slides along the 30 frames,
    so the same short movement pattern is recognized wherever it occurs in the window.
    """
    def __init__(self, frame_dim, num_classes=2):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv1d(frame_dim, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):                 # x: (batch, frames, frame_dim)
        return self.network(x.transpose(1, 2))


MODEL_TYPE = "cnn"   # "mlp" (baseline) | "cnn"  -> model comparison experiment


def build_model(model_type, seq_shape, num_classes):
    """seq_shape = (frames, frame_dim). Used here and in p4_predict.py."""
    if model_type == "mlp":
        return ActionMLP(input_size=int(np.prod(seq_shape)), num_classes=num_classes)
    if model_type == "cnn":
        return ActionCNN1D(frame_dim=seq_shape[1], num_classes=num_classes)
    raise ValueError(f"Unknown model type: {model_type}")


def main():
    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    # Load dataset
    X_train = np.load(DATA_DIR / "X_train.npy")
    y_train = np.load(DATA_DIR / "y_train.npy")
    X_val = np.load(DATA_DIR / "X_val.npy")
    y_val = np.load(DATA_DIR / "y_val.npy")

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE
    )

    # Create MLP model (input size follows the dataset: 30 x 45 for upper body, 30 x 99 for full body)
    with open(DATA_DIR / "labels.json", "r", encoding="utf-8") as f:
        num_classes = len(json.load(f))
    print(f"Model: {MODEL_TYPE} | input: {X_train.shape[1:]} | classes: {num_classes}")
    model = build_model(MODEL_TYPE, X_train.shape[1:], num_classes).to(device)
    model_path = MODEL_DIR / f"best_{MODEL_TYPE}.pth"
    # tells p4_predict.py which model to rebuild
    with open(MODEL_DIR / "model_meta.json", "w", encoding="utf-8") as f:
        json.dump({"model_type": MODEL_TYPE, "checkpoint": model_path.name,
                   "seq_shape": list(X_train.shape[1:])}, f, indent=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_val_acc = -1.0

    for epoch in range(EPOCHS):

        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()

            logits = model(X)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(y)
            train_correct += (
                logits.argmax(dim=1) == y
            ).sum().item()

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0

        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)

                logits = model(X)
                loss = criterion(logits, y)

                val_loss += loss.item() * len(y)
                val_correct += (
                    logits.argmax(dim=1) == y
                ).sum().item()

        train_loss /= len(train_dataset)
        val_loss /= len(val_dataset)

        train_acc = train_correct / len(train_dataset)
        val_acc = val_correct / len(val_dataset)

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.3f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            torch.save(
                model.state_dict(),
                model_path
            )

    print("\nTraining completed.")
    print(f"Best validation accuracy: {best_val_acc:.3f}")


if __name__ == "__main__":
    main()
