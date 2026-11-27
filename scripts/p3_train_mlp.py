
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
    input_size = int(np.prod(X_train.shape[1:]))
    print(f"Input: {X_train.shape[1:]} -> {input_size}, classes: {num_classes}")
    model = ActionMLP(input_size=input_size, num_classes=num_classes).to(device)

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
                MODEL_DIR / "best_mlp.pth"
            )

    print("\nTraining completed.")
    print(f"Best validation accuracy: {best_val_acc:.3f}")


if __name__ == "__main__":
    main()
