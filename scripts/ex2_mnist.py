import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# 1. Transform
transform = transforms.ToTensor()
# 2. Download MNIST
train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)
test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

# 3. Check data
print("Train data X:", train_dataset.data.shape)
print("Train data Y:", train_dataset.targets.shape)
# original data
print(train_dataset.data[0].shape)
print(train_dataset.targets[0].shape)

from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

# 1. Normalize Image Data
X_train = train_dataset.data.float() / 255.0
X_test = test_dataset.data.float() / 255.0

# 2. One-hot Encoding
y_train = F.one_hot(train_dataset.targets, num_classes=10).float()
y_test = F.one_hot(test_dataset.targets, num_classes=10).float()

# 3. Create TensorDataset
train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)

# 4. Create DataLoader
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# 5. Check Batch
images, labels = next(iter(train_loader))

print(images.shape)  # (64, 28, 28)
print(labels.shape)  # (64, 10)

# # 4. Visualize
# plt.imshow(image.squeeze(), cmap="gray")
# plt.title(f"Label: {label}")
# plt.show()

# 6. Construct the MLP
# nn.Linear(28 * 28, 128) + nn.Tanh() : Perceptron layer
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 512),
    nn.Tanh(),
    nn.Linear(512, 10),
    nn.Softmax(dim=1)
)

EPOCHS = 10
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for epoch in range(EPOCHS):
    total_loss = 0.0

    for images, labels in train_loader:

        # Forward propagation
        outputs = model(images)

        # Cross-Entropy between probabilities and One-hot labels
        loss = -torch.sum(
            labels * torch.log(outputs.clamp_min(1e-8)),
            dim=1
        ).mean()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    train_loss = total_loss / len(train_loader)
    
    # -------------------------
    # Validation
    # -------------------------
    model.eval()

    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            # Forward propagation
            outputs = model(images)

            # Cross-Entropy Loss
            loss = -torch.sum(
                labels * torch.log(outputs.clamp_min(1e-8)),
                dim=1
            ).mean()

            val_loss += loss.item()

            # Predicted and Ground-truth Labels
            predicted = torch.argmax(outputs, dim=1)
            ground_truth = torch.argmax(labels, dim=1)

            correct += (predicted == ground_truth).sum().item()
            total += labels.size(0)

    val_loss /= len(test_loader)
    val_accuracy = 100 * correct / total

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Accuracy: {val_accuracy:.2f}%"
    )

    print(f"Epoch {epoch+1}: Loss = {total_loss / len(train_loader):.4f}")


# -------------------------
# Prediction Visualization
# -------------------------
model.eval()

test_images, labels = next(iter(test_loader))

with torch.no_grad():
    outputs = model(test_images)
    predicted = torch.argmax(outputs, dim=1)
    ground_truth = torch.argmax(labels, dim=1)

plt.figure(figsize=(15, 5))

for i in range(10):

    plt.subplot(2, 5, i + 1)

    plt.imshow(images[i].squeeze(), cmap="gray")

    plt.title(
        f"Predicted: {predicted[i].item()}\n"
        f"Actual: {ground_truth[i].item()}",
        color="green" if predicted[i] == ground_truth[i] else "red"
    )

    plt.axis("off")

plt.tight_layout()
plt.show()


# model_student = nn.Sequential(
   # TODO: Build your own MLP for MNIST classification
   # HINT: Use PyTorch documentation or an LLM
# )