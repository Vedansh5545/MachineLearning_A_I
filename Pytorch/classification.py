"""
This script demonstrates how to train a PyTorch model for binary classification using a circular dataset.
The model is trained to classify points in a 2D space as either belonging to the inner circle or the outer circle.
The script generates a circular dataset, splits it into training and testing sets, defines a neural network model,
trains the model using the training set, evaluates the model on the testing set, and plots the decision boundaries,
loss, accuracy, and confusion matrix.
"""

import torch
import numpy as np
import pandas as pd
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import torch.nn as nn
import matplotlib.pyplot as plt

# Generate circular data
n_samples = 1000
X, y = make_circles(n_samples=n_samples, noise=0.03, random_state=42)

# Convert data to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scatter plot of the generated data
plt.figure(figsize=(6, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='k')
plt.title("Generated Data")
plt.show()

# Define your model
class CircleModelV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=15)
        self.layer_3 = nn.Linear(in_features=15, out_features=10)
        self.layer_4 = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, X):
        x = self.relu(self.layer_1(X))
        x = self.relu(self.layer_2(x))
        x = self.relu(self.layer_3(x))
        return self.layer_4(x)

# Instantiate the model
model = CircleModelV1()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define loss function and optimizer
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Function to calculate accuracy
def accuracy(y_pred, y_true):
    y_pred = y_pred.round()
    correct = (y_pred == y_true).sum().item()
    total = y_true.size(0)
    accuracy = correct / total
    return accuracy

# Training loop
torch.manual_seed(42)
epochs = 1000

X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)

train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    y_logits = model(X_train).squeeze()
    loss = loss_fn(y_logits, y_train)
    train_losses.append(loss.item())
    acc = accuracy(torch.sigmoid(y_logits), y_train)
    train_accuracies.append(acc)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        y_logits_test = model(X_test).squeeze()
        test_loss = loss_fn(y_logits_test, y_test)
        test_losses.append(test_loss.item())
        test_acc = accuracy(torch.sigmoid(y_logits_test), y_test)
        test_accuracies.append(test_acc)

    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Train Loss: {loss.item():.4f} | Train Acc: {acc:.4f} | Test Loss: {test_loss.item():.4f} | Test Acc: {test_acc:.4f}")

# Evaluate model on test set
model.eval()
with torch.no_grad():
    y_logits = model(X_test).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))
    print(f"Final Test Accuracy: {accuracy(y_pred, y_test):.4f}")

# Plotting decision boundaries
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = torch.meshgrid(torch.linspace(x_min, x_max, 100),
                            torch.linspace(y_min, y_max, 100), indexing='xy')
    grid = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1)], dim=1)
    Z = model(grid.to(device)).detach().cpu().numpy().reshape(xx.shape)
    Z = Z > 0  # Apply threshold to make it a binary classification boundary
    plt.contourf(xx.numpy(), yy.numpy(), Z, cmap=plt.cm.RdYlBu, alpha=0.8)
    plt.scatter(X[:, 0].numpy(), X[:, 1].numpy(), c=y.numpy(), cmap=plt.cm.RdYlBu, edgecolors='k')
    plt.xlim(xx.min().item(), xx.max().item())
    plt.ylim(yy.min().item(), yy.max().item())
    plt.xticks(())
    plt.yticks(())

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model, X_train.cpu(), y_train.cpu())
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model, X_test.cpu(), y_test.cpu())
plt.tight_layout()
plt.show()

# Plotting loss and accuracy over epochs
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss over Epochs')

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy over Epochs')

plt.tight_layout()
plt.show()

# Plotting confusion matrix for model efficiency
cm = confusion_matrix(y_test.cpu(), y_pred.cpu())
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.RdYlBu)
plt.title("Confusion Matrix")
plt.show()
