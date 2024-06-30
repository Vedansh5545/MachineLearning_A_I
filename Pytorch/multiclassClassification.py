import torch
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

"""

This code performs multiclass classification using PyTorch. 
It generates a synthetic dataset using the make_blobs function from scikit-learn, splits the data into train and test sets, 
defines a neural network model using PyTorch's nn.Module, trains the model using stochastic gradient descent (SGD), 
and evaluates the model's performance on the test set. It also plots the generated data, the decision boundary of the trained model, 
and the training and testing loss and accuracy over epochs.

"""


import torch.nn as nn
import matplotlib.pyplot as plt

# Hyperparameters
Num_Classes = 4
Num_features = 2
Random_State = 42
epochs = 10000
learning_rate = 0.01

# Generate data
X_blob, y_blob = make_blobs(n_samples=1000, centers=Num_Classes, n_features=Num_features, random_state=Random_State, cluster_std=1.5)

# Convert data to tensors
X_blob = torch.tensor(X_blob, dtype=torch.float32)
y_blob = torch.tensor(y_blob, dtype=torch.long)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_blob, y_blob, test_size=0.2, random_state=Random_State)

# Plot the data
plt.figure(figsize=(6, 6))
plt.scatter(X_blob[:, 0], X_blob[:, 1], c=y_blob, cmap=plt.cm.RdYlBu, edgecolors='k')
plt.title("Generated Data")
plt.show()

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Define the model
class BlobModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=Num_features, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=15)
        self.layer_3 = nn.Linear(in_features=15, out_features=10)
        self.layer_4 = nn.Linear(in_features=10, out_features=8)
        self.layer_5 = nn.Linear(in_features=8, out_features=6)
        self.layer_6 = nn.Linear(in_features=6, out_features=Num_Classes)
        self.relu = nn.ReLU()

    def forward(self, X):
        x = self.relu(self.layer_1(X))
        x = self.relu(self.layer_2(x))
        x = self.relu(self.layer_3(x))
        x = self.relu(self.layer_4(x))
        x = self.relu(self.layer_5(x))
        return self.layer_6(x)

model = BlobModel()
model.to(device)
print(model)

# Loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Accuracy function
def accuracy(y_pred, y_true):
    correct = (y_pred == y_true).float()
    acc = correct.sum() / len(correct)
    return acc

# Initialize lists to store loss and accuracy for plotting
train_losses = []
test_losses = []
test_accuracies = []

# Training loop
torch.manual_seed(42)

X_blob_train = X_train.to(device)
y_blob_train = y_train.to(device)
X_blob_test = X_test.to(device)
y_blob_test = y_test.to(device)

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_blob_train)
    loss = loss_fn(y_pred, y_blob_train)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        y_test_logits = model(X_blob_test)
        test_loss = loss_fn(y_test_logits, y_blob_test)
        y_test_pred = torch.argmax(y_test_logits, dim=1)
        test_acc = accuracy(y_test_pred, y_blob_test)
    
    # Store loss and accuracy for plotting
    train_losses.append(loss.item())
    test_losses.append(test_loss.item())
    test_accuracies.append(test_acc.item())
    
    if epoch % 1000 == 0:
        print(f"Epoch: {epoch} Train Loss: {loss.item()} Test Loss: {test_loss.item()} Test Accuracy: {test_acc.item()}")

# Function to plot decision boundary
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = torch.meshgrid(torch.linspace(x_min, x_max, 100),
                            torch.linspace(y_min, y_max, 100), indexing='xy')
    grid = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1)], dim=1)
    grid = grid.to(device)
    model.to(device)
    Z = model(grid).detach().cpu()
    Z = torch.argmax(Z, dim=1).reshape(xx.shape)  # Get the predicted class
    plt.contourf(xx.numpy(), yy.numpy(), Z.numpy(), cmap=plt.cm.RdYlBu, alpha=0.8)
    plt.scatter(X[:, 0].numpy(), X[:, 1].numpy(), c=y.numpy(), cmap=plt.cm.RdYlBu, edgecolors='k')
    plt.xlim(xx.min().item(), xx.max().item())
    plt.ylim(yy.min().item(), yy.max().item())
    plt.xticks(())
    plt.yticks(())

# Plot decision boundary for train and test datasets
model.to('cpu')
X_blob_train = X_blob_train.to('cpu')
y_blob_train = y_blob_train.to('cpu')
X_blob_test = X_blob_test.to('cpu')
y_blob_test = y_blob_test.to('cpu')

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plot_decision_boundary(model, X_blob_train, y_blob_train)
plt.title("Train")

plt.subplot(1, 2, 2)
plot_decision_boundary(model, X_blob_test, y_blob_test)
plt.title("Test")

plt.show()

# Plot training and testing loss
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.title("Loss over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

# Plot testing accuracy
plt.subplot(1, 2, 2)
plt.plot(test_accuracies, label='Test Accuracy')
plt.title("Accuracy over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.show()
