import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from timeit import default_timer as timer
from tqdm.auto import tqdm

print(torch.__version__)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"Device: {device}")

train_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

class_names = train_data.classes
print(f"Classes: {class_names}")

BATH_SIZE = 32

train_dataloader = DataLoader(train_data, batch_size=BATH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=BATH_SIZE, shuffle=True)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    

model = NeuralNetwork()
print(f"Model: {model}")

def accuracy_fn(pred, target):
    pred = torch.argmax(pred, dim=1)
    correct = pred.eq(target)
    accuracy = correct.sum() / torch.FloatTensor([target.shape[0]])
    return accuracy

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

def print_train_timer(start, end):
    print(f"Train Time: {end - start:.2f} seconds")

torch.manual_seed(42)
model = NeuralNetwork().to(device)
X, y = X.to(device), y.to(device)

train_time = timer()

epochs = 3
for epoch in tqdm(range(epochs)):
    train_loss = 0
    for batch, (X, y) in enumerate(tqdm(train_dataloader)):
        model.train()
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            print(f"Epoch: {epoch}, Loss: {loss.item()}")
