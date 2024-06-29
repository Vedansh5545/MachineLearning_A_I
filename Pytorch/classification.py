import torch # type: ignore

"""
pytorch neural network for classification building a simple neural network model with Liner Regression
"""

import torch.nn as nn # type: ignore
import matplotlib.pyplot as plt # type: ignore

# Print the PyTorch version
print(torch.__version__)

# Check if CUDA is available and set the device accordingly
device = "cuda" if torch.cuda.is_available() else "cpu"

# Print the selected device
print(f"Device: {device}")

# Initialize weight and bias variables
weight = 0.7
bais = 0.3

# Create Data

start = 0
end = 1
step = 0.02

X = torch.arange(start, end, step).unsqueeze(dim = 1)

y = weight * X + bais 

# Split the data into training and testing sets
train_split = int(0.8 * len(X))

X_train, X_test = X[:train_split], X[train_split:]
y_train, y_test = y[:train_split], y[train_split:]

# Define the neural network model

class LinearRegressionModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(in_features = 1, out_features =  1)

    def forward(self, x):
        return self.linear(x)
    
model_1 = LinearRegressionModelV0().to(device)
print(model_1)

# Define the loss function and optimizer
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params = model_1.parameters(), lr = 0.01)

# Train and Test the model loop
num_epochs = 1000
for epoch in range(num_epochs):
    model_1.train()
    y_pred = model_1(X_train.to(device))
    loss = loss_fn(y_pred, y_train.to(device))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    model_1.eval()
    with torch.no_grad():
        y_test_pred = model_1(X_test.to(device))
        test_loss = loss_fn(y_test_pred, y_test.to(device))

    if (epoch + 1) % 100 == 0:
        print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")

# Make predictions
model_1.eval()
with torch.no_grad():
    y_pred = model_1(X.to(device))

# Plot the results
def plotPrediction(X, y, y_pred):
    plt.figure(figsize=(12, 6))
    plt.plot(X, y, label='True')
    plt.plot(X, y_pred, label='Prediction')
    plt.legend()
    plt.show()

plotPrediction(X, y, y_pred)
