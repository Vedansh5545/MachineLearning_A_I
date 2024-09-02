# import library
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Device Agnostic Code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Load the data
data = yf.download('AAPL', start='2020-01-01', end='2024-08-08')
print(data.head())

# Data Preprocessing
data = data['Close'].values  # Use 'Close' price for prediction
data = data.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(-1, 1))
data_normalized = scaler.fit_transform(data)

# Convert data to sequences
def create_inout_sequences(input_data, seq_length):
    inout_seq = []
    L = len(input_data)
    for i in range(L-seq_length):
        train_seq = input_data[i:i+seq_length]
        train_label = input_data[i+seq_length:i+seq_length+1]
        inout_seq.append((train_seq, train_label))
    return inout_seq

seq_length = 100
train_inout_seq = create_inout_sequences(data_normalized, seq_length)

# Convert to PyTorch tensors
train_inout_seq = [(torch.tensor(seq, dtype=torch.float32).to(device),
                    torch.tensor(label, dtype=torch.float32).to(device)) 
                   for seq, label in train_inout_seq]

# Define the LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size).to(device),
                            torch.zeros(1, 1, self.hidden_layer_size).to(device))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

# Initialize model, loss function, and optimizer
model = LSTM().to(device)
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
epochs = 50
for i in range(epochs):
    for seq, labels in train_inout_seq:
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size).to(device),
                             torch.zeros(1, 1, model.hidden_layer_size).to(device))

        y_pred = model(seq)

        # Squeeze the labels to match the shape of y_pred
        single_loss = loss_function(y_pred, labels.squeeze())
        single_loss.backward()
        optimizer.step()

    if i % 10 == 0:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

# Testing the model on unseen data
model.eval()
test_inputs = data_normalized[-seq_length:].tolist()
model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size).to(device),
                     torch.zeros(1, 1, model.hidden_layer_size).to(device))

for i in range(10):  # Predicting 10 days into the future
    seq = torch.tensor(test_inputs[-seq_length:], dtype=torch.float32).to(device).view(seq_length, 1)
    with torch.no_grad():
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size).to(device),
                             torch.zeros(1, 1, model.hidden_layer_size).to(device))
        predicted_value = model(seq)
        test_inputs.append(predicted_value.item())

# Inverse transform the predictions
predicted_stock_price = scaler.inverse_transform(np.array(test_inputs[seq_length:]).reshape(-1, 1))

print(predicted_stock_price)
