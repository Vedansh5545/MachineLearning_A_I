
# PyTorch, NumPy, Pandas, and Matplotlib Cheat Sheet

## 1. Setting Up Your Environment
Make sure you have the required libraries installed:
```bash
pip install torch numpy pandas matplotlib
```

## 2. Creating and Handling Data

### Using NumPy
- Create arrays:
  ```python
  import numpy as np

  array = np.array([1, 2, 3])
  matrix = np.array([[1, 2], [3, 4]])
  ```
- Generate random data:
  ```python
  random_array = np.random.rand(3, 3)  # 3x3 matrix with random values
  ```

### Using Pandas
- Load data from CSV:
  ```python
  import pandas as pd

  df = pd.read_csv('data.csv')
  ```
- Basic DataFrame operations:
  ```python
  df.head()  # Display first 5 rows
  df.describe()  # Summary statistics
  df['column_name']  # Access specific column
  ```

## 3. Creating a Dataset in PyTorch
```python
import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

data = [i for i in range(100)]
dataset = CustomDataset(data)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
```

## 4. Defining a Model in PyTorch
```python
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleModel()
```

## 5. Training the Model
```python
import torch.optim as optim

criterion = nn.MSELoss()  # Loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Optimizer

for epoch in range(100):
    for batch in dataloader:
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(batch.float())  # Forward pass
        loss = criterion(outputs, batch.float().view(-1, 1))  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

    if epoch % 10 == 0:
        print(f'Epoch [{epoch}/100], Loss: {loss.item():.4f}')
```

## 6. Making Predictions
```python
model.eval()  # Set model to evaluation mode
with torch.no_grad():
    predictions = model(torch.tensor(data, dtype=torch.float32))
print(predictions)
```

## 7. Visualizing Data and Predictions with Matplotlib
```python
import matplotlib.pyplot as plt

# Example data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Plotting
plt.plot(x, y, label='Sin Curve')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.title('Sin Curve')
plt.legend()
plt.show()
```

## Important Functions and Methods to Remember

### PyTorch
- `torch.tensor(data)`: Convert data to tensor.
- `nn.Module`: Base class for all models.
- `nn.Linear(in_features, out_features)`: Fully connected layer.
- `torch.relu(tensor)`: Apply ReLU activation.
- `nn.MSELoss()`: Mean Squared Error loss function.
- `optim.Adam(parameters, lr)`: Adam optimizer.
- `model.train()`: Set model to training mode.
- `model.eval()`: Set model to evaluation mode.
- `optimizer.zero_grad()`: Zero all gradients.
- `loss.backward()`: Backpropagate the loss.
- `optimizer.step()`: Perform a single optimization step.

### NumPy
- `np.array(data)`: Create array.
- `np.random.rand(d0, d1, ..., dn)`: Random values in a given shape.
- `np.linspace(start, stop, num)`: Evenly spaced numbers over a specified interval.

### Pandas
- `pd.read_csv('file.csv')`: Load CSV file into DataFrame.
- `df.head()`: Display first 5 rows.
- `df.describe()`: Summary statistics.
- `df['column']`: Access specific column.

### Matplotlib
- `plt.plot(x, y, label)`: Plot data.
- `plt.xlabel('label')`: Set x-axis label.
- `plt.ylabel('label')`: Set y-axis label.
- `plt.title('title')`: Set plot title.
- `plt.legend()`: Display legend.
- `plt.show()`: Show plot.

## Resources
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [NumPy Documentation](https://numpy.org/doc/stable/)
- [Pandas Documentation](https://pandas.pydata.org/pandas-docs/stable/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
