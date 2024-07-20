
# Activation Functions in Neural Networks

Activation functions are crucial in neural networks as they introduce non-linearity into the model, allowing it to learn complex patterns. Here are the most common activation functions, their usage, and resources for further reading.

## 1. Sigmoid (Logistic) Activation Function
### Formula:
$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

### Characteristics:
- Output range: (0, 1)
- Commonly used in binary classification problems.
- Can cause vanishing gradient problem.

### Usage:
```python
import torch.nn as nn

# Using in a single layer
layer = nn.Linear(10, 5)
activation = nn.Sigmoid()
output = activation(layer(input))
```

### Resources:
- [Sigmoid Function - Wikipedia](https://en.wikipedia.org/wiki/Sigmoid_function)
- [Understanding Sigmoid Activation Function](https://towardsdatascience.com/understanding-sigmoid-function-79b8a4b4582c)

## 2. Tanh (Hyperbolic Tangent) Activation Function
### Formula:
$$	anh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

### Characteristics:
- Output range: (-1, 1)
- Zero-centered outputs, which can make optimization easier.

### Usage:
```python
import torch.nn as nn

# Using in a single layer
layer = nn.Linear(10, 5)
activation = nn.Tanh()
output = activation(layer(input))
```

### Resources:
- [Tanh Function - Wikipedia](https://en.wikipedia.org/wiki/Hyperbolic_function)
- [Understanding Tanh Activation Function](https://towardsdatascience.com/understanding-tanh-function-345107c5234b)

## 3. ReLU (Rectified Linear Unit) Activation Function
### Formula:
$$	ext{ReLU}(x) = \max(0, x)$$

### Characteristics:
- Output range: [0, ∞)
- Commonly used in hidden layers of neural networks.
- Helps mitigate the vanishing gradient problem.

### Usage:
```python
import torch.nn as nn

# Using in a single layer
layer = nn.Linear(10, 5)
activation = nn.ReLU()
output = activation(layer(input))
```

### Resources:
- [ReLU Function - Wikipedia](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))
- [Understanding ReLU Activation Function](https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/)

## 4. Leaky ReLU Activation Function
### Formula:
$$	ext{Leaky ReLU}(x) = \max(0.01x, x)$$

### Characteristics:
- Output range: (-∞, ∞)
- Allows a small gradient when the unit is not active, preventing dead neurons.

### Usage:
```python
import torch.nn as nn

# Using in a single layer
layer = nn.Linear(10, 5)
activation = nn.LeakyReLU(0.01)
output = activation(layer(input))
```

### Resources:
- [Leaky ReLU Function - Wikipedia](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)#Leaky_ReLU)
- [Understanding Leaky ReLU Activation Function](https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/)

## 5. Softmax Activation Function
### Formula:
$$\sigma(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}$$

### Characteristics:
- Output range: (0, 1), sums to 1
- Commonly used in the output layer for multi-class classification.

### Usage:
```python
import torch.nn as nn

# Using in a single layer
layer = nn.Linear(10, 5)
activation = nn.Softmax(dim=1)
output = activation(layer(input))
```

### Resources:
- [Softmax Function - Wikipedia](https://en.wikipedia.org/wiki/Softmax_function)
- [Understanding Softmax Activation Function](https://deepai.org/machine-learning-glossary-and-terms/softmax-layer)

## 6. Swish Activation Function
### Formula:
$$	ext{Swish}(x) = x \cdot \sigma(x) = x \frac{1}{1 + e^{-x}}$$

### Characteristics:
- Output range: (-∞, ∞)
- Self-gated activation function proposed by Google.

### Usage:
```python
import torch.nn as nn

# Using a custom implementation
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

layer = nn.Linear(10, 5)
activation = Swish()
output = activation(layer(input))
```

### Resources:
- [Swish Function - Wikipedia](https://en.wikipedia.org/wiki/Swish_function)
- [Understanding Swish Activation Function](https://ai.googleblog.com/2017/11/novel-activation-function-for-deep.html)

## 7. ELU (Exponential Linear Unit) Activation Function
### Formula:
$$	ext{ELU}(x) = 
\begin{cases} 
x & 	ext{if } x > 0 \\
lpha (e^x - 1) & 	ext{if } x \leq 0 
\end{cases}
$$

### Characteristics:
- Output range: (-α, ∞)
- Helps to bring mean activations closer to zero.

### Usage:
```python
import torch.nn as nn

# Using in a single layer
layer = nn.Linear(10, 5)
activation = nn.ELU(alpha=1.0)
output = activation(layer(input))
```

### Resources:
- [ELU Function - Wikipedia](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)#ELU)
- [Understanding ELU Activation Function](https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6)

## Important Notes:
- Different activation functions may perform better depending on the specific task and dataset.
- Experimentation and hyperparameter tuning are key to finding the best activation function for your model.

## Resources for Further Reading
- [Comprehensive Guide to Activation Functions](https://www.analyticsvidhya.com/blog/2021/04/activation-functions-and-its-types-which-is-better/)
- [Activation Functions in Neural Networks](https://medium.com/@shyam.sunder.r/activation-functions-in-neural-networks-7ddcb20ba8f9)
- [PyTorch Activation Functions Documentation](https://pytorch.org/docs/stable/nn.html#non-linear-activations)

