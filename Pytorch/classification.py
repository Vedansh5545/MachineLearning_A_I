"""
pytorch neural network for classification
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

print(torch.__version__)

device = "cuda" if torch.cuda.is_available() else "cpu"

print(device)
