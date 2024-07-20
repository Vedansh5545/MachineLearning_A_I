import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import os
import pathlib 
import requests
import zipfile
from pathlib import Path

import torch.nn as nn
import matplotlib.pyplot as plt

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.__version__)

# Setup path to data folder
data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"

# If the image folder doesn't exist, download and extract the data
if not image_path.is_dir():
    print(f"Did not find {image_path} directory, creating one...")
    image_path.mkdir(parents=True, exist_ok=True)
    
    # Download pizza, steak, sushi data
    url = "https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip"
    zip_path = data_path / "pizza_steak_sushi.zip"
    with open(zip_path, "wb") as f:
        request = requests.get(url)
        print("Downloading pizza, steak, sushi data...")
        f.write(request.content)

    # Unzip pizza, steak, sushi data
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        print("Unzipping pizza, steak, sushi data...") 
        zip_ref.extractall(image_path)

def walk_through_dir(dir_path):
    num_dirs = 0
    num_images = 0
    for dirpath, dirnames, filenames in os.walk(dir_path):
        num_dirs += len(dirnames)
        num_images += len(filenames)
    print(f"There are {num_dirs} directories and {num_images} images in '{dir_path}'.")

walk_through_dir(image_path)

# Get a random image path
image_path_list = list(image_path.glob("*/*/*.jpg"))
random_image_path = random.choice(image_path_list)
image_class = random_image_path.parent.stem

# Open and display the image
img = Image.open(random_image_path)
print(f"Image path: {random_image_path}")
print(f"Image class: {image_class}")
print(f"Image size: {img.size}")

# Display the image
plt.figure(figsize=(10, 7))
plt.imshow(img)
plt.axis(False)

# Define data transformations
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])

# Plot transformed images
def plot_transformed_images(image_paths, transform, n=3, seed=42):
    random.seed(seed)
    random_image_paths = random.sample(image_paths, k=n)
    for image_path in random_image_paths:
        with Image.open(image_path) as img:
            fig, axs = plt.subplots(1, 2, figsize=(10, 7))
            axs[0].imshow(img)
            axs[0].set_title("Original Image")
            axs[0].axis("off")
            transformed_image = transform(img).permute(1, 2, 0)
            axs[1].imshow(transformed_image)
            axs[1].set_title("Transformed Image")
            axs[1].axis("off")
            fig.suptitle(f"{image_path.parent.name}")
            plt.show()

plot_transformed_images(image_path_list, transform=data_transform, n=3)

# Define train and test directories
train_dir = image_path / "train"
test_dir = image_path / "test"

# Create train and test datasets
train_data = datasets.ImageFolder(root=train_dir, transform=data_transform, target_transform=None)
test_data = datasets.ImageFolder(root=test_dir, transform=data_transform, target_transform=None)

print(f"Train data: \n{train_data} \nTest data: \n{test_data}")

# Get class names and class dictionary
class_names = train_data.classes
class_dict = train_data.class_to_idx
print(f"Class names: {class_names}")
print(f"Class dictionary: {class_dict}")

# Get an example image and label from the train dataset
img, label = train_data[0]
print(f"Image Tensor: {img}, Label: {label}")
print(f"Image shape: {img.shape}, Label: {label}")
print(f"Image data type: {img.dtype}, Label data type: {label}")
print(f"Image label: {class_names[label]}")

# Permute the image dimensions and display
image_permute = img.permute(1, 2, 0)
print(f"Original image shape: {img.shape}, Permuted image shape: {image_permute.shape}")

plt.figure(figsize=(10, 7))
plt.imshow(image_permute)
plt.axis(False)
plt.title(class_names[label])
plt.show()

# Create train and test data loaders
train_data_loader = DataLoader(train_data, batch_size=1, num_workers=1, shuffle=True)
test_data_loader = DataLoader(test_data, batch_size=1, num_workers=1, shuffle=False)

print(f"Train data loader: {train_data_loader}")
print(f"Test data loader: {test_data_loader}")

# Get an example batch from the train data loader
img, label = next(iter(train_data_loader))
print(f"Image shape: {img.shape}, Label shape: {label.shape}")
print(f"Image data type: {img.dtype}, Label data type: {label.dtype}")
