
'''
The code is an implementation of an image classification model using PyTorch. It downloads a dataset of pizza, steak, and sushi images, preprocesses the data, defines a convolutional neural network model, trains the model, and plots the loss and accuracy curves.The code is an implementation of an image classification model using PyTorch. It downloads a dataset of pizza, steak, and sushi images, preprocesses the data, defines a convolutional neural network model, trains the model, and plots the loss and accuracy curves.
'''

import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import os
import requests
import zipfile
from pathlib import Path
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def main():
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

    # Define data transformations
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

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

    # Create train and test data loaders
    train_data_loader = DataLoader(train_data, batch_size=32, num_workers=4, shuffle=True)
    test_data_loader = DataLoader(test_data, batch_size=32, num_workers=4, shuffle=False)

    print(f"Train data loader: {train_data_loader}")
    print(f"Test data loader: {test_data_loader}")

    class EnhancedVGG(nn.Module):
        def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
            super().__init__()
            self.conv_block_1 = nn.Sequential(
                nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_units),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
            )
            self.conv_block_2 = nn.Sequential(
                nn.Conv2d(hidden_units, hidden_units*2, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_units*2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
            )
            self.conv_block_3 = nn.Sequential(
                nn.Conv2d(hidden_units*2, hidden_units*4, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_units*4),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(hidden_units*4*28*28, 512),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(512, output_shape)
            )
        
        def forward(self, x: torch.Tensor):
            x = self.conv_block_1(x)
            x = self.conv_block_2(x)
            x = self.conv_block_3(x)
            x = self.classifier(x)
            return x

    torch.manual_seed(42)
    model = EnhancedVGG(input_shape=3, hidden_units=32, output_shape=len(train_data.classes)).to(device)
    model

    # Set random seeds
    torch.manual_seed(42) 
    torch.cuda.manual_seed(42)

    # Set number of epochs
    NUM_EPOCHS = 20

    # Setup loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    def train_step(model, dataloader, loss_fn, optimizer):
        model.train()
        train_loss, train_acc = 0, 0
        
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            y_pred_class = torch.argmax(y_pred, dim=1)
            train_acc += (y_pred_class == y).sum().item() / len(y_pred)

        train_loss /= len(dataloader)
        train_acc /= len(dataloader)
        return train_loss, train_acc

    def test_step(model, dataloader, loss_fn):
        model.eval()
        test_loss, test_acc = 0, 0
        
        with torch.inference_mode():
            for batch, (X, y) in enumerate(dataloader):
                X, y = X.to(device), y.to(device)
                test_pred_logits = model(X)
                loss = loss_fn(test_pred_logits, y)
                test_loss += loss.item()
                test_pred_labels = test_pred_logits.argmax(dim=1)
                test_acc += (test_pred_labels == y).sum().item() / len(test_pred_labels)
                
        test_loss /= len(dataloader)
        test_acc /= len(dataloader)
        return test_loss, test_acc

    from tqdm.auto import tqdm

    def train(model, train_dataloader, test_dataloader, optimizer, scheduler, loss_fn, epochs):
        results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
        
        for epoch in tqdm(range(epochs)):
            train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer)
            test_loss, test_acc = test_step(model, test_dataloader, loss_fn)
            
            print(
                f"Epoch: {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}"
            )

            scheduler.step()

            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["test_loss"].append(test_loss)
            results["test_acc"].append(test_acc)

        return results

    model_results = train(model, train_data_loader, test_data_loader, optimizer, scheduler, loss_fn, NUM_EPOCHS)

    # Plot loss curves
    def plot_loss_curves(results):
        train_loss = results['train_loss']
        train_accuracy = results['train_acc']
        test_loss = results['test_loss']
        test_accuracy = results['test_acc']
        epochs = range(len(results['train_loss']))

        plt.figure(figsize=(15, 7))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_loss, label='train_loss')
        plt.plot(epochs, test_loss, label='test_loss')
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_accuracy, label='train_accuracy')
        plt.plot(epochs, test_accuracy, label='test_accuracy')
        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.legend()

    plot_loss_curves(model_results)
    plt.show()

if __name__ == "__main__":
    main()
