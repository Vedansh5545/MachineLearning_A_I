


import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import tqdm

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the data
transform = transforms.Compose([transforms.ToTensor()])

train_data = torchvision.datasets.MNIST(root='data', train=True, download=True, transform=transform)
test_data = torchvision.datasets.MNIST(root='data', train=False, download=True, transform=transform)

BATCH_SIZE = 32

train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_data_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# Define the model
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(8, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=8*7*7, out_features=10)  # Change out_features to 10
        )
        
    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.classifier(x)
        return x

model = Model().to(device)
print(model)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
def train_model(model, data_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm.tqdm(data_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    return running_loss / len(data_loader)

# Test the model
def test_model(model, data_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm.tqdm(data_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return running_loss / len(data_loader), correct / total

# Train the model
EPOCHS = 10

train_loss = []
print('Training the model...')
for epoch in range(EPOCHS):
    loss = train_model(model, train_data_loader, criterion, optimizer)
    train_loss.append(loss)
    print(f'Epoch: {epoch + 1}, Loss: {loss}')

# Test the model
print('Testing the model...')
test_loss, test_accuracy = test_model(model, test_data_loader, criterion)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

# Plot the training loss
plt.plot(train_loss)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()


# Make predictions on test samples
def make_predictions(model, data_loader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)
            outputs = model(images)
            predictions.append(outputs)
    return torch.cat(predictions, dim=0)

# Class names for MNIST dataset
class_names = [str(i) for i in range(10)]

# Get prediction probabilities and labels
pred_probs = make_predictions(model, test_data_loader)
pred_classes = pred_probs.argmax(dim=1)

# Plot predictions
plt.figure(figsize=(9, 9))
nrows = 3
ncols = 3
for i, (sample, label) in enumerate(zip(test_data.data[:9], test_data.targets[:9])):
    # Create a subplot
    plt.subplot(nrows, ncols, i+1)

    # Plot the target image
    plt.imshow(sample, cmap="gray")

    # Find the prediction label (in text form, e.g. "7")
    pred_label = class_names[pred_classes[i].item()]

    # Get the truth label (in text form, e.g. "3")
    truth_label = class_names[label.item()] 

    # Create the title text of the plot
    title_text = f"Pred: {pred_label} | Truth: {truth_label}"
  
    # Check for equality and change title colour accordingly
    if pred_label == truth_label:
        plt.title(title_text, fontsize=10, c="g")  # green text if correct
    else:
        plt.title(title_text, fontsize=10, c="r")  # red text if wrong
    plt.axis(False)

plt.show()