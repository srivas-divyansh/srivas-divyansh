"""
Convolutional Neural Networks for Image Classification
Implementation of CNN models for MNIST and CIFAR-10 datasets using PyTorch

Author: Divyansh Srivastava
Description: This script implements two separate CNN models for image classification:
1. MNIST CNN - for digit classification (28x28 grayscale images)
2. CIFAR-10 CNN - for object classification (32x32 color images)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time


class MNISTNet(nn.Module):
    """
    CNN model for MNIST digit classification
    Architecture: 2 Conv layers + 2 Pooling layers + 2 FC layers
    """
    
    def __init__(self):
        super(MNISTNet, self).__init__()
        # First convolutional layer: 1 input channel (grayscale), 32 output channels, 3x3 kernel
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # Second convolutional layer: 32 input channels, 64 output channels, 3x3 kernel
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Max pooling layer: 2x2 kernel, stride 2
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout for regularization
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # Fully connected layers
        # After 2 pooling operations: 28x28 -> 14x14 -> 7x7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 classes for digits 0-9
    
    def forward(self, x):
        # First conv + ReLU + pooling
        x = self.pool(F.relu(self.conv1(x)))
        # Second conv + ReLU + pooling
        x = self.pool(F.relu(self.conv2(x)))
        # Dropout
        x = self.dropout1(x)
        
        # Flatten the tensor for fully connected layers
        x = x.view(-1, 64 * 7 * 7)
        
        # First FC layer + ReLU + dropout
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        
        # Output layer
        x = self.fc2(x)
        return x


class CIFAR10Net(nn.Module):
    """
    CNN model for CIFAR-10 image classification
    Architecture: 2 Conv layers + 2 Pooling layers + 3 FC layers
    """
    
    def __init__(self):
        super(CIFAR10Net, self).__init__()
        # First convolutional layer: 3 input channels (RGB), 64 output channels, 3x3 kernel
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        # Second convolutional layer: 64 input channels, 128 output channels, 3x3 kernel
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Max pooling layer: 2x2 kernel, stride 2
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout for regularization
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # Fully connected layers
        # After 2 pooling operations: 32x32 -> 16x16 -> 8x8
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)  # 10 classes for CIFAR-10
    
    def forward(self, x):
        # First conv + ReLU + pooling
        x = self.pool(F.relu(self.conv1(x)))
        # Second conv + ReLU + pooling
        x = self.pool(F.relu(self.conv2(x)))
        # Dropout
        x = self.dropout1(x)
        
        # Flatten the tensor for fully connected layers
        x = x.view(-1, 128 * 8 * 8)
        
        # First FC layer + ReLU + dropout
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        
        # Second FC layer + ReLU
        x = F.relu(self.fc2(x))
        
        # Output layer
        x = self.fc3(x)
        return x


def load_mnist_data(batch_size=64):
    """
    Load and preprocess MNIST dataset
    
    Args:
        batch_size (int): Batch size for data loaders
        
    Returns:
        tuple: (train_loader, test_loader)
    """
    # Define transforms for MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    # Download and load training data
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Download and load test data
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def load_cifar10_data(batch_size=64):
    """
    Load and preprocess CIFAR-10 dataset
    
    Args:
        batch_size (int): Batch size for data loaders
        
    Returns:
        tuple: (train_loader, test_loader)
    """
    # Define transforms for CIFAR-10
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),  # Data augmentation
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # CIFAR-10 mean and std
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Download and load training data
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Download and load test data
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def train_model(model, train_loader, criterion, optimizer, device, num_epochs=5):
    """
    Train the neural network model
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimization algorithm
        device: Device to run on (CPU/GPU)
        num_epochs (int): Number of training epochs
        
    Returns:
        list: Training accuracies for each epoch
    """
    model.train()
    train_accuracies = []
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 30)
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, target)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Calculate statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # Print progress every 100 batches
            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx:4d}/{len(train_loader):4d} | '
                      f'Loss: {loss.item():.6f} | '
                      f'Accuracy: {100 * correct / total:.2f}%')
        
        epoch_accuracy = 100 * correct / total
        train_accuracies.append(epoch_accuracy)
        
        print(f'Epoch {epoch + 1} Summary:')
        print(f'Average Loss: {running_loss / len(train_loader):.6f}')
        print(f'Training Accuracy: {epoch_accuracy:.2f}%')
    
    return train_accuracies


def evaluate_model(model, test_loader, device, class_names=None):
    """
    Evaluate the trained model on test data
    
    Args:
        model: Trained neural network model
        test_loader: Test data loader
        device: Device to run on (CPU/GPU)
        class_names (list): List of class names for detailed reporting
        
    Returns:
        float: Test accuracy
    """
    model.eval()
    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # Calculate per-class accuracy
            c = (predicted == target).squeeze()
            for i in range(target.size(0)):
                label = target[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    # Overall accuracy
    test_accuracy = 100 * correct / total
    print(f'\nTest Accuracy: {test_accuracy:.2f}% ({correct}/{total})')
    
    # Per-class accuracy
    if class_names:
        print('\nPer-class Accuracy:')
        for i in range(10):
            if class_total[i] > 0:
                class_acc = 100 * class_correct[i] / class_total[i]
                print(f'{class_names[i]}: {class_acc:.2f}%')
    
    return test_accuracy


def main():
    """
    Main function to train and evaluate both CNN models
    """
    print("=" * 80)
    print("CNN Image Classification - MNIST and CIFAR-10")
    print("=" * 80)
    
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # CIFAR-10 class names
    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                      'dog', 'frog', 'horse', 'ship', 'truck']
    
    mnist_classes = [str(i) for i in range(10)]  # '0', '1', ..., '9'
    
    # Training parameters
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 5
    
    # ========================================
    # MNIST Classification
    # ========================================
    print("\n" + "=" * 50)
    print("MNIST DIGIT CLASSIFICATION")
    print("=" * 50)
    
    # Load MNIST data
    print("Loading MNIST dataset...")
    mnist_train_loader, mnist_test_loader = load_mnist_data(batch_size)
    print(f"MNIST dataset loaded: {len(mnist_train_loader.dataset)} training samples, "
          f"{len(mnist_test_loader.dataset)} test samples")
    
    # Initialize MNIST model
    mnist_model = MNISTNet().to(device)
    mnist_criterion = nn.CrossEntropyLoss()
    mnist_optimizer = optim.Adam(mnist_model.parameters(), lr=learning_rate)
    
    print(f"\nMNIST Model Architecture:")
    print(f"Input: 28x28 grayscale images")
    print(f"Conv1: 1 -> 32 channels, 3x3 kernel")
    print(f"Conv2: 32 -> 64 channels, 3x3 kernel")
    print(f"FC1: {64*7*7} -> 128 neurons")
    print(f"FC2: 128 -> 10 classes")
    
    # Train MNIST model
    print(f"\nTraining MNIST model for {num_epochs} epochs...")
    start_time = time.time()
    mnist_train_acc = train_model(mnist_model, mnist_train_loader, mnist_criterion, 
                                 mnist_optimizer, device, num_epochs)
    mnist_train_time = time.time() - start_time
    
    # Evaluate MNIST model
    print(f"\nEvaluating MNIST model...")
    mnist_test_acc = evaluate_model(mnist_model, mnist_test_loader, device, mnist_classes)
    
    print(f"\nMNIST Training completed in {mnist_train_time:.2f} seconds")
    print(f"Final Training Accuracy: {mnist_train_acc[-1]:.2f}%")
    print(f"Test Accuracy: {mnist_test_acc:.2f}%")
    
    # ========================================
    # CIFAR-10 Classification
    # ========================================
    print("\n" + "=" * 50)
    print("CIFAR-10 IMAGE CLASSIFICATION")
    print("=" * 50)
    
    # Load CIFAR-10 data
    print("Loading CIFAR-10 dataset...")
    cifar10_train_loader, cifar10_test_loader = load_cifar10_data(batch_size)
    print(f"CIFAR-10 dataset loaded: {len(cifar10_train_loader.dataset)} training samples, "
          f"{len(cifar10_test_loader.dataset)} test samples")
    
    # Initialize CIFAR-10 model
    cifar10_model = CIFAR10Net().to(device)
    cifar10_criterion = nn.CrossEntropyLoss()
    cifar10_optimizer = optim.Adam(cifar10_model.parameters(), lr=learning_rate)
    
    print(f"\nCIFAR-10 Model Architecture:")
    print(f"Input: 32x32 RGB images")
    print(f"Conv1: 3 -> 64 channels, 3x3 kernel")
    print(f"Conv2: 64 -> 128 channels, 3x3 kernel")
    print(f"FC1: {128*8*8} -> 512 neurons")
    print(f"FC2: 512 -> 128 neurons")
    print(f"FC3: 128 -> 10 classes")
    
    # Train CIFAR-10 model
    print(f"\nTraining CIFAR-10 model for {num_epochs} epochs...")
    start_time = time.time()
    cifar10_train_acc = train_model(cifar10_model, cifar10_train_loader, cifar10_criterion, 
                                   cifar10_optimizer, device, num_epochs)
    cifar10_train_time = time.time() - start_time
    
    # Evaluate CIFAR-10 model
    print(f"\nEvaluating CIFAR-10 model...")
    cifar10_test_acc = evaluate_model(cifar10_model, cifar10_test_loader, device, cifar10_classes)
    
    print(f"\nCIFAR-10 Training completed in {cifar10_train_time:.2f} seconds")
    print(f"Final Training Accuracy: {cifar10_train_acc[-1]:.2f}%")
    print(f"Test Accuracy: {cifar10_test_acc:.2f}%")
    
    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 50)
    print("FINAL RESULTS SUMMARY")
    print("=" * 50)
    
    print(f"MNIST CNN Performance:")
    print(f"  - Training time: {mnist_train_time:.2f} seconds")
    print(f"  - Final training accuracy: {mnist_train_acc[-1]:.2f}%")
    print(f"  - Test accuracy: {mnist_test_acc:.2f}%")
    print(f"  - Training accuracy progression: {[f'{acc:.1f}%' for acc in mnist_train_acc]}")
    
    print(f"\nCIFAR-10 CNN Performance:")
    print(f"  - Training time: {cifar10_train_time:.2f} seconds")
    print(f"  - Final training accuracy: {cifar10_train_acc[-1]:.2f}%")
    print(f"  - Test accuracy: {cifar10_test_acc:.2f}%")
    print(f"  - Training accuracy progression: {[f'{acc:.1f}%' for acc in cifar10_train_acc]}")
    
    print(f"\nModel Parameters:")
    mnist_params = sum(p.numel() for p in mnist_model.parameters() if p.requires_grad)
    cifar10_params = sum(p.numel() for p in cifar10_model.parameters() if p.requires_grad)
    print(f"  - MNIST model parameters: {mnist_params:,}")
    print(f"  - CIFAR-10 model parameters: {cifar10_params:,}")
    
    print("\n" + "=" * 80)
    print("Training and evaluation completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()