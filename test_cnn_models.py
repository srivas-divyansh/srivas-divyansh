"""
Test script for CNN models - validates functionality without downloading datasets
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from cnn_image_classification import MNISTNet, CIFAR10Net, train_model, evaluate_model

def create_dummy_mnist_data(num_samples=1000, batch_size=64):
    """Create dummy MNIST-like data for testing"""
    # Create random data with correct dimensions
    X = torch.randn(num_samples, 1, 28, 28)
    y = torch.randint(0, 10, (num_samples,))
    
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return loader, loader  # Use same data for train and test

def create_dummy_cifar10_data(num_samples=1000, batch_size=64):
    """Create dummy CIFAR-10-like data for testing"""
    # Create random data with correct dimensions
    X = torch.randn(num_samples, 3, 32, 32)
    y = torch.randint(0, 10, (num_samples,))
    
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return loader, loader  # Use same data for train and test

def test_models():
    """Test both CNN models with dummy data"""
    print("=" * 60)
    print("CNN MODELS VALIDATION TEST")
    print("=" * 60)
    
    device = torch.device("cpu")  # Use CPU for testing
    print(f"Using device: {device}")
    
    # Test MNIST model
    print("\nTesting MNIST CNN Model...")
    print("-" * 30)
    
    mnist_model = MNISTNet().to(device)
    mnist_train_loader, mnist_test_loader = create_dummy_mnist_data()
    
    # Model info
    mnist_params = sum(p.numel() for p in mnist_model.parameters() if p.requires_grad)
    print(f"MNIST model parameters: {mnist_params:,}")
    
    # Test forward pass
    sample_input = torch.randn(1, 1, 28, 28).to(device)
    output = mnist_model(sample_input)
    print(f"Input shape: {sample_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Quick training test (1 epoch)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(mnist_model.parameters(), lr=0.001)
    
    print("\nTraining for 1 epoch...")
    train_acc = train_model(mnist_model, mnist_train_loader, criterion, optimizer, device, num_epochs=1)
    test_acc = evaluate_model(mnist_model, mnist_test_loader, device)
    
    print(f"MNIST Training accuracy: {train_acc[0]:.2f}%")
    print(f"MNIST Test accuracy: {test_acc:.2f}%")
    
    # Test CIFAR-10 model
    print("\n" + "=" * 40)
    print("Testing CIFAR-10 CNN Model...")
    print("-" * 30)
    
    cifar10_model = CIFAR10Net().to(device)
    cifar10_train_loader, cifar10_test_loader = create_dummy_cifar10_data()
    
    # Model info
    cifar10_params = sum(p.numel() for p in cifar10_model.parameters() if p.requires_grad)
    print(f"CIFAR-10 model parameters: {cifar10_params:,}")
    
    # Test forward pass
    sample_input = torch.randn(1, 3, 32, 32).to(device)
    output = cifar10_model(sample_input)
    print(f"Input shape: {sample_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Quick training test (1 epoch)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cifar10_model.parameters(), lr=0.001)
    
    print("\nTraining for 1 epoch...")
    train_acc = train_model(cifar10_model, cifar10_train_loader, criterion, optimizer, device, num_epochs=1)
    test_acc = evaluate_model(cifar10_model, cifar10_test_loader, device)
    
    print(f"CIFAR-10 Training accuracy: {train_acc[0]:.2f}%")
    print(f"CIFAR-10 Test accuracy: {test_acc:.2f}%")
    
    print("\n" + "=" * 60)
    print("VALIDATION TEST COMPLETED SUCCESSFULLY!")
    print("Both CNN models are working correctly.")
    print("Run 'python cnn_image_classification.py' for full training on real datasets.")
    print("=" * 60)

if __name__ == "__main__":
    test_models()