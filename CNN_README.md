# CNN Image Classification

A PyTorch implementation of Convolutional Neural Networks (CNNs) for image classification on MNIST and CIFAR-10 datasets.

## Overview

This project implements two separate CNN models:
- **MNIST CNN**: For handwritten digit classification (0-9)
- **CIFAR-10 CNN**: For object classification (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)

## Features

- **Two CNN Architectures**: Custom models optimized for each dataset
- **Automatic Dataset Loading**: Uses torchvision for seamless data handling
- **Data Preprocessing**: Proper normalization and data augmentation
- **Training Pipeline**: Complete training loop with progress monitoring
- **Evaluation Metrics**: Comprehensive accuracy reporting
- **GPU Support**: Automatic GPU detection and usage if available

## Requirements

- Python 3.7+
- PyTorch 1.9.0+
- torchvision 0.10.0+
- numpy 1.21.0+

## Installation

1. Clone this repository:
```bash
git clone https://github.com/srivas-divyansh/srivas-divyansh.git
cd srivas-divyansh
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install torch torchvision numpy
```

## Usage

Simply run the main script:
```bash
python cnn_image_classification.py
```

The script will automatically:
1. Download the MNIST and CIFAR-10 datasets
2. Train both CNN models
3. Evaluate performance on test sets
4. Display comprehensive results

## Model Architectures

### MNIST CNN
- Input: 28x28 grayscale images
- Conv Layer 1: 1 → 32 channels (3x3 kernel)
- MaxPool: 2x2
- Conv Layer 2: 32 → 64 channels (3x3 kernel)
- MaxPool: 2x2
- FC Layer 1: 3136 → 128 neurons
- FC Layer 2: 128 → 10 classes
- Activation: ReLU
- Regularization: Dropout (0.25, 0.5)

### CIFAR-10 CNN
- Input: 32x32 RGB images
- Conv Layer 1: 3 → 64 channels (3x3 kernel)
- MaxPool: 2x2
- Conv Layer 2: 64 → 128 channels (3x3 kernel)
- MaxPool: 2x2
- FC Layer 1: 8192 → 512 neurons
- FC Layer 2: 512 → 128 neurons
- FC Layer 3: 128 → 10 classes
- Activation: ReLU
- Regularization: Dropout (0.25, 0.5)

## Sample Output

```
================================================================================
CNN Image Classification - MNIST and CIFAR-10
================================================================================
Using device: cpu

==================================================
MNIST DIGIT CLASSIFICATION
==================================================
Loading MNIST dataset...
MNIST dataset loaded: 60000 training samples, 10000 test samples

MNIST Model Architecture:
Input: 28x28 grayscale images
Conv1: 1 -> 32 channels, 3x3 kernel
Conv2: 32 -> 64 channels, 3x3 kernel
FC1: 3136 -> 128 neurons
FC2: 128 -> 10 classes

Training MNIST model for 5 epochs...

Epoch 1/5
------------------------------
Batch    0/ 938 | Loss: 2.302274 | Accuracy: 12.50%
Batch  100/ 938 | Loss: 0.421563 | Accuracy: 87.34%
...

Final Training Accuracy: 97.85%
Test Accuracy: 98.23%

==================================================
CIFAR-10 IMAGE CLASSIFICATION
==================================================
...
```

## Performance

Expected performance after 5 epochs:
- **MNIST**: ~98% test accuracy
- **CIFAR-10**: ~70-75% test accuracy

## Dataset Information

### MNIST
- 60,000 training images
- 10,000 test images
- 28x28 grayscale
- 10 classes (digits 0-9)

### CIFAR-10
- 50,000 training images
- 10,000 test images
- 32x32 color (RGB)
- 10 classes (objects)

## Customization

You can modify the following parameters in the `main()` function:
- `batch_size`: Training batch size (default: 64)
- `learning_rate`: Learning rate for optimization (default: 0.001)
- `num_epochs`: Number of training epochs (default: 5)

## File Structure

```
.
├── cnn_image_classification.py  # Main implementation
├── requirements.txt             # Dependencies
├── .gitignore                  # Git ignore file
├── README.md                   # This file
└── data/                       # Dataset directory (auto-created)
```

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

This project is open source and available under the MIT License.

## Author

**Divyansh Srivastava**
- GitHub: [@srivas-divyansh](https://github.com/srivas-divyansh)
- Email: srivas.divyansh22@gmail.com

## Acknowledgments

- PyTorch team for the excellent deep learning framework
- Yann LeCun et al. for the MNIST dataset
- Alex Krizhevsky for the CIFAR-10 dataset