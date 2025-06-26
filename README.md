# 🧠 MNIST Digit Classifier (PyTorch)

A simple but powerful Convolutional Neural Network (CNN) built using PyTorch to classify handwritten digits (0–9) from the MNIST dataset.

## 🚀 Features
- Built with PyTorch
- Achieves ~98%+ accuracy on test set
- Modular code structure for easy reuse
- Supports GPU acceleration
- Separates training, evaluation, and data loading for clarity

## 📁 Project Structure

```
mnist_classifier/
├── data_loader.py       # Loads and preprocesses the MNIST dataset
├── model.py             # CNN architecture definition
├── train.py             # Model training script
├── evaluate.py          # Model evaluation script
```

## 🛠️ Requirements

- Python 3.8+
- torch
- torchvision
- matplotlib (optional, for visualizations)

Install dependencies with:

```bash
pip install torch torchvision matplotlib
```

## ▶️ How to Run

### 1. Train the Model

```bash
python train.py
```

This will:
- Download the MNIST dataset
- Train the CNN model
- Save the model to `mnist_cnn.pth`

### 2. Evaluate the Model

```bash
python evaluate.py
```

This will:
- Load the saved model
- Print accuracy on the test dataset

## 📊 Sample Output

```
Epoch 1, Loss: 0.3452
Epoch 2, Loss: 0.1221
...
Test Accuracy: 98.45%
```

## 🧠 Model Architecture

- Conv2d → ReLU → Conv2d → ReLU → MaxPool → Dropout → FC → ReLU → FC

## 📌 License

MIT License
