# Handwritten Digit Recognition using Neural Networks

## Project Overview
This project implements a neural network for recognizing handwritten digits (0-9) using the MNIST dataset. The model achieves over 97% accuracy on validation data.

## Features
- Data preprocessing and normalization
- Neural network with dense layers
- Model training with early stopping
- Performance evaluation
- Prediction visualization

## Dataset
The project uses the MNIST dataset from Kaggle:
- Training data: 42,000 samples (train.csv)
- Test data: 28,000 samples (test.csv)

## Model Architecture
- Input layer: Flattened 28×28×1 image
- Hidden layer 1: 128 neurons, ReLU activation
- Hidden layer 2: 64 neurons, ReLU activation
- Output layer: 10 neurons, Softmax activation

