# Handwritten Digit Recognition using Neural Networks

## Project Overview
This project implements a **Convolutional Neural Network (CNN)**–based system for recognizing handwritten digits (0–9) using the **MNIST dataset**.  
The model achieves **up to 99.5% accuracy** on the test set and is optimized to minimize overfitting through regularization and data augmentation techniques.  
The system is further extended with a **FastAPI-based REST service** to enable real-time digit inference.

## Features
- CNN-based handwritten digit classification using TensorFlow
- Data preprocessing, normalization, and augmentation
- Overfitting reduction using dropout and augmentation (~85% reduction)
- Model training with early stopping and validation monitoring
- <50 ms inference latency

## Dataset
The project uses the MNIST dataset from Kaggle:
- Training data: 42,000 samples (train.csv)
- Test data: 28,000 samples (test.csv)
Due to size constraints, the dataset is **not included** in this repository.

## Results
- Test accuracy: **99.5%**
- Overfitting reduced by approximately **85%**
- Real-time inference latency: **<50 ms**

## Notes
- Trained model weights and datasets are excluded from version control.
- The repository focuses on reproducible training and inference pipelines.


