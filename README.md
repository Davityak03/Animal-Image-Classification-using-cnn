# Animal Species Classification Using CNN

## Overview
Animage is a machine learning project that focuses on image classification using Convolutional Neural Networks (CNNs). The project aims to classify images into predefined categories with high accuracy by leveraging the powerful feature extraction capabilities of CNNs.

## Dataset
Link for dataset https://drive.google.com/drive/folders/1Oe7fsq0Ej3FUZqvl6lLmA4726Qq8E1qa?usp=sharing

## Features
- **Image Preprocessing**: Resizing, normalization, and data augmentation techniques to enhance the model's performance.
- **CNN Architecture**: A robust CNN model designed for image classification.
- **Training and Validation**: Efficient training pipeline with validation steps to monitor performance.
- **Performance Metrics**: Accuracy, precision, recall, and F1-score for evaluating the model.

## Requirements
- Python 3.7 or higher
- TensorFlow 2.0 or higher
- Keras
- NumPy
- Pandas
- Matplotlib
- scikit-learn

## Model Architecture
The CNN architecture consists of:
- **Convolutional Layers**: For feature extraction.
- **Pooling Layers**: For down-sampling.
- **Fully Connected Layers**: For classification.

The model is designed to handle the complexity of image data and achieve high classification accuracy.

## Data Augmentation
Data augmentation techniques such as rotation, zoom, and horizontal flipping are applied to enhance the model's generalization capability.

## Evaluation Metrics
- **Accuracy**: Measures the proportion of correctly classified images.
- **Precision**: Measures the accuracy of positive predictions.
- **Recall**: Measures the ability of the model to identify all relevant instances.
- **F1-Score**: Harmonic mean of precision and recall.
