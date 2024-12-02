# Deep-Learning

## 1. Artificial Neural Network (ANN) for Churn Prediction

This project demonstrates how to use an Artificial Neural Network (ANN) to predict customer churn. The dataset used contains customer information such as credit score, geography, gender, age, balance, and whether the customer exited or not. The goal is to predict if a customer will exit (churn) based on these features.

The code uses PyTorch to implement the ANN and scikit-learn for data preprocessing.

## Features
- **Data Preprocessing**: Handles missing data, encodes categorical data, scales numerical features.
- **Model Training**: Trains a simple ANN with one hidden layer using PyTorch.
- **Evaluation**: Evaluates the model using accuracy and confusion matrix.

## Requirements

To run this project, you'll need the following libraries:

- numpy
- pandas
- torch
- scikit-learn

You can install the required dependencies using the following command:

```bash
pip install numpy pandas torch scikit-learn

```

## 2. CNN - Dog vs. Cat Image Classifier

This project demonstrates how to build a Convolutional Neural Network (CNN) to classify images of dogs and cats using PyTorch. The model is based on the ResNet18 architecture and is trained on a dataset of dog and cat images. After training, the model can be used to classify new images using a web interface built with Streamlit.

## Features
- **Model Training:** Trains a CNN model using the ResNet18 architecture.
- **Image Classification:** Classifies whether the uploaded image is of a dog or a cat.
- **Web Interface:** Allows users to upload images and see predictions in real time.

## Requirements
To run this project, you'll need the following libraries:

- torch
- torchvision
- pillow
- streamlit

You can install the required dependencies using the following command:

```bash
pip install torch torchvision pillow streamlit


