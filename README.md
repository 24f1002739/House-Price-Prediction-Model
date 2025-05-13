# 🏠 House Price Predictor using Keras & TensorFlow
A machine learning regression model built using Keras (TensorFlow backend) to predict house prices based on numerical input features. This project showcases the use of a deep neural network on structured 2D data, with end-to-end steps including data preprocessing, model training, evaluation, and visualization.

### 📊 Table of Contents
Overview

Features

Dataset

Technologies Used

Results

Visualizations

Future Work

License

### 📌 Overview
This project demonstrates a supervised learning approach to predict house prices using a fully connected deep learning model. It utilizes a structured dataset of housing features and aims to minimize prediction error through Mean Squared Error (MSE) loss optimization.

### 🚀 Features
Clean and modular code structure using Python

End-to-end machine learning pipeline:

Data cleaning and preprocessing

Train/test split

Feature normalization

Deep learning model using Keras Sequential API

Early stopping to prevent overfitting

Visualization of training loss and prediction performance

### 🗂 Dataset
Format: CSV file (.csv)

Contains numerical input features (e.g., area, number of rooms, etc.)

Target variable: price

Assumes no categorical variables (purely numerical features)

Note: The dataset is not public. Ensure your CSV file is named and formatted correctly.

### 🛠 Technologies Used
Python 3.x

TensorFlow (Keras API)

Pandas

NumPy

Matplotlib

Scikit-learn

### 📈 Results
Model evaluation using:

Mean Absolute Error (MAE)

Mean Squared Error (MSE)

### Plots:

Training vs Validation Loss curve

Actual vs Predicted Price scatter plot

### 📊 Visualizations
Loss Curve	Actual vs Predicted

Add images in the /assets/ folder if using GitHub.

### 🧠 Future Work
Include categorical feature encoding

Implement advanced model tuning (Grid Search, Bayesian Optimization)

Try other algorithms (XGBoost, LightGBM)

Deploy the model using Flask or FastAPI

Convert model to TensorFlow Lite for mobile deployment

### 📄 License
This project is licensed under the MIT License.
