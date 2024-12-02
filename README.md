# Perceptron & Logistic Regression for Breast Cancer

This project explores the implementation and evaluation of two foundational machine learning models: the **Perceptron Classifier** and **Logistic Regression**. The goal is to demonstrate the use of Python for building, training, and analyzing these models for binary classification tasks, specifically on the **Breast Cancer Dataset**.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Technologies Used](#technologies-used)
4. [Setup](#setup)
5. [Results](#results)

## Project Overview
This project focuses on working with two models for binary classification tasks:
1. **Perceptron Classifier**: A simple linear classifier that uses a decision boundary defined by a weight vector and bias.
2. **Logistic Regression**: A probabilistic classifier that uses the sigmoid function to output class probabilities.

We apply these models to the **Breast Cancer Dataset** from sklearn, aiming to predict whether a tumor is malignant or benign based on various features.

### Perceptron Classifier
The **Perceptron Classifier** is a binary classifier that learns a decision boundary to separate two classes. The decision rule is defined as:
\[
$W \cdot X + b = 0$
\]
where **W** is the weight vector and **b** is the bias term. The model updates its weights and bias using the gradient descent algorithm based on misclassified samples.

### Logistic Regression
**Logistic Regression** uses a logistic function (sigmoid) to model the probability of class membership. Unlike the Perceptron, it outputs probabilities rather than making hard predictions.

### Detailed Report
You can access the comprehensive analysis by clicking here: [Detailed Report](./Perceptron%20Classifier%20and%20Logistic%20Regression.pdf)

## Features
- **Perceptron Classifier Implementation**: Implements a custom Perceptron class for binary classification using gradient descent.
- **Logistic Regression**: Implements logistic regression using sklearn's `LogisticRegression` class for comparison.
- **Accuracy Tracking**: Both models track and visualize training and testing accuracy across epochs.
- **Data Normalization**: The dataset is normalized to improve model performance.
- **Dataset**: The project uses the UCI ML Breast Cancer Wisconsin (Diagnostic) Dataset.

## Technologies Used
- **Python**: Programming language used for model implementation.
- **NumPy**: Used for numerical operations and matrix manipulations.
- **scikit-learn**: Provides dataset loading, model training, and evaluation tools.
- **Matplotlib**: Used for plotting and visualizing accuracy over epochs.
- **Pandas**: For data manipulation and analysis.

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Project**:
   Use Jupyter Notebook (`jupyter notebook`) to open and run the project.

## Results

### Perceptron Classifier Results

The Perceptron model was trained on the **Breast Cancer Wisconsin Diagnostic dataset**. The results were evaluated based on both the **training accuracy** and **testing accuracy**. The following key observations were made:

#### Training Accuracy
As the model was trained over 900 epochs, the training accuracy steadily increased. The final training accuracy achieved was **98.25%**.

#### Testing Accuracy
On the testing dataset, our Perceptron model achieved a high accuracy of **98.25%** after 900 epochs.

#### Learning Dynamics
The accuracy of our Perceptron model improved over time with each epoch, and the plot in the detailed report visualizes the accuracy progression.

### Sklearn Perceptron Classifier Results

We also trained the **scikit-learn Perceptron** model using the same dataset. The final accuracy achieved by the model was **99.12%** on the test dataset.



