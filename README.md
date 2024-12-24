# Breast Cancer Classification Using Artificial Neural Networks (ANN)

## Overview
This Assignment explores the use of Artificial Neural Networks (ANNs) to classify breast cancer tumors as benign or malignant. Using the Breast Cancer Wisconsin (Diagnostic) dataset, the model aims to assist in early detection and diagnosis, improving patient outcomes and supporting clinical decision-making.


**Name**: Calistus Chukwuebuka Ndubuisi  
**Class**: BAN6440 - Applied Machine Learning for Analytics  
**Date**: December 18, 2024

---

## Dataset
- **Source**: [Breast Cancer Wisconsin (Diagnostic) Data Set](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
- **Attributes**:
  - **ID Number**: Unique identifier for each record.
  - **Diagnosis**: Binary classification (M = Malignant, B = Benign).
  - **30 Numerical Features**: Descriptions of cell nuclei based on digitized images, including:
    - Radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension.

---

## Project Workflow

### 1. Data Preprocessing
- **Loading Data**: The dataset was loaded and inspected to understand its structure and attributes.
- **Exploratory Data Analysis (EDA)**:
  - Identified missing values and unnecessary columns (e.g., `Unnamed: 32`).
  - Visualized data distributions to understand feature variance.
- **Handling Missing Values**: Missing values were imputed using the mean of the respective features.
- **Feature Selection**: Retained relevant features (`diagnosis` and 30 numerical columns).
- **Normalization**: Standardized the dataset using `StandardScaler` to optimize ANN training.

### 2. Model Building
- Designed a sequential ANN using TensorFlow and Keras:
  - **Input Layer**: Matching the number of features in the dataset.
  - **Hidden Layers**: Two layers with 16 neurons each and ReLU activation.
  - **Output Layer**: A single neuron with a Sigmoid activation function for binary classification.
- Compiled the model using the Adam optimizer and binary cross-entropy loss.

### 3. Model Training
- **Training**: Trained the model over 100 epochs with a batch size of 32.
- **Validation**: Used 20% of the dataset as a validation set to monitor performance during training.

### 4. Model Evaluation
- **Performance Metrics**:
  - Accuracy, precision, recall, F1-score.
- **Confusion Matrix**: Visualized to analyze true positives, false positives, true negatives, and false negatives.
- **ROC Curve**: Evaluated the model's ability to distinguish between classes.

### 5. Model Improvement
- **Regularization**: Added dropout layers to prevent overfitting.
- **Hyperparameter Tuning**: Adjusted the number of neurons and learning rate to improve generalization.

---

## Challenges and Solutions

### Challenge: Missing Values
- **Problem**: The dataset contained missing values in some features, including an irrelevant column (`Unnamed: 32`).
- **Solution**: Dropped the unnecessary column and imputed missing values using the `SimpleImputer` with the mean strategy.

### Challenge: Overfitting
- **Problem**: The model performed well on the training set but showed reduced generalization on the validation set.
- **Solution**:
  - Added dropout layers to reduce overfitting.
  - Monitored validation loss during training using early stopping.

---

## Results
- **Test Accuracy**: 0.9737
- **Precision and Recall**: High scores indicating the model's effectiveness in correctly identifying malignant and benign tumors.
- **Confusion Matrix**: Demonstrated minimal misclassification.
- **ROC Curve**: Showed an area under the curve (AUC) close to 1, indicating strong classification performance.

---

## Conclusion
This project demonstrates the potential of using Artificial Neural Networks for breast cancer classification. The ANN model effectively distinguishes between benign and malignant tumors, supporting its application in medical diagnostics. With further improvements, such models can play a crucial role in aiding radiologists and healthcare providers.

---

## Access the Dataset
You can download the dataset from the following link:
[Breast Cancer Wisconsin (Diagnostic) Data Set](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
