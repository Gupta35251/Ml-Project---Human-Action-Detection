# Ml-Project---Human-Action-Detection
# Human Activity Detection Using Machine Learning

This project aims to build a machine learning model that can automatically detect human physical activities (such as walking, sitting, standing, etc.) using sensor data collected from wearable devices.

## 📁 Dataset

- The dataset consists of time-series sensor data (e.g., accelerometer, gyroscope readings).
- Each row represents a windowed segment of sensor measurements.
- It includes:
  - Feature columns (sensor values)
  - `Activity`: the activity label
  - `subject`: the person who performed the activity

## 🧹 Preprocessing

- Removed outliers using **IQR (Interquartile Range)** method for each feature.
- Scaled data using `RobustScaler` to reduce the impact of remaining outliers.
- Converted activity labels into numerical form (if needed).
- Optional: used subject IDs to perform subject-wise splits to avoid data leakage.

## ⚙️ Model Training

Trained classification models to predict human activities using:

- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- XGBoost (optional)

### 🧪 Train-Test Split

- Used `train_test_split()` with stratified sampling.
- Alternatively, `GroupKFold` was used to split data based on `subject`.

## 📊 Evaluation Metrics

Used the following metrics to evaluate performance:

- Accuracy
- Precision (Macro Average)
- Recall (Macro Average)
- F1-Score (Macro Average)
- Confusion Matrix

Visualization:
- Confusion matrix plotted using `seaborn.heatmap` to show misclassification patterns.

## 📈 Results

Example output:
Accuracy Score : 92.3456%
Precision Score : 91.2245%
Recall Score : 90.8934%
F1 Score : 90.8543%
