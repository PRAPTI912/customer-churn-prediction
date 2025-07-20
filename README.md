# 📉 Customer Churn Prediction Using Machine Learning

Welcome to the **Customer Churn Prediction** project! This end-to-end machine learning pipeline identifies customers likely to stop using a service, helping businesses take proactive action.

## 🧠 Project Overview

Customer churn is a critical metric for any subscription-based or service-driven business. This project applies supervised ML techniques to **predict whether a customer will churn or stay**, based on various features like tenure, service usage, and payment method.

## 📁 Dataset

- Dataset: `customer_churn.csv`
- Source: [Kaggle - Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)
- Size: ~7K rows, 21 columns


## 🔧 Technologies & Libraries Used

- Python 🐍
- Scikit-learn
- imbalanced-learn (SMOTE)
- Pandas, NumPy
- Matplotlib, Seaborn


## 🔬 ML Pipeline Stages

1. 📊 Data Analysis & Visualization
   - Basic EDA using `.info()`, `.describe()`, and countplots
2. 🧼 Data Cleaning
   - Handle missing values, drop duplicates
3. 🛠 Feature Engineering
   - One-Hot Encoding for categoricals
   - Scaling for numericals using `StandardScaler`
4. 🔀 Train-Test Split
   - `train_test_split` with stratification
5. ⚖️ Imbalance Handling
   - Applied **SMOTE** to balance class distribution
6. 🧪 Model Training
   - `RandomForestClassifier` inside a pipeline
7. 📈 Evaluation
   - Accuracy, Confusion Matrix, Classification Report
