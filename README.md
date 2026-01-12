# 👩🏻‍💻🎧 Customer Subscription Churn Prediction

## 📌 Project Overview

This repository presents an end-to-end machine learning pipeline for **Customer Subscription Churn Prediction**, with a strong emphasis on **data preparation, feature engineering, deployment-oriented evaluation, and Dashboard Visualization**.

The project explores **Binary Classification Techniques** applied to Customer subscription metadata and dataset.

This repository prioritizes:
- Robust ETL and preprocessing design
- Behavioral feature engineering
- Scalable and reusable pipelines
- Validation prediction performance
- Realistic deployment-style evaluation under extreme class imbalance

The pipeline covers the full workflow from **dataset preparation and exploratory analysis**, through **model development and validation**, **deployment and inference simulation**, to **data visualization and dashboard**

---

## 📊 Dataset

The dataset used in this project is:

**Synthetic Customer Churn Datasets**  
Source: Kaggle  
https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset

Please download the dataset manually and place it in the following directory:
Datasets/
└── customer_churn_dataset-training-master.csv
└── customer_churn_dataset-testing-master.csv

---

## ⚙️ System Requirements

Install all required dependencies using:

```bash
pip install -r "Artifacts/requirements.txt"
```

## 🧱 Project Structure and Workflow

The workflow is divided into four main stages:

### 1️⃣ Dataset Preparation & Exploratory Analysis

This stage focuses on building a clean, leakage-aware dataset suitable for customer churn detection.

**Key steps**
- Loading and inspecting raw customer subscription data 
- Feature analysis and column selection to reduce redundancy and prevent leakage

**Feature engineering**
- Label encoding
- Manual encoding
- Data type conversion (i.e., Float64 to int, vice versa)

**Exploratory analysis**
- Pearson correlation matrix
- Feature boxplots
- Random Forest Gini feature importance
- Important feature selection

**Notebook:**
- Dataset_Preparation.ipynb

**Outputs:**
1. Preprocessed datasets:
- Datasets/preprocessed_train.csv
- Datasets/preprocessed_test.csv

2. Artifacts:
- Artifacts/label_encoders.pkl
- Artifacts/manual_mapping.pkl
- important_features.pkl

### 2️⃣ Model Development & Performance Validation

Multiple **binary classification** models are developed and evaluated using the validation dataset to understand their behavior under the adopted sampling strategy.

**Models evaluated:**
- Gaussian Naive-Baiyes (package: scikit-learn)
- AdaBoost (package: scikit-learn)
- CatBoost (package: catboost)
- Decision Tree (package: scikit-learn)

Each non-neural model undergoes manual loop-based hyperparameter tuning, with F1-score used as the primary optimization metric.

**Notebook:**
- Model_Development.ipynb

**Outputs:**
1. Trained models:
- Saved Models/

2. Artifact:
- Artifacts/standard_scaler.pkl


### 3️⃣ Model Deployment & Inference
This stage simulates a deployment scenario by running trained models on the test dataset, which represents unseen future customer subscriptions.

**Notebook:**
- Model_Deployment.ipynb

**Outputs:**
- Datasets/test_pred_results_{model}.csv


### 3️⃣ Data Visualization and Dashboard
A Power BI interactive dashboard that captures important customer details based on deployment prediction results.

**BI file:**
- BI_Dashboard.pbix



## Model Performance Evaluation Analysis

### 🎯 Definition of Positive and Negative Classes

In binary classification problems, the positive class is defined by the event of interest, not by class frequency.

**In the context of churn detection:**
- Positive class (label = 1): Churned customers
- Negative class (label = 0): Normal (non-churned) customers

Although churn is least expected, it is treated as positive because detecting churned customers is the primary objective of the system.

### 📐 Evaluation Metrics

Given the extreme class imbalance, accuracy alone is insufficient. The following metrics are emphasized:

**Precision:**
- Measures the proportion of correctly predicted churned customers among all subscriptions.
- Precision = TP / (TP + FP)
- A high precision indicates fewer false churn alerts.

**Recall:**
- Measures the proportion of actual churned customers correctly detected.
- Recall = TP / (TP + FN)
- A high recall minimizes undetected churn cases.

**F1-Score:**
- Balances precision and recall into a single metric.
- F1-score = 2 × (Precision × Recall) / (Precision + Recall)

### Model validation prediction results

The following results are obtained on the **development validation set**, which simulates unseen future transactions containing a mixture of normal and fraudulent behavior. Due to the adopted sampling strategy, evaluation focuses on **precision, recall, and F1-score** rather than accuracy alone.

---

#### 🔹 Gaussian Naive-Baiyes

- **Accuracy:** 0.90  
- **Recall:** 0.88
- **Precision:** 0.94  
- **F1-Score:** 0.91  

**Interpretation:**  
Gaussian Naive-Bayes provides a solid baseline performance with a good balance between recall and precision. 
The model is able to identify most churn cases while maintaining relatively low false positives. 
However, its simplifying assumption of feature independence limits its ability to capture complex patterns in the data, resulting in lower performance compared to ensemble-based models.
---

#### 🔹 AdaBoost

- **Accuracy:** 0.95 
- **Recall:** 0.92  
- **Precision:** 0.99  
- **F1-Score:** 0.96 

**Interpretation:**  
AdaBoost significantly improves performance by combining multiple weak learners, resulting in high precision and strong recall. 
The model is particularly effective at minimizing false positives while still capturing the majority of churn cases. 
This makes AdaBoost a strong candidate when the cost of incorrectly flagging non-churn customers is high.
---

#### 🔹 CatBoost 

- **Accuracy:** 0.97  
- **Recall:** 0.95  
- **Precision:** 1.0 
- **F1-Score:** 0.97

**Interpretation:**  
CatBoost achieves excellent performance across all evaluation metrics. 
The perfect precision indicates that all customers predicted as churn are indeed churners, while the high recall shows strong coverage of actual churn cases. 
Its ability to handle complex, non-linear relationships makes CatBoost particularly suitable for real-world churn prediction scenarios.
---

#### 🔹 Decision Tree

- **Accuracy:** 0.97  
- **Recall:** 0.95  
- **Precision:** 1.0  
- **F1-Score:** 0.97  

**Interpretation:**  
The Decision Tree model matches CatBoost’s performance on this validation set, delivering perfect precision and high recall. 
While decision trees offer strong interpretability, their performance may be sensitive to data variations and overfitting. 
Proper regularization and validation are required to ensure robustness in production.
---

**Overall Observation:**  
Ensemble-based and tree-based models consistently outperform the probabilistic baseline. Gaussian Naive-Bayes serves as a strong reference model, while AdaBoost, CatBoost, and Decision Tree demonstrate superior capability in capturing complex feature interactions. 
Among them, CatBoost and Decision Tree achieve the best overall balance between precision and recall, making them the top-performing models on the validation set.

## 🧠 Summary
This project demonstrates the effectiveness of machine learning models in predicting customer churn from transactional and behavioral features. By prioritizing precision, recall, and F1-score, the evaluation aligns with real-world business objectives where misclassification costs are asymmetric.

Tree-based and ensemble models show strong predictive performance and provide actionable insights for customer retention strategies. The final predictions can be directly integrated into business intelligence tools (e.g., Power BI) to support targeted interventions and data-driven decision-making.
