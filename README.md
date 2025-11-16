# MLOPS_1
A simple MLOPS project using DVC,Dagshub,Git,MLflow

📌 Project Overview

This project implements an end-to-end Machine Learning pipeline to predict diabetes using the PIMA Indian Diabetes Dataset.
It covers data ingestion, preprocessing, model training, evaluation, experiment tracking, and reproducibility using modern MLOps tools.

🚀 Key Features

End-to-end ML pipeline using Python & Scikit-learn

EDA + Data Cleaning using Pandas

Multiple ML models (Logistic Regression, Random Forest, SVM, etc.)

MLflow for experiment tracking

DVC for data/model versioning

Clean, modular, reproducible workflow using Git + DVC

🧠 Tech Stack

Python

Pandas

Scikit-learn

MLflow

📝 Dataset

PIMA Indian Diabetes Dataset

Medical diagnostic dataset of women 21+

Target variable: Outcome (0 = Non-diabetic, 1 = Diabetic)

Key features:

Pregnancies

Glucose

Blood Pressure

Skin Thickness

Insulin

BMI

Diabetes Pedigree

Age

🔍 Exploratory Data Analysis

Performed EDA using Pandas + Matplotlib/Seaborn:

Handling missing/zero values

Distribution plots

Feature correlations

Relationship analysis with target variable

🔧 ML Pipeline Components
1. Data Ingestion

Loads raw CSV

Saves it with DVC tracking

2. Data Preprocessing

Replaces zeros with NaNs where needed

Scaling using StandardScaler

Train-test split

Stores processed data via DVC

3. Model Training

Models trained and compared:

Random Forest


MLflow logs:

Parameters

Metrics

Confusion matrix

Model artifacts

4. Model Evaluation

Metrics used:

Accuracy

Precision

Recall

F1-score

ROC AUC

💡 Future Improvements

Model deployment with FastAPI or Streamlit

Hyperparameter tuning using Optuna

Docker containerization

CI/CD pipeline integration

Feature engineering automation

DVC

Git/GitHub
