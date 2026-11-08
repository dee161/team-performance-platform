"""
Model Training and Artifact Generation Script
Dataset: Telco Customer Churn (Kaggle / IBM)
Models:
  1. Logistic Regression
  2. Decision Tree Classifier
  3. K-Nearest Neighbor (KNN) Classifier
  4. Naive Bayes Classifier (GaussianNB)
  5. Support Vector Machine (SVC)
Evaluation Metrics:
  - Accuracy
  - ROC-AUC Score
  - Precision
  - Recall
  - F1 Score
  - Matthews Correlation Coefficient (MCC)
"""

import os
import urllib.request
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix
)

# 1. Create directories
os.makedirs("model", exist_ok=True)

# 2. Download or load dataset
DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
DATA_PATH = "telco_customer_churn.csv"

if not os.path.exists(DATA_PATH):
    print(f"Downloading Telco Customer Churn dataset from {DATA_URL}...")
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    print("Download completed.")

df = pd.read_csv(DATA_PATH)
print(f"Original dataset shape: {df.shape}")

# 3. Data Cleaning
# TotalCharges has empty spaces (' ') for 11 customers with tenure=0
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(subset=['TotalCharges'], inplace=True)

# Drop CustomerID as it is a unique identifier with no predictive value
if 'customerID' in df.columns:
    df = df.drop(columns=['customerID'])

print(f"Cleaned dataset shape: {df.shape}")

# Features and target
target_col = 'Churn'
X = df.drop(columns=[target_col])
y = (df[target_col].str.strip() == 'Yes').astype(int)

# Identify numerical and categorical columns
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
cat_cols = [col for col in X.columns if col not in num_cols]

print(f"Total features: {len(X.columns)} (Numerical: {len(num_cols)}, Categorical: {len(cat_cols)})")

# 4. Train-Test Split (80% Train, 20% Test, stratified)
train_df, test_df = train_test_split(df, test_size=0.20, random_state=42, stratify=df[target_col])

# Save test_data.csv for assignment submission and Streamlit upload
test_df.to_csv("test_data.csv", index=False)
print(f"Saved test_data.csv with {len(test_df)} rows and {len(test_df.columns)} columns.")

X_train = train_df.drop(columns=[target_col])
y_train = (train_df[target_col].str.strip() == 'Yes').astype(int)

X_test = test_df.drop(columns=[target_col])
y_test = (test_df[target_col].str.strip() == 'Yes').astype(int)

# 5. Define Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_cols)
    ],
    remainder='drop'
)

# Fit and save preprocessor as a standalone artifact
preprocessor.fit(X_train)
joblib.dump(preprocessor, os.path.join("model", "preprocessor.joblib"))
print("Saved model/preprocessor.joblib")

# 6. Define 5 ML Models
models = {
    "Logistic Regression": {
        "clf": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        "file": "logistic_regression.joblib"
    },
    "Decision Tree": {
        "clf": DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, random_state=42),
        "file": "decision_tree.joblib"
    },
    "KNN": {
        "clf": KNeighborsClassifier(n_neighbors=15, weights='uniform'),
        "file": "knn.joblib"
    },
    "Naive Bayes": {
        "clf": GaussianNB(),
        "file": "naive_bayes.joblib"
    },
    "Support Vector Machine": {
        "clf": SVC(kernel='rbf', C=1.0, probability=True, cache_size=1000, random_state=42),
        "file": "svm.joblib"
    }
}

# 7. Train models, evaluate, and save pipelines
metrics_list = []

print("\n" + "="*80)
print("TRAINING AND EVALUATING ALL 5 CLASSIFICATION MODELS")
print("="*80)

for name, config in models.items():
    print(f"\nTraining {name}...")
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', config["clf"])
    ])
    pipeline.fit(X_train, y_train)
    
    # Save model pipeline
    save_path = os.path.join("model", config["file"])
    joblib.dump(pipeline, save_path)
    print(f"Saved trained pipeline -> {save_path}")
    
    # Inference on Test Set
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Calculate required metrics
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    metrics_list.append({
        "ML Model Name": name,
        "Accuracy": round(acc, 4),
        "AUC": round(auc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1": round(f1, 4),
        "MCC": round(mcc, 4)
    })
    
    print(f"--> {name}: Acc={acc:.4f}, AUC={auc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}, MCC={mcc:.4f}")

# 8. Comparison Table
metrics_df = pd.DataFrame(metrics_list)
print("\n" + "="*80)
print("FINAL EVALUATION METRICS COMPARISON TABLE")
print("="*80)
print(metrics_df.to_string(index=False))

# Save metrics to CSV for easy inspection / loading
metrics_df.to_csv(os.path.join("model", "evaluation_metrics.csv"), index=False)
print("\nSaved model/evaluation_metrics.csv successfully!")
