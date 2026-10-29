import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

# 1. Generate Synthetic Credit Data
np.random.seed(42)
n_samples = 1000

data = pd.DataFrame({
    'income': np.random.normal(50000, 15000, n_samples).clip(15000, 120000),
    'age': np.random.randint(20, 65, n_samples),
    'debts': np.random.normal(15000, 8000, n_samples).clip(0, 50000),
    'past_defaults': np.random.poisson(0.3, n_samples),
    'credit_card_utilization': np.random.uniform(0.0, 1.0, n_samples)
})

# Target: 1 for High Risk (Default), 0 for Low Risk (Good Credit)
risk_score = (
    (data['debts'] / data['income'] > 0.4).astype(int) +
    (data['past_defaults'] > 0).astype(int) +
    (data['credit_card_utilization'] > 0.7).astype(int)
)
data['credit_risk'] = (risk_score >= 2).astype(int)

# 2. Preprocessing & Feature Engineering
data['debt_to_income'] = data['debts'] / data['income']
X = data.drop(columns=['credit_risk'])
y = data['credit_risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Model Training - Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# 4. Evaluation
y_pred = rf_model.predict(X_test_scaled)
y_proba = rf_model.predict_proba(X_test_scaled)[:, 1]

print("=== Task 1: Credit Scoring Model ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
