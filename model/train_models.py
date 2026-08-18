"""
train_models.py
----------------
BITS WILP - M.Tech (AIML/DSE) - Machine Learning - Assignment 2

Trains 5 classification models on the Breast Cancer Wisconsin (Diagnostic)
dataset, evaluates them with 6 metrics, and saves:
  - each trained model as a .pkl file (in model/)
  - the fitted StandardScaler (needed by the Streamlit app)
  - a metrics comparison table (model/metrics.csv)
  - the held-out test split as test_data.csv (used to demo the Streamlit app)

Run this once before running the Streamlit app:
    python model/train_models.py
"""

import json
import numpy as np
import pandas as pd
import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

RANDOM_STATE = 42

# ---------------------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------------------
data = load_breast_cancer(as_frame=True)
X = data.data
y = data.target  # 0 = malignant, 1 = benign
feature_names = list(X.columns)

print(f"Dataset shape: {X.shape[0]} instances, {X.shape[1]} features")
print(f"Class balance: {dict(y.value_counts())}")

# ---------------------------------------------------------------------
# 2. Train/test split (stratified, 80/20)
# ---------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# ---------------------------------------------------------------------
# 3. Scale features (fit on train only, to avoid leakage)
#    KNN and Logistic Regression benefit from scaling; tree-based
#    models are unaffected by it, so using one scaled pipeline for all
#    models keeps things simple and consistent for the Streamlit app.
# ---------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------------
# 4. Define the 5 required models
# ---------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "kNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Random Forest (Ensemble)": RandomForestClassifier(
        n_estimators=200, random_state=RANDOM_STATE
    ),
}

# ---------------------------------------------------------------------
# 5. Train, evaluate, save
# ---------------------------------------------------------------------
results = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        "ML Model Name": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "AUC": round(roc_auc_score(y_test, y_proba), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1": round(f1_score(y_test, y_pred), 4),
        "MCC": round(matthews_corrcoef(y_test, y_pred), 4),
    }
    results.append(metrics)
    print(metrics)

    # Save model with a filesystem-safe name
    safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    joblib.dump(model, f"model/{safe_name}.pkl")

# Save the scaler too - the app needs it to transform uploaded test data
joblib.dump(scaler, "model/scaler.pkl")

# Save feature names + target names for the app
with open("model/meta.json", "w") as f:
    json.dump(
        {
            "feature_names": feature_names,
            "target_names": list(data.target_names),  # ['malignant', 'benign']
        },
        f,
        indent=2,
    )

# ---------------------------------------------------------------------
# 6. Save comparison table
# ---------------------------------------------------------------------
results_df = pd.DataFrame(results)
results_df.to_csv("model/metrics.csv", index=False)
print("\nComparison table:\n", results_df.to_string(index=False))

# ---------------------------------------------------------------------
# 7. Save test_data.csv (this is the file you upload into the Streamlit
#    app, and the one required in the GitHub repo / submission)
# ---------------------------------------------------------------------
test_df = X_test.copy()
test_df["target"] = y_test.values
test_df.to_csv("test_data.csv", index=False)
print(f"\ntest_data.csv saved with {len(test_df)} rows.")
