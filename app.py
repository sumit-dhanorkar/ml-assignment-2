"""
app.py
------
BITS WILP - M.Tech (AIML/DSE) - Machine Learning - Assignment 2
Streamlit demo app for 5 classification models trained on the
Breast Cancer Wisconsin (Diagnostic) dataset.

Run locally:
    streamlit run app.py

Deploy: push this repo to GitHub, then deploy on
https://streamlit.io/cloud pointing at this file.
"""

import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

st.set_page_config(page_title="Breast Cancer Classifier Demo", layout="wide")

# ---------------------------------------------------------------------
# Load artifacts (trained once by model/train_models.py)
# ---------------------------------------------------------------------
MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "kNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest (Ensemble)": "model/random_forest_ensemble.pkl",
}


@st.cache_resource
def load_artifacts():
    scaler = joblib.load("model/scaler.pkl")
    with open("model/meta.json") as f:
        meta = json.load(f)
    metrics_table = pd.read_csv("model/metrics.csv")
    models = {name: joblib.load(path) for name, path in MODEL_FILES.items()}
    return scaler, meta, metrics_table, models


scaler, meta, metrics_table, models = load_artifacts()
FEATURE_NAMES = meta["feature_names"]
TARGET_NAMES = meta["target_names"]  # ['malignant', 'benign']

# ---------------------------------------------------------------------
# Sidebar - controls
# ---------------------------------------------------------------------
st.sidebar.title("Controls")

st.sidebar.markdown("**1. Upload test data (CSV)**")
uploaded_file = st.sidebar.file_uploader(
    "Upload test_data.csv (or any CSV with the same 30 feature columns, "
    "optionally with a 'target' column)",
    type=["csv"],
)

st.sidebar.markdown("**2. Choose a model**")
selected_model_name = st.sidebar.selectbox("Model", list(MODEL_FILES.keys()))

# ---------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------
st.title("🔬 Breast Cancer Classification — Model Demo")
st.caption(
    "BITS WILP M.Tech (AIML/DSE) — Machine Learning — Assignment 2. "
    "Dataset: Breast Cancer Wisconsin (Diagnostic), 30 features, 569 instances, binary classification."
)

st.subheader("📊 Comparison of all 5 models (test split used during training)")
st.dataframe(metrics_table.set_index("ML Model Name"), use_container_width=True)

st.divider()

if uploaded_file is None:
    st.info(
        "⬅️ Upload the provided `test_data.csv` from the sidebar to see live "
        "predictions, metrics, and a confusion matrix for the selected model."
    )
else:
    df = pd.read_csv(uploaded_file)

    missing_cols = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing_cols:
        st.error(
            f"Uploaded file is missing {len(missing_cols)} required feature "
            f"column(s), e.g. {missing_cols[:5]}. Please upload a CSV with "
            f"the same 30 feature columns as test_data.csv."
        )
    else:
        has_target = "target" in df.columns

        X_input = df[FEATURE_NAMES]
        X_scaled = scaler.transform(X_input)

        model = models[selected_model_name]
        y_pred = model.predict(X_scaled)
        y_proba = model.predict_proba(X_scaled)[:, 1]

        pred_labels = [TARGET_NAMES[p] for p in y_pred]

        st.subheader(f"🔎 Predictions — {selected_model_name}")
        result_df = df.copy()
        result_df["predicted_class"] = pred_labels
        result_df["predicted_probability_benign"] = y_proba.round(4)
        st.dataframe(result_df, use_container_width=True, height=250)

        if has_target:
            y_true = df["target"]

            st.subheader("📈 Evaluation metrics on uploaded data")
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.4f}")
            col2.metric("AUC", f"{roc_auc_score(y_true, y_proba):.4f}")
            col3.metric("Precision", f"{precision_score(y_true, y_pred):.4f}")
            col4.metric("Recall", f"{recall_score(y_true, y_pred):.4f}")
            col5.metric("F1 Score", f"{f1_score(y_true, y_pred):.4f}")
            col6.metric("MCC", f"{matthews_corrcoef(y_true, y_pred):.4f}")

            left, right = st.columns(2)

            with left:
                st.subheader("🔢 Confusion Matrix")
                cm = confusion_matrix(y_true, y_pred)
                fig, ax = plt.subplots(figsize=(4, 3.5))
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    xticklabels=TARGET_NAMES,
                    yticklabels=TARGET_NAMES,
                    ax=ax,
                )
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)

            with right:
                st.subheader("📋 Classification Report")
                report = classification_report(
                    y_true, y_pred, target_names=TARGET_NAMES, output_dict=True
                )
                st.dataframe(pd.DataFrame(report).transpose().round(3))
        else:
            st.warning(
                "No 'target' column found in the uploaded file — showing "
                "predictions only. Upload test_data.csv (which includes the "
                "true labels) to see metrics, confusion matrix, and "
                "classification report."
            )

st.divider()
st.caption("Models trained in model/train_models.py — Logistic Regression, Decision Tree, kNN, Naive Bayes, Random Forest.")
