# Breast Cancer Classification — ML Assignment 2

**BITS WILP — M.Tech (AIML/DSE) — Machine Learning**

---

## a. Problem Statement

Breast cancer diagnosis relies on correctly distinguishing malignant tumors
from benign ones based on measurements taken from a digitized image of a
fine needle aspirate (FNA) of a breast mass. The goal of this project is to
build and compare multiple classification models that predict whether a
tumor is **malignant** or **benign** from a set of 30 numeric cell-nuclei
features, and to expose the best-performing models through an interactive
Streamlit application for evaluation.

## b. Dataset Description

- **Name:** Breast Cancer Wisconsin (Diagnostic) Data Set
- **Source:** UCI Machine Learning Repository (also distributed via
  `sklearn.datasets.load_breast_cancer`)
- **Instances:** 569
- **Features:** 30 numeric features (mean, standard error, and "worst"
  value of 10 real-valued measurements computed for each cell nucleus:
  radius, texture, perimeter, area, smoothness, compactness, concavity,
  concave points, symmetry, fractal dimension)
- **Target:** Binary — `0 = malignant`, `1 = benign`
- **Class balance:** 212 malignant / 357 benign (no missing values)

## c. GitHub Repository Link
    `https://github.com/sumit-dhanorkar/ml-assignment-2`

## d. Models Used

All 5 models were trained on an 80/20 stratified train/test split
(`random_state=42`) of the same dataset, using standardized features.

### Comparison Table

| ML Model Name             | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|----------------------------|----------|--------|-----------|--------|--------|--------|
| Logistic Regression        | 0.9825   | 0.9954 | 0.9861    | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree               | 0.9123   | 0.9157 | 0.9559    | 0.9028 | 0.9286 | 0.8174 |
| kNN                         | 0.9561   | 0.9788 | 0.9589    | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes                 | 0.9298   | 0.9868 | 0.9444    | 0.9444 | 0.9444 | 0.8492 |
| Random Forest (Ensemble)    | 0.9561   | 0.9932 | 0.9589    | 0.9722 | 0.9655 | 0.9054 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best overall performer on this dataset — the 30 features are largely linearly separable after standardization, which suits a linear decision boundary well. Highest accuracy, F1, and MCC of all 5 models, with very few misclassifications. |
| Decision Tree | Weakest performer. A single unpruned tree overfits the training split and is sensitive to small variations in feature thresholds, which shows up as the lowest AUC and MCC — the ensemble version (Random Forest) fixes this same weakness. |
| kNN | Solid, consistent performance once features are scaled (essential for a distance-based model). Ties Random Forest on accuracy, recall, F1, and MCC, but its AUC is noticeably lower since class-probability estimates from neighbor voting are coarser than a probabilistic/ensemble model. |
| Naive Bayes | Middling accuracy despite a very strong AUC (0.9868) — the model ranks predictions well, but its accuracy suffers because the Gaussian/independence assumption is violated (many of the 30 features are highly correlated, e.g. radius/perimeter/area), which hurts the calibration of its 0.5-threshold decisions. |
| Random Forest (Ensemble) | Very strong and stable — bagging many decision trees corrects the overfitting problem of the single Decision Tree and gives the highest AUC (0.9932) among the tree-based models, essentially matching kNN on the other metrics. |
| **Overall Winner for your dataset?** | **Logistic Regression** — it leads on Accuracy, Precision, Recall, F1, and MCC, and is a close second on AUC (0.9954 vs Random Forest's 0.9932), while also being the simplest, cheapest, and most interpretable model in the comparison. |

---

## Project Structure

```
ml-assignment-2/
├── app.py                 # Streamlit app
├── requirements.txt
├── README.md
├── test_data.csv          # held-out test split (features + true label)
└── model/
    ├── train_models.py    # trains all 5 models, computes metrics, saves everything
    ├── metrics.csv         # comparison table (generated)
    ├── meta.json            # feature/target names (generated)
    ├── scaler.pkl            # fitted StandardScaler (generated)
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    └── random_forest_ensemble.pkl
```

## How to Run Locally

```bash
pip install -r requirements.txt
python model/train_models.py   # trains models, writes *.pkl and test_data.csv
streamlit run app.py           # opens the app at http://localhost:8501
```

Upload `test_data.csv` in the sidebar, pick a model from the dropdown, and
the app will show predictions, live metrics, a confusion matrix, and a
classification report.

## Live App

`https://2025ac05259-ml-assignment.streamlit.app/`
