# Breast Cancer Classification — ML Assignment 2

**BITS WILP — M.Tech (AIML/DSE) — Machine Learning**

---

## a. Problem Statement

Breast cancer diagnosis relies on correctly distinguishing malignant tumors from benign ones based on measurements taken from a digitized image of a fine needle aspirate (FNA) of a breast mass. The goal of this project is to build and compare multiple classification models that predict whether a tumor is **malignant** or **benign** from a set of 30 numeric cell-nuclei features, and to expose the best-performing models through an interactive Streamlit application for evaluation.

## b. Dataset Description

* **Name:** Breast Cancer Wisconsin (Diagnostic) Data Set
* **Source:** UCI Machine Learning Repository (also distributed via `sklearn.datasets.load_breast_cancer`)
* **Instances:** 569
* **Features:** 30 numeric features (mean, standard error, and "worst" value of 10 real-valued measurements computed for each cell nucleus: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension)
* **Target:** Binary — `0 = malignant`, `1 = benign`
* **Class Balance:** 212 malignant / 357 benign (no missing values)

## c. GitHub Repository Link

[View GitHub Repository](https://github.com/sumit-dhanorkar/ml-assignment-2)

## d. Models Used

All 5 models were trained on an 80/20 stratified train/test split (`random_state=42`) of the same dataset, using standardized features.

The following classification models were implemented:

1. Logistic Regression
2. Decision Tree
3. k-Nearest Neighbors (kNN)
4. Naive Bayes
5. Random Forest (Ensemble)

### Comparison Table

| ML Model Name            | Accuracy |    AUC | Precision | Recall |     F1 |    MCC |
| ------------------------ | -------: | -----: | --------: | -----: | -----: | -----: |
| Logistic Regression      |   0.9825 | 0.9954 |    0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree            |   0.9123 | 0.9157 |    0.9559 | 0.9028 | 0.9286 | 0.8174 |
| kNN                      |   0.9561 | 0.9788 |    0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes              |   0.9298 | 0.9868 |    0.9444 | 0.9444 | 0.9444 | 0.8492 |
| Random Forest (Ensemble) |   0.9561 | 0.9932 |    0.9589 | 0.9722 | 0.9655 | 0.9054 |

### Observations

| ML Model Name                     | Observation about Model Performance                                                                                                                                                                                                                                                                          |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Logistic Regression               | Best overall performer on this dataset — the 30 features are largely linearly separable after standardization, which suits a linear decision boundary well. It achieves the highest accuracy, F1, and MCC of all 5 models, with very few misclassifications.                                                 |
| Decision Tree                     | Weakest performer. A single unpruned tree overfits the training split and is sensitive to small variations in feature thresholds, which shows up as the lowest AUC and MCC. The ensemble version (Random Forest) helps address this weakness.                                                                |
| kNN                               | Solid and consistent performance once features are scaled, which is essential for a distance-based model. It ties Random Forest on accuracy, recall, F1, and MCC, but its AUC is lower since class-probability estimates from neighbor voting are coarser than those from a probabilistic or ensemble model. |
| Naive Bayes                       | Achieves moderate accuracy despite a very strong AUC (0.9868). The model ranks predictions well, but its accuracy is affected because the Gaussian independence assumption is not fully satisfied, as many of the 30 features are highly correlated (e.g., radius, perimeter, and area).                     |
| Random Forest (Ensemble)          | Very strong and stable. Bagging many decision trees helps reduce the overfitting problem of a single Decision Tree and gives a high AUC (0.9932), while essentially matching kNN on the other metrics.                                                                                                       |
| **Overall Best Performing Model** | **Logistic Regression** — it leads on Accuracy, Precision, Recall, F1, and MCC, while also achieving an AUC of 0.9954. It is the simplest, computationally efficient, and most interpretable model among the top-performing models in this comparison.                                                       |

---

## Project Structure

```text
ml-assignment-2/
├── app.py                           # Streamlit application
├── requirements.txt                 # Python dependencies
├── README.md                        # Project documentation
├── test_data.csv                    # Held-out test split (features + true label)
└── model/
    ├── train_models.py              # Trains all 5 models, computes metrics, and saves artifacts
    ├── metrics.csv                  # Comparison table (generated)
    ├── meta.json                    # Feature and target names (generated)
    ├── scaler.pkl                   # Fitted StandardScaler (generated)
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    └── random_forest_ensemble.pkl
```

## How to Run Locally

Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

Train the models and generate the required model artifacts and test data:

```bash
python model/train_models.py
```

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open locally at `http://localhost:8501`.

Upload `test_data.csv` using the sidebar and select a model from the dropdown. The application will display predictions, evaluation metrics, a confusion matrix, and a classification report for the selected model.

## Live Streamlit App

[Open Streamlit Application](https://2025ac05259-ml-assignment.streamlit.app/)
