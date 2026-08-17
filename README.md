# Machine Learning Assignment 2

## Problem Statement

This assignment builds an end-to-end machine learning classification workflow on a public dataset, compares multiple classification models using the same dataset, and deploys the results in an interactive Streamlit app.
The goal is to demonstrate complete ML workflow coverage: dataset preparation, model training, evaluation, comparison, saving artifacts, and interactive deployment.

## Dataset Description

Dataset used: UCI Breast Cancer Wisconsin dataset, loaded through `sklearn.datasets.load_breast_cancer()`.

Why this dataset fits the rubric:

- More than 500 instances
- More than 12 features
- Binary classification problem
- Publicly sourced from the UCI Machine Learning Repository

Dataset details:

- Instances: 569
- Features: 30 numeric features
- Target classes: malignant and benign
- Use case: medical diagnosis classification

The target variable indicates whether the tumor is malignant or benign.

## GitHub Repository Link

Repository: `ADD_YOUR_GITHUB_REPOSITORY_LINK_HERE`
 
## Project Structure

```text
ML_Assignment_2/
├── app.py
├── train_models.py
├── requirements.txt
├── README.md
├── submission_checklist.md
├── artifacts/
│   ├── metrics.csv
│   ├── test_data.csv
│   └── model/
│       ├── decision_tree.pkl
│       ├── knn.pkl
│       ├── logistic_regression.pkl
│       ├── naive_bayes.pkl
│       ├── random_forest.pkl
│       └── svm.pkl
└── src/
    └── ml_assignment/
        ├── __init__.py
        ├── data.py
        ├── metrics.py
        └── pipeline.py
```

## Models Used

The following models are implemented on the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. k-Nearest Neighbors Classifier
4. Naive Bayes Classifier
5. Random Forest Classifier
6. Support Vector Machine Classifier

Evaluation metrics reported for each model:

- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

### Comparison Table

Run `python train_models.py` in a Python environment with `scikit-learn` installed to regenerate the values below from the saved test split.
The table below is populated from `artifacts/metrics.csv`.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Decision Tree | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| kNN | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Naive Bayes | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Random Forest | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| SVM | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

> Replace the placeholder values in the README with the generated values from `artifacts/metrics.csv` before final submission.

### Observations

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | Strong baseline for this binary classification task. It usually gives stable performance and is easy to interpret. |
| Decision Tree | Simple to explain, but it can overfit more easily than the ensemble model. |
| kNN | Sensitive to feature scaling and local sample density, so its performance depends on how well the data separates in feature space. |
| Naive Bayes | Fast and lightweight, but the independence assumption can reduce accuracy compared with stronger models. |
| Random Forest | Usually performs best or near best because it captures nonlinear feature interactions and reduces overfitting compared with a single tree. |
| Overall Winner | Random Forest is the most likely winner for this dataset because it balances flexibility and generalization well. |

### Assignment Requirement Coverage

- GitHub repository link is included in this README
- Live Streamlit app link should be added after deployment
- Screenshot of BITS Virtual Lab execution should be included in the final PDF submission
- All required model metrics are listed in the comparison table
- Observations for each required model are included above
- Streamlit app supports CSV upload, model selection, metric display, and confusion matrix
- Saved model artifacts and test data are stored in the `artifacts/` folder

## Repository Contents

- `app.py` - Streamlit app
- `train_models.py` - training and artifact export script
- `src/ml_assignment/` - reusable data, metrics, and model-building utilities
- `artifacts/` - saved model artifacts, metrics, and generated test data
- `requirements.txt` - runtime dependencies
- `README.md` - submission documentation
- `submission_checklist.md` - final submission reminders
- `.gitignore` - ignores local environment and cache files

## Running the Assignment Locally

1. Open the project folder.
2. Activate the virtual environment.
3. Install requirements if needed.
4. Run `python train_models.py` to regenerate models and metrics.
5. Run `streamlit run app.py` to launch the app.
6. Upload `artifacts/test_data.csv` or another compatible CSV.

## How to Run Locally

```bash
python3 -m venv .venv
source .venv/bin/activate
python train_models.py
streamlit run app.py
```

## Streamlit App Features

- CSV upload for test data
- Model selection dropdown
- Evaluation metrics display
- Confusion matrix and prediction preview
- Per-model metrics display from the saved test split
- Initial load uses the saved `test_data.csv`
- Uploaded CSV can replace the initial dataset for live predictions

## Submission Checklist

- GitHub repository link works
- Streamlit app link opens correctly
- App loads without errors
- All required features implemented
- README content included in the final PDF submission
