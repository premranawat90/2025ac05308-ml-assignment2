# Machine Learning Assignment 2

## Problem Statement

This assignment builds an end-to-end machine learning classification workflow on a public dataset, compares multiple classification models using the same dataset, and deploys the results in an interactive Streamlit app.

## Dataset Description

Dataset used: UCI Breast Cancer Wisconsin dataset, loaded through `sklearn.datasets.load_breast_cancer()`.

Why this dataset fits the rubric:

- More than 500 instances
- More than 12 features
- Binary classification problem
- Publicly sourced from the UCI Machine Learning Repository

The target variable indicates whether the tumor is malignant or benign.

## GitHub Repository Link

Repository: `ADD_YOUR_GITHUB_REPOSITORY_LINK_HERE`

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

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Decision Tree | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| kNN | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Naive Bayes | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Random Forest | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| SVM | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

> The table above is populated automatically by `artifacts/metrics.csv` after training. Replace the placeholder values in the README with the generated metrics before final submission.

### Observations

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | Strong baseline for this binary problem and usually stable because the dataset is well behaved. |
| Decision Tree | Easy to interpret, but it can overfit more than ensemble methods. |
| kNN | Works well after scaling, but performance depends heavily on local neighborhood structure. |
| Naive Bayes | Fast and simple, but the independence assumption can limit accuracy. |
| Random Forest | Usually the strongest classical model here because it handles nonlinear interactions well. |
| SVM | Often competitive on this dataset after scaling and may match or exceed logistic regression. |
| Overall Winner | Random Forest is the most likely winner for this dataset. |

## Repository Contents

- `app.py` - Streamlit app
- `train_models.py` - training and artifact export script
- `src/ml_assignment/` - reusable data, metrics, and model-building utilities
- `artifacts/` - saved model artifacts, metrics, and generated test data
- `requirements.txt` - runtime dependencies
- `README.md` - submission documentation
- `submission_checklist.md` - final submission reminders

## How to Run Locally

```bash
python train_models.py
streamlit run app.py
```

## Streamlit App Features

- CSV upload for test data
- Model selection dropdown
- Evaluation metrics display
- Confusion matrix and prediction preview
- Per-model metrics display from the saved test split

## Submission Checklist

- GitHub repository link works
- Streamlit app link opens correctly
- App loads without errors
- All required features implemented
- README content included in the final PDF submission
