from __future__ import annotations

from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: object
    needs_scaling: bool = False


def _scaled(estimator):
    return Pipeline([("scaler", StandardScaler()), ("model", estimator)])


def build_model_specs(random_state: int = 42) -> list[ModelSpec]:
    return [
        ModelSpec(
            "Logistic Regression",
            _scaled(LogisticRegression(max_iter=2000, random_state=random_state)),
            True,
        ),
        ModelSpec(
            "Decision Tree",
            DecisionTreeClassifier(random_state=random_state),
        ),
        ModelSpec(
            "kNN",
            _scaled(KNeighborsClassifier(n_neighbors=7)),
            True,
        ),
        ModelSpec(
            "Naive Bayes",
            GaussianNB(),
        ),
        ModelSpec(
            "Random Forest",
            RandomForestClassifier(
                n_estimators=300,
                random_state=random_state,
                class_weight="balanced",
            ),
        ),
        ModelSpec(
            "SVM",
            _scaled(SVC(kernel="rbf", probability=True, random_state=random_state)),
            True,
        ),
    ]


def build_preprocess_pipeline():
    """Keep feature handling simple and explicit for the app and saved models."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
