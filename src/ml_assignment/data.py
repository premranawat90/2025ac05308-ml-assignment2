from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class DatasetBundle:
    features: pd.DataFrame
    target: pd.Series
    feature_names: list[str]
    target_name: str


def load_dataset() -> DatasetBundle:
    """Load the UCI Breast Cancer Wisconsin dataset from scikit-learn."""
    data = load_breast_cancer(as_frame=True)
    features = data.data.copy()
    target = data.target.copy()
    return DatasetBundle(
        features=features,
        target=target,
        feature_names=list(features.columns),
        target_name=str(data.target.name),
    )


def split_dataset(
    bundle: DatasetBundle,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Create a stratified train/test split for the assignment."""
    return train_test_split(
        bundle.features,
        bundle.target,
        test_size=test_size,
        random_state=random_state,
        stratify=bundle.target,
    )


def ensure_directory(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

