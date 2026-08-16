from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

from src.ml_assignment.data import ensure_directory, load_dataset, split_dataset
from src.ml_assignment.metrics import evaluate_predictions
from src.ml_assignment.pipeline import build_model_specs, build_preprocess_pipeline


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
ARTIFACT_DIR = ROOT / "artifacts"


def train_and_save_all(output_dir: str | Path = ARTIFACT_DIR) -> pd.DataFrame:
    output_dir = ensure_directory(output_dir)
    model_dir = ensure_directory(output_dir / "model")

    bundle = load_dataset()
    X_train, X_test, y_train, y_test = split_dataset(bundle)

    preprocess = build_preprocess_pipeline()
    X_train_processed = preprocess.fit_transform(X_train)
    X_test_processed = preprocess.transform(X_test)

    test_data = X_test.copy()
    test_data["target"] = y_test.values
    test_data.to_csv(output_dir / "test_data.csv", index=False)

    results = []
    specs = build_model_specs()

    for spec in specs:
        model = spec.estimator
        model.fit(X_train_processed, y_train)
        y_pred = model.predict(X_test_processed)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_processed)[:, 1]
        else:
            y_prob = model.decision_function(X_test_processed)
            y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min() + 1e-12)

        metrics = evaluate_predictions(y_test, y_pred, y_prob)
        record = {
            "model": spec.name,
            **metrics,
        }
        results.append(record)
        with open(model_dir / f"{spec.name.lower().replace(' ', '_')}.pkl", "wb") as f:
            pickle.dump(
                {
                    "name": spec.name,
                    "preprocess": preprocess,
                    "model": model,
                    "feature_names": bundle.feature_names,
                },
                f,
            )

    results_df = pd.DataFrame(results).sort_values("model").reset_index(drop=True)
    results_df.to_csv(output_dir / "metrics.csv", index=False)
    return results_df


def main():
    results = train_and_save_all()
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
