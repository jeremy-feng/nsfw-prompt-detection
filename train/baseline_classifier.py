"""Train and run the baseline classifier."""

import os

import fire
from pytorch_lightning import seed_everything

from data import DatasetLoader
from model.baseline_classifier import BaselineClassifier


def main(
    seed: int = 42,
    pred_path: str = "artifacts/pred/pred_baseline.pkl",
    label_path: str = "artifacts/label/label.pkl",
) -> None:
    """Train the baseline model and persist predictions/labels.

    Args:
        seed: Random seed for reproducibility.
        pred_path: Destination path for predicted probabilities pickle.
        label_path: Destination path for label pickle aligned with predictions.
    """
    seed_everything(seed)

    loader = DatasetLoader()
    classifier = BaselineClassifier()

    classifier.fit()
    pred = classifier.predict(loader)

    os.makedirs(os.path.dirname(pred_path), exist_ok=True)
    pred.to_pickle(pred_path)

    os.makedirs(os.path.dirname(label_path), exist_ok=True)
    test_df = loader._splits["test"]
    label = test_df["nsfw"].astype(int)
    label.to_pickle(label_path)

    print(f"Saved prediction to: {pred_path}")
    print(f"Saved label to: {label_path}")


if __name__ == "__main__":
    fire.Fire(main)
