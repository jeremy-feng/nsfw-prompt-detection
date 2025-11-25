"""Train the NSFW classifier with configurable hyperparameters."""

import os

import fire
from pytorch_lightning import seed_everything

from data import DatasetLoader
from model.nsfw_classifier import NSFWClassifier, TrainingConfig


def main(
    seed: int = 42,
    pred_path: str = "artifacts/pred/pred_nsfw_classifier.pkl",
    model_path: str = "artifacts/model/JeremyFeng/nsfw-prompt-detection",
    **config_kwargs,
) -> None:
    """Train NSFW classifier and save predictions/model artifacts.

    Args:
        seed: Random seed for reproducibility.
        pred_path: Destination path for prediction pickle file.
        model_path: Directory to save model and tokenizer artifacts.
        **config_kwargs: Extra hyperparameters forwarded into ``TrainingConfig``.
    """
    seed_everything(seed)
    config = TrainingConfig(**config_kwargs)
    loader = DatasetLoader()

    classifier = NSFWClassifier(config)
    classifier.fit(loader)
    preds = classifier.predict(loader)

    os.makedirs(os.path.dirname(pred_path), exist_ok=True)
    preds.to_pickle(pred_path)

    os.makedirs(model_path, exist_ok=True)
    classifier.model.save_pretrained(model_path)
    classifier.tokenizer.save_pretrained(model_path)

    print(f"Predictions saved to: {pred_path}")
    print(f"Model artifacts saved to: {model_path}")


if __name__ == "__main__":
    fire.Fire(main)
