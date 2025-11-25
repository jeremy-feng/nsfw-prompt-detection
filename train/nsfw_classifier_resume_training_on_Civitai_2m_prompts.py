"""Resume NSFW classifier training using a mix of datasets."""

import os
import re

import fire
import pandas as pd
from datasets import load_dataset
from pytorch_lightning import seed_everything
from sklearn.model_selection import train_test_split

from model.nsfw_classifier import NSFWClassifier, TrainingConfig

DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42


class MixDatasetLoader:
    """Load train/valid from Civitai-2m and test from civitai-337k."""

    def __init__(
        self,
        train_dataset_name: str = "AdamCodd/Civitai-2m-prompts",
        test_dataset_name: str = "thefcraft/civitai-stable-diffusion-337k",
        train_sample_size: int = 200_000,
        valid_sample_size: int = 20_000,
        test_size: float = DEFAULT_TEST_SIZE,
        random_state: int = DEFAULT_RANDOM_STATE,
    ):
        """Configure mixed dataset sources and sampling sizes.

        Args:
            train_dataset_name: Hugging Face dataset id to stream for train/valid.
            test_dataset_name: Hugging Face dataset id to build the test split.
            train_sample_size: Number of rows to reserve for training.
            valid_sample_size: Number of rows to reserve for validation.
            test_size: Fraction of test rows to sample from the test dataset.
            random_state: Seed for shuffling and stratified splits.
        """
        self.train_dataset_name = train_dataset_name
        self.test_dataset_name = test_dataset_name
        self.train_sample_size = train_sample_size
        self.valid_sample_size = valid_sample_size
        self.test_size = test_size
        self.random_state = random_state
        self._splits: dict[str, pd.DataFrame] | None = None

    def preprocess(
        self, text: str | list[str], is_first: bool = True
    ) -> str | list[str]:
        """Normalize prompt text using the project-specific rules.

        Args:
            text: Prompt to normalize. Can be a string or nested list of strings.
            is_first: Internal flag to control recursion for nested segments.

        Returns:
            Cleaned text matching the structure of the input.
        """
        if is_first:
            if isinstance(text, str):
                pass
            elif isinstance(text, list):
                return [self.preprocess(item) for item in text]

        # Strip HTML tags
        text = re.sub(r"<.*?>", "", text)
        # Normalize parentheses
        text = re.sub(r"\(+", "(", text)
        text = re.sub(r"\)+", ")", text)

        # Handle weighted tokens such as (word:1.2)
        matches = re.findall(r"\(.*?\)", text)
        for match in matches:
            text = text.replace(match, self.preprocess(match[1:-1], is_first=False))

        # Normalize separators
        text = text.replace("\n", ",").replace("|", ",")

        if is_first:
            parts = [segment.strip() for segment in text.split(",")]
            parts = [segment for segment in parts if segment]
            return ", ".join(parts)

        return text

    def _load_samples(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Stream, shuffle, and take a fixed number of train/valid rows.

        Returns:
            Tuple of (train_df, valid_df) dataframes sized per configuration.

        Raises:
            ValueError: If the dataset does not yield enough rows.
        """
        stream = load_dataset(
            self.train_dataset_name, split="train", streaming=True
        ).shuffle(buffer_size=10_000, seed=self.random_state)

        needed = self.train_sample_size + self.valid_sample_size
        rows: list[dict] = []
        for i, row in enumerate(stream):
            rows.append(row)
            if i + 1 >= needed:
                break

        if len(rows) < needed:
            raise ValueError(
                f"Requested {needed} samples but dataset only yielded {len(rows)}."
            )

        df = pd.DataFrame(rows)
        df = df[["prompt", "negativePrompt", "nsfw"]]
        df = df.dropna(subset=["prompt", "nsfw"]).copy()
        df["negativePrompt"] = df["negativePrompt"].fillna("")
        df["prompt"] = df["prompt"].apply(self.preprocess)
        df["negativePrompt"] = df["negativePrompt"].apply(self.preprocess)
        df = df.reset_index(drop=True)

        train_df = df.iloc[: self.train_sample_size].reset_index(drop=True)
        valid_df = df.iloc[
            self.train_sample_size : self.train_sample_size + self.valid_sample_size
        ].reset_index(drop=True)
        return train_df, valid_df

    def _load_test(self) -> pd.DataFrame:
        """Build a test split from civitai-337k to match baseline behavior.

        Returns:
            Dataframe containing the stratified test split.
        """
        ds = load_dataset(self.test_dataset_name, split="train")
        df = ds.to_pandas()
        df = df[["prompt", "negativePrompt", "nsfw"]]
        df = df.dropna(subset=["prompt", "nsfw"]).copy()
        df["negativePrompt"] = df["negativePrompt"].fillna("")
        df["prompt"] = df["prompt"].apply(self.preprocess)
        df["negativePrompt"] = df["negativePrompt"].apply(self.preprocess)
        _, test_df = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.random_state,
            shuffle=True,
            stratify=df["nsfw"],
        )
        return test_df.reset_index(drop=True)

    def split(self) -> dict[str, pd.DataFrame]:
        """Return cached splits or construct them.

        Returns:
            Dictionary with keys ``train``, ``valid``, and ``test``.
        """
        if self._splits is not None:
            return self._splits

        train_df, valid_df = self._load_samples()
        test_df = self._load_test()
        self._splits = {"train": train_df, "valid": valid_df, "test": test_df}
        return self._splits


def main(
    seed: int = 42,
    pred_path: str = "artifacts/pred/pred_nsfw_classifier_resume_on_Civitai-2m-prompts.pkl",
    model_path: str = "artifacts/model/JeremyFeng/nsfw-prompt-detection-resume-on-Civitai-2m-prompts",
    train_sample_size: int = 200_000,
    valid_sample_size: int = 20_000,
    test_size: float = DEFAULT_TEST_SIZE,
    **config_kwargs,
) -> None:
    """Train NSFW classifier on mixed datasets and save artifacts.

    Args:
        seed: Random seed for reproducibility.
        pred_path: Destination path for prediction pickle file.
        model_path: Directory to save model and tokenizer artifacts.
        train_sample_size: Number of Civitai-2m examples for training.
        valid_sample_size: Number of Civitai-2m examples for validation.
        test_size: Proportion to carve out test split from civitai-337k.
        **config_kwargs: Extra hyperparameters forwarded into ``TrainingConfig``.

    Returns:
        None. Artifacts are written to the provided paths.
    """
    seed_everything(seed)
    if "model_name" not in config_kwargs:
        config_kwargs["model_name"] = "JeremyFeng/nsfw-prompt-detection"
    config = TrainingConfig(**config_kwargs)
    loader = MixDatasetLoader(
        train_sample_size=train_sample_size,
        valid_sample_size=valid_sample_size,
        test_size=test_size,
        random_state=seed,
    )

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
