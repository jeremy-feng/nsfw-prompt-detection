"""Data loading and splitting implementation for NSFW prompt detection."""

import re

import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_DATASET_NAME = "thefcraft/civitai-stable-diffusion-337k"
DEFAULT_TEST_SIZE = 0.2
DEFAULT_VAL_SIZE = 0.1
DEFAULT_RANDOM_STATE = 42


class DatasetLoader:
    """Load, clean, and split the dataset into train/valid/test splits."""

    def __init__(
        self,
        dataset_name: str = DEFAULT_DATASET_NAME,
        test_size: float = DEFAULT_TEST_SIZE,
        val_size: float = DEFAULT_VAL_SIZE,
        random_state: int = DEFAULT_RANDOM_STATE,
    ) -> None:
        """Initialize loader configuration and split ratios.

        Args:
            dataset_name: Hugging Face dataset identifier.
            test_size: Proportion of full data reserved for testing.
            val_size: Proportion of remaining train data reserved for validation.
            random_state: Seed used when splitting rows.
        """
        self.dataset_name = dataset_name
        self.test_size = test_size
        self.val_size = val_size
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

    def load(self) -> pd.DataFrame:
        """Load and clean the dataset rows.

        Returns:
            Cleaned dataframe containing prompts, negative prompts, and labels.
        """
        from datasets import load_dataset

        ds = load_dataset(self.dataset_name)
        df = ds["train"].to_pandas()
        df = df[["prompt", "negativePrompt", "nsfw"]]
        df = df.dropna(subset=["prompt", "nsfw"]).copy()
        df["negativePrompt"] = df["negativePrompt"].fillna("")
        df["prompt"] = df["prompt"].apply(self.preprocess)
        df["negativePrompt"] = df["negativePrompt"].apply(self.preprocess)
        return df.reset_index(drop=True)

    def split(self) -> dict[str, pd.DataFrame]:
        """Split dataset into train/valid/test dataframes.

        Returns:
            Dictionary with keys ``train``, ``valid``, ``test``.
        """
        if self._splits is not None:
            return self._splits

        df = self.load()
        train_df, test_df = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.random_state,
            shuffle=True,
            stratify=df["nsfw"],
        )
        train_df, valid_df = train_test_split(
            train_df,
            test_size=self.val_size,
            random_state=self.random_state,
            shuffle=True,
            stratify=train_df["nsfw"],
        )

        self._splits = {
            "train": train_df.reset_index(drop=True),
            "valid": valid_df.reset_index(drop=True),
            "test": test_df.reset_index(drop=True),
        }
        return self._splits
