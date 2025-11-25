"""Abstract base class for prompt classifiers driven by DatasetLoader."""

from abc import ABC, abstractmethod
from typing import Literal, Sequence

import pandas as pd

from data.data import DatasetLoader


class PromptClassifier(ABC):
    """Interface for prompt classifiers backed by a ``DatasetLoader``."""

    def _prepare_data(
        self, dataset_loader: DatasetLoader, subset: Literal["train", "valid", "test"]
    ) -> pd.DataFrame:
        """Extract a dataframe split with expected columns for prompts and labels.

        Args:
            dataset_loader: Loader that provides dataset splits.
            subset: Split name to fetch.

        Returns:
            A dataframe containing ``prompt``, ``negativePrompt``, and ``nsfw`` columns.
        """
        df = dataset_loader.split()[subset]
        return df[["prompt", "negativePrompt", "nsfw"]]

    @abstractmethod
    def _fit(
        self,
        prompts: Sequence[str] | pd.Series,
        negative_prompts: Sequence[str] | pd.Series,
        labels: Sequence[float] | pd.Series,
        **kwargs,
    ) -> "PromptClassifier":
        """Subclass hook to train or load model state.

        Args:
            prompts: Prompts text for the training split.
            negative_prompts: Negative prompts aligned with ``prompts``.
            labels: Ground-truth NSFW labels aligned with ``prompts``.
            **kwargs: Implementation-specific training options.

        Returns:
            Self to support method chaining.
        """

    def fit(self, dataset_loader: DatasetLoader) -> "PromptClassifier":
        """Train the model using the train and valid splits from the loader.

        Args:
            dataset_loader: Loader that provides the training and validation data.

        Returns:
            Self after fitting.
        """
        train_df = self._prepare_data(dataset_loader, subset="train")
        valid_df = self._prepare_data(dataset_loader, subset="valid")
        self._fit(
            prompts=train_df["prompt"],
            negative_prompts=train_df["negativePrompt"],
            labels=train_df["nsfw"],
            valid_prompts=valid_df["prompt"],
            valid_negative_prompts=valid_df["negativePrompt"],
            valid_labels=valid_df["nsfw"],
        )
        return self

    @abstractmethod
    def _predict(
        self,
        prompts: Sequence[str] | pd.Series,
        negative_prompts: Sequence[str] | pd.Series,
        **kwargs,
    ) -> Sequence[float]:
        """Subclass hook to generate probabilities for a batch of prompts.

        Args:
            prompts: Prompts to score.
            negative_prompts: Negative prompts aligned with ``prompts``.
            **kwargs: Implementation-specific inference options.

        Returns:
            Predicted probabilities of being NSFW for each prompt.
        """

    def predict(
        self,
        dataset_loader: DatasetLoader,
        subset: Literal["train", "valid", "test"] = "test",
    ) -> pd.Series:
        """Predict on a dataset split and return probabilities.

        Args:
            dataset_loader: Loader that provides dataset splits.
            subset: Split name to score; defaults to ``"test"``.

        Returns:
            Series of predicted probabilities indexed to the source split.
        """
        if self.tokenizer is None or self.model is None:
            raise RuntimeError("Tokenizer and model are not loaded. Call fit first.")
        df = self._prepare_data(dataset_loader, subset=subset)
        preds = self._predict(
            prompts=df["prompt"],
            negative_prompts=df["negativePrompt"],
        )
        return pd.Series(preds, index=df.index, name="pred")
