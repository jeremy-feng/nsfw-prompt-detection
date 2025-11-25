"""TensorFlow/Keras baseline classifier implementation."""

from typing import Any, Callable, List, Optional, Sequence, Union, override

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

from model.base import PromptClassifier
from nsfw_prompt_detection_sd.inference_demo import load_resources, preprocess

MAX_SEQUENCE_LENGTH: int = 50
DEFAULT_TOKENIZER_PATH = "nsfw_prompt_detection_sd/tokenizer.json"
DEFAULT_MODEL_PATH = "nsfw_prompt_detection_sd/nsfw_classifier.keras"


class BaselineClassifier(PromptClassifier):
    """Keras-based classifier that mirrors the existing baseline behavior.

    References:
        - https://github.com/thefcraft/nsfw-prompt-detection-sd
        - https://github.com/jeremy-feng/nsfw-prompt-detection-sd
    """

    def __init__(
        self,
        tokenizer_path: str = DEFAULT_TOKENIZER_PATH,
        model_path: str = DEFAULT_MODEL_PATH,
        max_len: int = MAX_SEQUENCE_LENGTH,
        preprocess_fn: Callable[[Union[str, List[str]]], Union[str, List[str]]]
        | None = None,
    ) -> None:
        """Initialize the baseline classifier.

        Args:
            tokenizer_path: Path to the tokenizer JSON file.
            model_path: Path to the Keras model file.
            max_len: Maximum sequence length for padding.
            preprocess_fn: Optional function for preprocessing input text.
        """
        self.tokenizer_path = tokenizer_path
        self.model_path = model_path
        self.max_len = max_len
        self.preprocess_fn = preprocess_fn or preprocess
        self.tokenizer: Optional[Any] = None
        self.model: Optional[Any] = None

    def _fit(self):
        """This method is a no-op for the baseline classifier."""
        pass

    @override
    def fit(self) -> "BaselineClassifier":
        """Load tokenizer and model resources."""
        self.tokenizer, self.model = load_resources(
            tokenizer_path=self.tokenizer_path,
            model_path=self.model_path,
        )
        return self

    def _predict(
        self,
        prompts: Sequence[str],
        negative_prompts: Sequence[str],
    ) -> Sequence[float]:
        """Generate probabilities for a batch of prompts.

        Args:
            prompts: Prompts to score.
            negative_prompts: Negative prompts aligned with ``prompts``.

        Returns:
            Predicted probabilities of being NSFW for each prompt.
        """
        processed_prompts = [self.preprocess_fn(p) for p in prompts]
        processed_negative = [self.preprocess_fn(n) for n in negative_prompts]

        x_seq = self.tokenizer.texts_to_sequences(processed_prompts)
        z_seq = self.tokenizer.texts_to_sequences(processed_negative)
        x_pad = pad_sequences(x_seq, maxlen=self.max_len)
        z_pad = pad_sequences(z_seq, maxlen=self.max_len)

        pred = self.model.predict([x_pad, z_pad], verbose=0)
        return np.asarray(pred).flatten().tolist()
