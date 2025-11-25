"""A fine-tuned classifier for NSFW prompt detection."""

from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Any, Optional, Sequence

import pkbar
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader, Dataset, RandomSampler
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    get_linear_schedule_with_warmup,
)

from model.base import PromptClassifier


@dataclass
class TrainingConfig:
    """Hyperparameters and toggles for fine-tuning."""

    model_name: str = "microsoft/deberta-v3-base"
    max_length: int = 256
    lr: float = 2e-5
    weight_decay: float = 0.01
    num_epochs: int = 10
    batch_size: int = 64
    warmup_ratio: float = 0.1
    grad_clip: float = 1.0
    gradient_checkpointing: bool = False
    fp16: bool = True
    grad_accum_steps: int = 1
    early_stop_patience: int | None = 3
    device: str = "auto"
    use_wandb: bool = False
    wandb_project: str = "nsfw-prompt-detection"
    wandb_run_name: str | None = None
    wandb_watch: bool = False
    wandb_log_every: int = 10


class PromptPairDataset(Dataset):
    """Torch dataset that tokenizes prompt/negative prompt pairs."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        prompts: Sequence[str],
        negative_prompts: Sequence[str],
        labels: Optional[Sequence[float]] = None,
        max_length: int = 256,
    ):
        """Construct the dataset with paired prompts and labels.

        Args:
            tokenizer: Tokenizer used to encode the prompt pairs.
            prompts: Prompts to classify.
            negative_prompts: Negative prompts aligned with ``prompts``.
            labels: Optional integer labels (0/1) aligned with ``prompts``.
            max_length: Maximum sequence length for truncation.
        """
        self.tokenizer = tokenizer
        self.prompts = list(prompts)
        self.negative_prompts = list(negative_prompts)
        self.labels = list(labels) if labels is not None else None
        self.max_length = max_length

    def __len__(self) -> int:
        """Return the number of prompt pairs."""
        return len(self.prompts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Tokenize a prompt pair and attach a label if available.

        Args:
            idx: Index of the example to retrieve.

        Returns:
            A dictionary of tokenized tensors and optional ``labels`` tensor.
        """
        encoding = self.tokenizer(
            self.prompts[idx],
            self.negative_prompts[idx],
            truncation=True,
            max_length=self.max_length,
        )
        output = {k: torch.tensor(v) for k, v in encoding.items()}
        # Attach label if available
        if self.labels is not None:
            output["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return output


class NSFWClassifier(PromptClassifier):
    """A fine-tuned DeBERTa-v3-base classifier for NSFW prompt detection."""

    def __init__(self, config: TrainingConfig | None = None):
        """Initialize the classifier with training configuration.

        Args:
            config: Optional overrides for training hyperparameters.
        """
        self.config = config or TrainingConfig()
        self.tokenizer: PreTrainedTokenizerBase | None = None
        self.model: PreTrainedModel | None = None
        self.device = torch.device(
            self.config.device
            if self.config.device != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

    def _build_model(self) -> None:
        """Load tokenizer/model and prepare label mappings."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        id2label = {0: "sfw", 1: "nsfw"}
        label2id = {"sfw": 0, "nsfw": 1}
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=2,
            id2label=id2label,
            label2id=label2id,
        )
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.model.to(self.device)

    def _get_data_loader(
        self,
        prompts: Sequence[str],
        negative_prompts: Sequence[str],
        labels: Optional[Sequence[float]] = None,
        *,
        batch_size: int,
        max_length: int,
        shuffle: bool = False,
    ) -> DataLoader:
        """Build a DataLoader for either training or inference.

        Args:
            prompts: Prompts used for training or inference.
            negative_prompts: Negative prompts aligned with ``prompts``.
            labels: Optional labels for supervised training.
            batch_size: Batch size for the loader.
            max_length: Tokenizer truncation length.
            shuffle: Whether to shuffle indices via sampler. Defaults to False.

        Returns:
            A configured ``DataLoader`` with padding collator.
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer must be initialized before dataloaders.")

        dataset = PromptPairDataset(
            tokenizer=self.tokenizer,
            prompts=prompts,
            negative_prompts=negative_prompts,
            labels=labels,
            max_length=max_length,
        )
        collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        sampler = RandomSampler(torch.arange(len(dataset))) if shuffle else None

        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            collate_fn=collator,
            pin_memory=True,
        )

    def _fit(
        self,
        prompts: Sequence[str],
        negative_prompts: Sequence[str],
        labels: Sequence[float],
        *,
        valid_prompts: Optional[Sequence[str]] = None,
        valid_negative_prompts: Optional[Sequence[str]] = None,
        valid_labels: Optional[Sequence[float]] = None,
        **kwargs: Any,
    ) -> "NSFWClassifier":
        """Fine-tune model on provided prompts.

        Args:
            prompts: Prompts used for training.
            negative_prompts: Negative prompts aligned with ``prompts``.
            labels: Labels aligned with ``prompts``. 0 for SFW, 1 for NSFW.
            valid_prompts: Optional validation prompts for monitoring training.
            valid_negative_prompts: Validation negative prompts aligned with
                ``valid_prompts``.
            valid_labels: Validation labels aligned with ``valid_prompts``. 0 for SFW, 1
                for NSFW.
            **kwargs: Optional overrides such as ``batch_size``, ``max_length`` and
                ``shuffle``.

        Returns:
            Self after updating model weights.
        """
        self._build_model()
        assert self.model is not None and self.tokenizer is not None

        train_loader = self._get_data_loader(
            prompts=prompts,
            negative_prompts=negative_prompts,
            labels=labels,
            batch_size=kwargs.get("batch_size", self.config.batch_size),
            max_length=kwargs.get("max_length", self.config.max_length),
            shuffle=kwargs.get("shuffle", True),
        )
        valid_loader: DataLoader | None = None
        if (
            valid_prompts is not None
            and valid_negative_prompts is not None
            and valid_labels is not None
        ):
            valid_loader = self._get_data_loader(
                prompts=valid_prompts,
                negative_prompts=valid_negative_prompts,
                labels=valid_labels,
                batch_size=kwargs.get("batch_size", self.config.batch_size),
                max_length=kwargs.get("max_length", self.config.max_length),
                shuffle=False,
            )

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        total_steps = (
            len(train_loader)
            * self.config.num_epochs
            // max(1, self.config.grad_accum_steps)
        )
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        loss_fn = nn.BCEWithLogitsLoss()
        amp_device_type = (
            self.device.type
            if self.device.type in {"cuda", "xpu", "hpu", "mlu", "mps"}
            else None
        )
        scaler = torch.amp.GradScaler(
            amp_device_type or "cuda",
            enabled=self.config.fp16 and amp_device_type is not None,
        )

        best_metric = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        bad_epochs = 0

        wandb_run = None
        if self.config.use_wandb:
            wandb_run = wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                config=asdict(self.config),
            )
            if self.config.wandb_watch:
                wandb.watch(self.model, log="all", log_freq=self.config.wandb_log_every)

        try:
            for epoch in range(self.config.num_epochs):
                self.model.train()
                total_loss = 0.0
                kbar = pkbar.Kbar(
                    target=len(train_loader),
                    epoch=epoch,
                    num_epochs=self.config.num_epochs,
                    width=20,
                )
                for step, batch in enumerate(train_loader):
                    labels_tensor = batch.pop("labels").to(self.device)
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    autocast_ctx = (
                        torch.amp.autocast(
                            device_type=amp_device_type, enabled=scaler.is_enabled()
                        )
                        if amp_device_type
                        else nullcontext()
                    )
                    with autocast_ctx:
                        outputs = self.model(**batch)
                        logits = outputs.logits.squeeze(-1)
                        if logits.ndim == 2:
                            logits = logits[:, 1]
                        loss = loss_fn(logits, labels_tensor.float())
                        # Scale for gradient accumulation
                        loss_to_backward = loss / max(1, self.config.grad_accum_steps)
                    scaler.scale(loss_to_backward).backward()
                    if (step + 1) % self.config.grad_accum_steps == 0:
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.grad_clip
                        )
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                        scheduler.step()
                    total_loss += loss.item()
                    global_step = epoch * len(train_loader) + step + 1
                    if (
                        wandb_run is not None
                        and (step + 1) % self.config.wandb_log_every == 0
                    ):
                        wandb.log(
                            {
                                "train/loss": loss.item(),
                                "train/lr": scheduler.get_last_lr()[0],
                                "train/epoch": epoch,
                            },
                            step=global_step,
                        )
                    kbar.update(current=step, values=[("loss", loss.item())])

                avg_loss = total_loss / len(train_loader)

                val_loss = None
                if valid_loader is not None:
                    self.model.eval()
                    val_total_loss = 0.0
                    with torch.inference_mode():
                        for batch in valid_loader:
                            labels_tensor = batch.pop("labels").to(self.device)
                            batch = {k: v.to(self.device) for k, v in batch.items()}
                            outputs = self.model(**batch)
                            logits = outputs.logits.squeeze(-1)
                            if logits.ndim == 2:
                                logits = logits[:, 1]
                            loss = loss_fn(logits, labels_tensor.float())
                            val_total_loss += loss.item()
                    val_loss = val_total_loss / len(valid_loader)

                kbar.update(
                    current=len(train_loader),
                    values=[
                        ("train_loss", avg_loss),
                        ("val_loss", val_loss)
                        if val_loss is not None
                        else ("loss", avg_loss),
                    ],
                )
                if wandb_run is not None:
                    epoch_step = (epoch + 1) * len(train_loader)
                    log_payload = {"train/loss_epoch": avg_loss, "epoch": epoch}
                    if val_loss is not None:
                        log_payload["val/loss"] = val_loss
                    wandb.log(log_payload, step=epoch_step)

                metric_to_track = val_loss if val_loss is not None else avg_loss
                if metric_to_track < best_metric:
                    best_metric = metric_to_track
                    best_state = {
                        k: v.cpu().clone() for k, v in self.model.state_dict().items()
                    }
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                    if (
                        self.config.early_stop_patience is not None
                        and bad_epochs >= self.config.early_stop_patience
                    ):
                        break
        finally:
            if wandb_run is not None:
                wandb_run.finish()

        if best_state is not None:
            self.model.load_state_dict(best_state)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return self

    def _predict(
        self,
        prompts: Sequence[str],
        negative_prompts: Sequence[str],
        **kwargs: Any,
    ) -> Sequence[float]:
        """Generate probabilities for a batch of prompts.

        Args:
            prompts: Prompts to score.
            negative_prompts: Negative prompts aligned with ``prompts``.
            **kwargs: Optional overrides such as ``batch_size``, ``max_length`` and
                ``shuffle``.

        Returns:
            Predicted probabilities of being NSFW for each prompt.
        """
        loader = self._get_data_loader(
            prompts=prompts,
            negative_prompts=negative_prompts,
            labels=None,
            batch_size=kwargs.get("batch_size", self.config.batch_size),
            max_length=kwargs.get("max_length", self.config.max_length),
            shuffle=kwargs.get("shuffle", False),
        )

        self.model.eval()
        pred: list[float] = []
        with torch.inference_mode():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                logits = outputs.logits.squeeze(-1)
                if logits.ndim == 2:
                    logits = logits[:, 1]
                batch_probs = torch.sigmoid(logits).detach().cpu().tolist()
                pred.extend(batch_probs)
        return pred
