"""Generic training loop wrapper for sentiment analysis models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from src.training.early_stopping import EarlyStopping
from src.training.utils import get_device

logger = logging.getLogger(__name__)


class Trainer:
    """Simple Trainer supporting train/validate steps and checkpointing."""

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        criterion: torch.nn.Module,
        optimizer: Optimizer,
        scheduler: _LRScheduler | None = None,
        device: torch.device | None = None,
        grad_clip: float | None = None,
        progress_bar: bool = False,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or get_device()
        self.grad_clip = grad_clip
        self.progress_bar = progress_bar

        self.history: list[dict[str, float]] = []

        self.model.to(self.device)
        logger.info("Trainer initialised. Device=%s", self.device)

    # ------------------------------------------------------------------
    # Epoch routines
    # ------------------------------------------------------------------
    def train_epoch(self) -> Tuple[float, float]:
        self.model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        from tqdm.auto import tqdm  # type: ignore

        iterator = tqdm(
            self.train_loader, disable=not self.progress_bar, desc="Train", leave=False
        )
        for batch in iterator:
            if isinstance(batch, dict):
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
            else:
                input_ids, labels = batch[0].to(self.device), batch[1].to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(input_ids)
            loss = self.criterion(logits, labels)
            loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            epoch_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = epoch_loss / total
        train_acc = correct / total
        return train_loss, train_acc

    @torch.no_grad()
    def evaluate(self) -> Tuple[float, float]:
        self.model.eval()
        if self.val_loader is None:
            return 0.0, 0.0
        total_loss, correct, total = 0.0, 0, 0
        iterator = self.val_loader
        if self.progress_bar:
            from tqdm.auto import tqdm  # type: ignore

            iterator = tqdm(self.val_loader, disable=False, desc="Eval", leave=False)
        for batch in iterator:
            if isinstance(batch, dict):
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
            else:
                input_ids, labels = batch[0].to(self.device), batch[1].to(self.device)
            logits = self.model(input_ids)
            loss = self.criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        val_loss = total_loss / total
        val_acc = correct / total
        return val_loss, val_acc

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def fit(
        self, epochs: int = 3, early_stopper: "EarlyStopping" | None = None
    ) -> None:
        """Run complete training process and store metrics in ``self.history``."""
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.evaluate()
            metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
            self.history.append(metrics)
            logger.info(
                "Epoch %d | Train loss %.4f acc %.3f | Val loss %.4f acc %.3f",
                epoch,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
            )

            # Early stopping on validation loss
            if early_stopper is not None:
                early_stopper.step(val_loss)
                if early_stopper.should_stop:
                    logger.info("Early stopping triggered at epoch %d", epoch)
                    break

    def save_checkpoint(
        self, path: str | Path, epoch: int, metrics: Dict[str, float]
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "metrics": metrics,
            },
            path,
        )
        logger.info("Saved checkpoint to %s", path)

    @staticmethod
    def load_checkpoint(
        path: str | Path, model: torch.nn.Module, optimizer: Optimizer | None = None
    ) -> Dict[str, float]:
        """Load checkpoint and restore model / optimizer state.

        Returns metrics dict stored in checkpoint.
        """
        path = Path(path)
        ckpt = torch.load(path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])  # type: ignore[arg-type]
        if optimizer is not None and "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])  # type: ignore[arg-type]
        logger.info("Loaded checkpoint from %s (epoch %s)", path, ckpt.get("epoch"))
        return ckpt.get("metrics", {})
