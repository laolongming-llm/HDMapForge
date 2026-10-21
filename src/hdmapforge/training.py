"""Reproducible Lightning training for the four-class local geometry baseline."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import lightning as L
import torch
import yaml
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from .data import GeometryDataModule
from .losses import MaskedBCEDiceLoss
from .model import GEOMETRY_CLASSES, GeometryUNet


class GeometryModule(L.LightningModule):
    def __init__(
        self,
        base_channels: int = 24,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        loss: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.network = GeometryUNet(base_channels=base_channels)
        self.criterion = MaskedBCEDiceLoss(**dict(loss or {}))
        # Keep rank-local metric state out of DDP's per-forward buffer broadcasts.
        self.val_intersection = torch.zeros(4, dtype=torch.float64)
        self.val_union = torch.zeros(4, dtype=torch.float64)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss = self.criterion(self(batch["image"]), batch["targets"], batch["valid"])
        self.log("train_loss", loss, on_step=True, on_epoch=True, batch_size=len(batch["image"]))
        return loss

    def on_validation_epoch_start(self) -> None:
        self.val_intersection = torch.zeros(4, dtype=torch.float64, device=self.device)
        self.val_union = torch.zeros(4, dtype=torch.float64, device=self.device)

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        logits = self(batch["image"])
        loss = self.criterion(logits, batch["targets"], batch["valid"])
        self.log("val_loss", loss, batch_size=len(batch["image"]), sync_dist=True)
        prediction, target, mask = (
            logits.sigmoid() >= 0.5,
            batch["targets"] >= 0.5,
            batch["valid"].bool(),
        )
        self.val_intersection += (prediction & target & mask).sum(dim=(0, 2, 3))
        self.val_union += ((prediction | target) & mask).sum(dim=(0, 2, 3))

    def on_validation_epoch_end(self) -> None:
        intersection, union = self.val_intersection.clone(), self.val_union.clone()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(intersection)
            torch.distributed.all_reduce(union)
        present = union > 0
        if present.any():
            self.log("val_mean_iou", (intersection[present] / union[present]).mean())
        for index, name in enumerate(GEOMETRY_CLASSES):
            if present[index]:
                self.log(f"val_{name}_iou", intersection[index] / union[index])

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )


def _resolve(path: str | Path, base: Path) -> Path:
    path = Path(path)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def train(config: Mapping[str, Any], *, config_dir: str | Path | None = None) -> Path:
    """Fit and return the best checkpoint. Config paths are relative to its YAML."""
    base = Path(config_dir or ".").resolve()
    L.seed_everything(int(config.get("seed", 42)), workers=True)
    data_config = dict(config["data"])
    data_config["manifest"] = _resolve(data_config["manifest"], base)
    datamodule = GeometryDataModule(**data_config)
    module = GeometryModule(
        **dict(config.get("model", {})),
        **dict(config.get("optimizer", {})),
        loss=config.get("loss"),
    )
    output = _resolve(config.get("output_dir", "runs/geometry"), base)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = ModelCheckpoint(
        dirpath=output / "checkpoints",
        filename="best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )
    callbacks = [checkpoint, LearningRateMonitor(logging_interval="epoch")]
    early_stopping = config.get("early_stopping")
    if early_stopping is not None:
        if not isinstance(early_stopping, Mapping):
            raise TypeError("early_stopping must be a mapping of Lightning EarlyStopping options")
        callbacks.append(EarlyStopping(**{
            "monitor": "val_loss", "mode": "min", **dict(early_stopping)
        }))
    trainer_config = {
        "max_epochs": 100,
        "accelerator": "auto",
        "devices": 1,
        "precision": "32-true",
        "deterministic": True,
        "gradient_clip_val": 1.0,
        "log_every_n_steps": 10,
        **dict(config.get("trainer", {})),
    }
    trainer = L.Trainer(
        **trainer_config,
        default_root_dir=output,
        logger=CSVLogger(save_dir=output, name="metrics"),
        callbacks=callbacks,
    )
    resume = config.get("resume_checkpoint")
    trainer.fit(
        module,
        datamodule=datamodule,
        ckpt_path=str(_resolve(resume, base)) if resume else None,
        weights_only=True,
    )
    if not checkpoint.best_model_path:
        raise RuntimeError("Training produced no best checkpoint; ensure validation is enabled")
    return Path(checkpoint.best_model_path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError("Training configuration must be a YAML mapping")
    print(train(config, config_dir=args.config.parent))


if __name__ == "__main__":
    main()
