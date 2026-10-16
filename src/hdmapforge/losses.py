"""Masked supervision for reviewed positive and negative geometry regions."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite

import torch
from torch import nn
from torch.nn import functional as F


class MaskedBCEDiceLoss(nn.Module):
    """Class-balanced masked BCE plus Dice, with optional BCE positive weights.

    ``valid=0`` excludes uncertain label-alignment bands from both loss terms.
    Such bands must be prepared offline around reviewed centerline positives and
    trusted negatives. This loss does not estimate or correct registration offsets.
    """

    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        class_weights: Sequence[float] = (1.0, 1.0, 1.0, 1.0),
        positive_weights: Sequence[float] = (1.0, 1.0, 1.0, 1.0),
    ) -> None:
        super().__init__()
        weights = torch.tensor(class_weights, dtype=torch.float32)
        positive = torch.tensor(positive_weights, dtype=torch.float32)
        if weights.shape != (4,) or positive.shape != (4,):
            raise ValueError("class_weights and positive_weights must each have four values")
        if not torch.isfinite(weights).all() or not torch.isfinite(positive).all():
            raise ValueError("Loss weights must be finite")
        if (weights <= 0).any() or (positive <= 0).any():
            raise ValueError("Loss weights must be positive")
        if (
            not isfinite(bce_weight)
            or not isfinite(dice_weight)
            or bce_weight < 0
            or dice_weight < 0
            or bce_weight + dice_weight <= 0
        ):
            raise ValueError("At least one nonnegative loss coefficient must be positive")
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.register_buffer("class_weights", weights)
        self.register_buffer("positive_weights", positive.view(1, 4, 1, 1))

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        if logits.shape != targets.shape or logits.shape != valid.shape:
            raise ValueError("logits, targets and valid must have identical B4HW shapes")
        if logits.ndim != 4 or logits.shape[1] != 4:
            raise ValueError("Expected four geometry channels in B4HW format")
        # FP32 reductions avoid overflow/underflow for large patches under AMP.
        logits, targets, mask = logits.float(), targets.float(), valid.float()
        reduction_dims = (0, 2, 3)
        counts = mask.sum(dim=reduction_dims)
        effective_weights = self.class_weights * (counts > 0)
        denominator = effective_weights.sum().clamp_min(1e-6)
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none", pos_weight=self.positive_weights
        )
        bce = (bce * mask).sum(dim=reduction_dims) / counts.clamp_min(1.0)
        probabilities = logits.sigmoid()
        intersection = (probabilities * targets * mask).sum(dim=reduction_dims)
        total = ((probabilities + targets) * mask).sum(dim=reduction_dims)
        dice = 1.0 - (2.0 * intersection + 1e-6) / (total + 1e-6)
        combined = self.bce_weight * bce + self.dice_weight * dice
        return (combined * effective_weights).sum() / denominator
