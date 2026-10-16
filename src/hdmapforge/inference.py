"""Load trained geometry weights for tiled, georeferenced inference."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from .training import GeometryModule


def load_predictor(
    checkpoint_path: str | Path, device: str = "cpu"
) -> Callable[[np.ndarray], np.ndarray]:
    """Return an HW float[0,1] -> 4HW float[0,1] probability predictor.

    Channel order: lane, arrow, crosswalk, stopline. The caller owns geographic
    tiling, consistent offline normalization, overlap fusion and vector output.
    """
    target_device = torch.device(device)
    model = GeometryModule.load_from_checkpoint(
        checkpoint_path, map_location=target_device, weights_only=True
    )
    model.to(target_device).eval()

    def predict(image: np.ndarray) -> np.ndarray:
        image = np.asarray(image, dtype=np.float32)
        if image.ndim != 2 or min(image.shape) < 8:
            raise ValueError("Predictor expects an intensity image HW with H,W >= 8")
        if not np.isfinite(image).all() or (image < 0).any() or (image > 1).any():
            raise ValueError("Predictor intensity must be finite and normalized within [0,1]")
        tensor = torch.from_numpy(np.ascontiguousarray(image))[None, None].to(target_device)
        with torch.inference_mode():
            return model(tensor).sigmoid()[0].float().cpu().numpy()

    return predict
