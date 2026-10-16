"""Load offline prepared NPZ patches; no raw-raster or SHP preprocessing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def read_manifest(manifest: str | Path) -> list[dict[str, Any]]:
    """Read samples and reject group leakage before constructing any loader."""
    manifest = Path(manifest).resolve()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    samples = payload.get("samples") if isinstance(payload, dict) else payload
    if not isinstance(samples, list) or not samples:
        raise ValueError("Manifest must contain a nonempty sample list or {'samples': [...]}.")
    groups: dict[str, str] = {}
    paths: set[Path] = set()
    entries = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict) or not {"path", "split", "group_id"} <= sample.keys():
            raise ValueError(f"Manifest sample {index} requires path, split and group_id")
        split = sample["split"]
        group = sample["group_id"]
        if (
            not isinstance(split, str)
            or split not in {"train", "val", "test"}
            or not isinstance(group, str)
            or not group.strip()
        ):
            raise ValueError(f"Invalid split/group_id in manifest sample {index}")
        if not isinstance(sample["path"], str) or not sample["path"]:
            raise ValueError(f"Invalid path in manifest sample {index}")
        path = Path(sample["path"])
        path = (manifest.parent / path).resolve() if not path.is_absolute() else path.resolve()
        if path.suffix.lower() != ".npz":
            raise ValueError(f"Expected NPZ sample: {path}")
        if path in paths:
            raise ValueError(f"Duplicate sample path in manifest: {path}")
        paths.add(path)
        if group in groups and groups[group] != split:
            raise ValueError(f"Spatial group {group!r} occurs in more than one split")
        groups[group] = split
        entries.append({"path": path, "split": split, "group_id": group})
    return entries


class ManifestDataset(Dataset):
    """Intensity float [0,1], targets 4HW float [0,1], valid binary 4HW.

    All samples in one batched loader need the same spatial dimensions. Physical
    sampling resolution and normalization parameters are fixed offline. Adjacent
    patches need independently chosen spatial groups with an appropriate buffer;
    group IDs alone cannot detect geographical adjacency.
    """

    def __init__(self, manifest: str | Path, split: str, *, augment: bool = False) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val or test")
        self.samples = [item for item in read_manifest(manifest) if item["split"] == split]
        if not self.samples:
            raise ValueError(f"Manifest has no samples for split={split!r}")
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        path = self.samples[index]["path"]
        with np.load(path, allow_pickle=False) as patch:
            if not {"image", "targets", "valid"} <= set(patch.files):
                raise ValueError(f"{path}: NPZ requires image, targets and valid arrays")
            image = np.asarray(patch["image"], dtype=np.float32)
            targets = np.asarray(patch["targets"], dtype=np.float32)
            valid = np.asarray(patch["valid"], dtype=np.float32)
        if image.ndim == 2:
            image = image[None]
        if image.ndim != 3 or image.shape[0] != 1 or min(image.shape[-2:]) < 8:
            raise ValueError(f"{path}: image must be HW or 1HW with H,W >= 8")
        expected = (4, *image.shape[-2:])
        if targets.shape != expected or valid.shape != expected:
            raise ValueError(f"{path}: targets and valid must have shape {expected}")
        for name, array in (("image", image), ("targets", targets), ("valid", valid)):
            if not np.isfinite(array).all() or (array < 0).any() or (array > 1).any():
                raise ValueError(f"{path}: {name} must be finite and within [0,1]")
        if not np.isin(valid, (0.0, 1.0)).all():
            raise ValueError(f"{path}: valid must be binary; use 0 for ignored pixels")
        arrays = [image, targets, valid]
        if self.augment:
            # Torch's worker seeds make this reproducible with seed_everything(workers=True).
            if torch.rand(()) < 0.5:
                arrays = [np.flip(array, axis=-1) for array in arrays]
            if torch.rand(()) < 0.5:
                arrays = [np.flip(array, axis=-2) for array in arrays]
            if image.shape[-2] == image.shape[-1]:
                k = int(torch.randint(0, 4, ()).item())
                arrays = [np.rot90(array, k, axes=(-2, -1)) for array in arrays]
        return {
            key: torch.from_numpy(np.ascontiguousarray(array))
            for key, array in zip(("image", "targets", "valid"), arrays, strict=True)
        }


class GeometryDataModule(L.LightningDataModule):
    def __init__(
        self,
        manifest: str | Path,
        batch_size: int = 2,
        num_workers: int = 0,
        augment: bool = True,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()
        if batch_size < 1 or num_workers < 0:
            raise ValueError("batch_size must be positive and num_workers nonnegative")
        self.manifest = Path(manifest)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augment = augment
        self.pin_memory = pin_memory

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            self.train_set = ManifestDataset(self.manifest, "train", augment=self.augment)
            self.val_set = ManifestDataset(self.manifest, "val")
        elif stage == "validate":
            self.val_set = ManifestDataset(self.manifest, "val")
        elif stage == "test":
            self.test_set = ManifestDataset(self.manifest, "test")

    def _loader(self, dataset: ManifestDataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_set, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_set)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_set)
