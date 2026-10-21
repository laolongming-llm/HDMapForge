"""Exercise masking, split isolation and an actual checkpoint/inference cycle."""

import csv
import json

import numpy as np
import pytest
import torch

from hdmapforge.data import ManifestDataset, read_manifest
from hdmapforge.inference import load_predictor
from hdmapforge.losses import MaskedBCEDiceLoss
from hdmapforge.training import train


def test_ignored_regions_have_no_loss_gradient():
    logits = torch.zeros(1, 4, 8, 8, requires_grad=True)
    target = torch.zeros_like(logits)
    valid = torch.ones_like(logits)
    valid[:, :, :4] = 0
    target[:, :, :4] = 1
    loss = MaskedBCEDiceLoss()(logits, target, valid)
    loss.backward()
    assert torch.count_nonzero(logits.grad[:, :, :4]) == 0
    assert torch.count_nonzero(logits.grad[:, :, 4:]) > 0
    empty = MaskedBCEDiceLoss()(logits, target, torch.zeros_like(valid))
    assert empty.item() == 0 and torch.isfinite(empty)


def test_manifest_rejects_group_leakage(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            [
                {"path": "a.npz", "split": "train", "group_id": "same-road"},
                {"path": "b.npz", "split": "val", "group_id": "same-road"},
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="more than one split"):
        read_manifest(manifest)


@pytest.mark.parametrize("early_stop", [False, True])
def test_tiny_training_checkpoint_prediction(tmp_path, early_stop):
    # Synthetic data verifies mechanics only; it is not a road-accuracy estimate.
    torch.set_num_threads(1)
    image = np.zeros((32, 32), dtype=np.float32)
    targets = np.zeros((4, 32, 32), dtype=np.float32)
    targets[0, :, 7:10] = 1
    targets[1, 4:12, 18:23] = 1
    targets[2, 20:29, 16:29] = 1
    targets[3, 14:16, :] = 1
    image[:] = targets.max(axis=0)
    valid = np.ones_like(targets)
    samples = []
    for split in ("train", "val"):
        np.savez_compressed(tmp_path / f"{split}.npz", image=image, targets=targets, valid=valid)
        samples.append({"path": f"{split}.npz", "split": split, "group_id": split})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"samples": samples}), encoding="utf-8")
    assert ManifestDataset(manifest, "train")[0]["image"].shape == (1, 32, 32)
    checkpoint = train(
        {
            "seed": 13,
            "data": {"manifest": str(manifest), "batch_size": 1, "augment": False},
            "model": {"base_channels": 4},
            **({"optimizer": {"lr": 0.0},
                "early_stopping": {"patience": 0, "min_delta": 0.0}} if early_stop else {}),
            "output_dir": str(tmp_path / "run"),
            "trainer": {
                "max_epochs": 5 if early_stop else 1,
                "accelerator": "cpu",
                "devices": 1,
                "limit_train_batches": 1,
                "limit_val_batches": 1,
                "num_sanity_val_steps": 0,
                "enable_progress_bar": False,
                "enable_model_summary": False,
                "log_every_n_steps": 1,
            },
        }
    )
    assert checkpoint.exists()
    assert (checkpoint.parent / "last.ckpt").exists()
    if early_stop:
        # lr=0 keeps validation loss constant; stop after the first bad epoch,
        # well before the configured budget, and retain a reloadable checkpoint.
        # Lightning may reuse the unchanged best checkpoint for save_last.
        # The logged validation epochs establish the actual stopping time.
        with (tmp_path / "run/metrics/version_0/metrics.csv").open(newline="") as stream:
            validation_epochs = [int(row["epoch"]) for row in csv.DictReader(stream)
                                 if row.get("val_loss")]
        assert validation_epochs == [0, 1]
    predict = load_predictor(checkpoint)
    probability = predict(image)
    assert probability.shape == (4, 32, 32)
    assert np.isfinite(probability).all()
    assert probability.min() >= 0 and probability.max() <= 1
    assert np.allclose(probability, predict(image))
    with pytest.raises(ValueError, match="normalized"):
        predict(image * 255)
