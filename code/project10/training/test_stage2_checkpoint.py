"""Regression checks for Stage 2's Stage 1 dependency and training resume."""

import importlib
import math
import sys
import types
from unittest.mock import patch

import pytest
import torch


# The checkpoint helpers need no dataset or MONAI model. Avoid importing those
# GPU and compiled-extension dependencies in this CPU-only regression test.
dataset_module = types.ModuleType("code.project10.datasets.multiphase_paired_dataset")
dataset_module.MultiphasePairedDataset = object
synthesis_module = types.ModuleType("code.project10.models.synthesis_net")
synthesis_module.AnatomyConditionalSynthesisNet = object
with patch.dict(sys.modules, {
    "monai": types.ModuleType("monai"),
    "monai.networks": types.ModuleType("monai.networks"),
    "monai.networks.nets": types.ModuleType("monai.networks.nets"),
    dataset_module.__name__: dataset_module,
    synthesis_module.__name__: synthesis_module,
}):
    trainer = importlib.import_module("code.project10.training.train_stage2_synthesis")
AutoEncoder3DConfig = trainer.AutoEncoder3DConfig
build_autoencoder = trainer.build_autoencoder
CONDITIONING = {"totalseg_task": "total", "mask_dir": "/data/masks"}


def _tiny_ae_checkpoint(path, use_vq=False):
    cfg = AutoEncoder3DConfig(
        input_size=(16, 16, 16), channels=(8, 8), strides=(2, 1),
        latent_channels=4, use_vq=use_vq,
    )
    model = build_autoencoder(cfg)
    torch.save({"model": model.state_dict(), "cfg": vars(cfg)}, path)
    return model


def _training_objects():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
    return model, optimizer, scheduler


def test_stage1_checkpoint_must_exist(tmp_path):
    with pytest.raises(FileNotFoundError, match="Stage 1.*checkpoint"):
        trainer.load_stage1_autoencoder(tmp_path / "missing.pt", torch.device("cpu"))


def test_stage1_checkpoint_must_be_continuous(tmp_path):
    path = tmp_path / "vq.pt"
    _tiny_ae_checkpoint(path, use_vq=True)
    with pytest.raises(ValueError, match="continuous"):
        trainer.load_stage1_autoencoder(path, torch.device("cpu"))


def test_stage1_checkpoint_loads_every_weight_and_freezes(tmp_path):
    path = tmp_path / "continuous.pt"
    source = _tiny_ae_checkpoint(path)
    loaded = trainer.load_stage1_autoencoder(path, torch.device("cpu"))
    assert loaded.training is False
    assert all(not param.requires_grad for param in loaded.parameters())
    assert all(torch.equal(loaded.state_dict()[key], value)
               for key, value in source.state_dict().items())


def test_stage1_checkpoint_rejects_partial_weights(tmp_path):
    path = tmp_path / "partial.pt"
    _tiny_ae_checkpoint(path)
    payload = torch.load(path, weights_only=False)
    payload["model"].pop(next(iter(payload["model"])))
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="incompatible"):
        trainer.load_stage1_autoencoder(path, torch.device("cpu"))


def test_resume_restores_scheduler_optimizer_and_best_loss(tmp_path):
    model, optimizer, scheduler = _training_objects()
    model(torch.ones(1, 2)).sum().backward()
    for _ in range(4):
        optimizer.step()
        scheduler.step()
    expected_lr = optimizer.param_groups[0]["lr"]
    path = tmp_path / "last.pt"
    torch.save({"epoch": 4, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "best_val_loss": 0.25, "ae_checkpoint_sha256": "same-ae",
                "conditioning": CONDITIONING}, path)

    resumed_model, resumed_optimizer, resumed_scheduler = _training_objects()
    epoch, best = trainer.restore_stage2_checkpoint(
        path, resumed_model, resumed_optimizer, resumed_scheduler,
        torch.device("cpu"), total_epochs=10, ae_checkpoint_sha256="same-ae",
        conditioning=CONDITIONING,
    )
    assert (epoch, best) == (4, 0.25)
    assert resumed_scheduler.last_epoch == 4
    assert resumed_optimizer.param_groups[0]["lr"] == pytest.approx(expected_lr)
    assert resumed_optimizer.state_dict()["state"]
    resumed_optimizer.step()
    resumed_scheduler.step()
    expected_next_lr = 1e-6 + (1e-3 - 1e-6) * (1 + math.cos(math.pi * 5 / 10)) / 2
    assert resumed_optimizer.param_groups[0]["lr"] == pytest.approx(expected_next_lr)


def test_legacy_resume_continues_epoch_and_lr_without_new_fields(tmp_path):
    model, optimizer, _ = _training_objects()
    path = tmp_path / "legacy.pt"
    torch.save({"epoch": 4, "model": model.state_dict(),
                "optimizer": optimizer.state_dict()}, path)
    resumed_model, resumed_optimizer, resumed_scheduler = _training_objects()
    epoch, best = trainer.restore_stage2_checkpoint(
        path, resumed_model, resumed_optimizer, resumed_scheduler,
        torch.device("cpu"), total_epochs=10, ae_checkpoint_sha256="same-ae",
        conditioning=CONDITIONING,
        allow_legacy_ae_resume=True,
    )
    assert epoch == 4
    assert math.isinf(best)
    assert resumed_scheduler.last_epoch == 4
    expected_lr = 1e-6 + (1e-3 - 1e-6) * (1 + math.cos(math.pi * 4 / 10)) / 2
    assert resumed_optimizer.param_groups[0]["lr"] == pytest.approx(expected_lr)


def test_legacy_resume_preserves_best_checkpoint_metric(tmp_path):
    model, optimizer, _ = _training_objects()
    path = tmp_path / "last_synthesis_net.pt"
    torch.save({"epoch": 4, "model": model.state_dict(),
                "optimizer": optimizer.state_dict()}, path)
    torch.save({"epoch": 3, "model": model.state_dict(), "val_loss": 0.2},
               tmp_path / "best_synthesis_net.pt")
    resumed_model, resumed_optimizer, resumed_scheduler = _training_objects()
    _, best = trainer.restore_stage2_checkpoint(
        path, resumed_model, resumed_optimizer, resumed_scheduler,
        torch.device("cpu"), total_epochs=10, ae_checkpoint_sha256="same-ae",
        conditioning=CONDITIONING,
        allow_legacy_ae_resume=True,
    )
    assert best == 0.2


def test_resume_rejects_different_autoencoder_even_with_override(tmp_path):
    model, optimizer, scheduler = _training_objects()
    path = tmp_path / "last.pt"
    torch.save({"epoch": 4, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "ae_checkpoint_sha256": "old-ae", "conditioning": CONDITIONING}, path)
    resumed_model, resumed_optimizer, resumed_scheduler = _training_objects()
    with pytest.raises(ValueError, match="different Stage 1 AE"):
        trainer.restore_stage2_checkpoint(
            path, resumed_model, resumed_optimizer, resumed_scheduler,
            torch.device("cpu"), total_epochs=10, ae_checkpoint_sha256="new-ae",
            conditioning=CONDITIONING,
            allow_legacy_ae_resume=True,
        )


def test_legacy_resume_requires_explicit_override(tmp_path):
    model, optimizer, scheduler = _training_objects()
    path = tmp_path / "last.pt"
    torch.save({"epoch": 4, "model": model.state_dict(),
                "optimizer": optimizer.state_dict()}, path)
    with pytest.raises(ValueError, match="allow_legacy_ae_resume"):
        trainer.restore_stage2_checkpoint(
            path, model, optimizer, scheduler,
            torch.device("cpu"), total_epochs=10, ae_checkpoint_sha256="new-ae",
            conditioning=CONDITIONING,
        )


def test_totalseg_task_is_passed_to_dataset(monkeypatch, tmp_path):
    received = {}

    class DatasetReached(Exception):
        pass

    def capture_dataset(**kwargs):
        received.update(kwargs)
        raise DatasetReached

    monkeypatch.setattr(trainer, "MultiphasePairedDataset", capture_dataset)
    monkeypatch.setattr(trainer, "setup_ddp", lambda: (0, 1, 0, False))
    monkeypatch.setattr(trainer, "load_stage1_autoencoder", lambda *_args: object())
    monkeypatch.setattr(trainer, "checkpoint_sha256", lambda *_args: "trained-ae")
    monkeypatch.setattr(sys, "argv", ["train_stage2_synthesis.py", "--out_dir", str(tmp_path),
                                    "--totalseg_task", "gi"])
    with pytest.raises(DatasetReached):
        trainer.main()
    assert received["totalseg_task"] == "gi"


def test_new_checkpoint_records_resume_state_and_ae_identity():
    model, optimizer, scheduler = _training_objects()
    payload = trainer.stage2_checkpoint_payload(
        epoch=4, model=model, optimizer=optimizer, scheduler=scheduler,
        best_val_loss=0.2, ae_checkpoint_sha256="trained-ae", conditioning=CONDITIONING,
    )
    assert payload["epoch"] == 4
    assert payload["model"].keys() == model.state_dict().keys()
    assert payload["optimizer"] == optimizer.state_dict()
    assert payload["scheduler"] == scheduler.state_dict()
    assert payload["best_val_loss"] == 0.2
    assert payload["ae_checkpoint_sha256"] == "trained-ae"
    assert payload["conditioning"] == CONDITIONING


@pytest.mark.parametrize("different", [
    {"totalseg_task": "total_v1", "mask_dir": "/data/masks"},
    {"totalseg_task": "total", "mask_dir": "/other/masks"},
])
def test_resume_rejects_different_mask_conditioning_even_with_override(tmp_path, different):
    model, optimizer, scheduler = _training_objects()
    path = tmp_path / "last.pt"
    torch.save({"epoch": 4, "model": model.state_dict(),
                "ae_checkpoint_sha256": "same-ae", "conditioning": CONDITIONING}, path)
    with pytest.raises(ValueError, match="different mask conditioning"):
        trainer.restore_stage2_checkpoint(
            path, model, optimizer, scheduler, torch.device("cpu"),
            total_epochs=10, ae_checkpoint_sha256="same-ae",
            conditioning=different, allow_legacy_ae_resume=True,
        )


def test_resume_without_mask_identity_requires_explicit_override(tmp_path):
    model, optimizer, scheduler = _training_objects()
    path = tmp_path / "last.pt"
    torch.save({"epoch": 4, "model": model.state_dict(),
                "ae_checkpoint_sha256": "same-ae"}, path)
    with pytest.raises(ValueError, match="unaudited mask conditioning"):
        trainer.restore_stage2_checkpoint(
            path, model, optimizer, scheduler, torch.device("cpu"),
            total_epochs=10, ae_checkpoint_sha256="same-ae",
            conditioning=CONDITIONING,
        )
    epoch, _ = trainer.restore_stage2_checkpoint(
        path, model, optimizer, scheduler, torch.device("cpu"),
        total_epochs=10, ae_checkpoint_sha256="same-ae",
        conditioning=CONDITIONING, allow_legacy_ae_resume=True,
    )
    assert epoch == 4


def test_mask_identity_uses_resolved_directory(tmp_path):
    identity = trainer.mask_conditioning_identity("total", tmp_path, "masks/../masks")
    assert identity == {"totalseg_task": "total", "mask_dir": str(tmp_path / "masks")}


def test_main_rejects_missing_ae_before_empty_dataset(monkeypatch, tmp_path):
    class EmptyDataset:
        def __init__(self, **_kwargs):
            pass

        def __len__(self):
            return 0

    monkeypatch.setattr(trainer, "MultiphasePairedDataset", EmptyDataset)
    monkeypatch.setattr(trainer, "setup_ddp", lambda: (0, 1, 0, False))
    monkeypatch.setattr(sys, "argv", ["train_stage2_synthesis.py", "--out_dir", str(tmp_path),
                                    "--ae_checkpoint", str(tmp_path / "missing.pt")])
    with pytest.raises(FileNotFoundError, match="Stage 1.*checkpoint"):
        trainer.main()
