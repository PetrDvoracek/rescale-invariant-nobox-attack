"""NEW-checkpoint reload tests — PROGRAM.md §6.1a (layer-2, CPU).

The §6.1a layer-2 fix changes both LightningModules to
``save_hyperparameters(ignore=[...])`` so the ``.ckpt`` stores only config/primitives;
submodules are reconstructed from config and passed to ``load_from_checkpoint``. This
suite exercises that NEW reload path on CPU (the CI-exercised path, §6.1a / §10):

- ``Trainee``: ignore ``["model", "model_sim"]``, keep ``epochs`` as a primitive hparam
  (CST-104: do NOT re-pass ``epochs=`` at load). Reconstruct ``model`` /``model_sim`` via
  the builders; assert ``m.model`` is not None and a forward works.
- ``TraineeAttacker``: ignore ``["model_attacker", "model_augmentfactor"]`` (both ctor
  args are modules; no primitive to restore). Reconstruct the U-Net and pass a stub F_Q.

Robust save: instantiate, then ``torch.save`` a Lightning-shaped dict (``state_dict`` +
``hyper_parameters`` + version) so the test does not depend on a Trainer or a real fit.
All CPU, ``pretrained=False`` / ``encoder_weights=None``, no network, no real checkpoint.
"""

from __future__ import annotations

import lightning as L
import torch
from torch import nn

from rina.config import AttackerConfig, QualityConfig
from rina.attacker import TraineeAttacker, build_attacker_unet
from rina.quality import Trainee, build_modelsim, build_quality, build_quality_backbone


def _save_lightning_ckpt(module: L.LightningModule, path) -> None:
    """Write a minimal Lightning-shaped checkpoint (no Trainer required)."""
    ckpt = {
        "state_dict": module.state_dict(),
        "hyper_parameters": dict(module.hparams),
        "pytorch-lightning_version": L.__version__,
    }
    torch.save(ckpt, str(path))


class StubFQ(nn.Module):
    """Minimal F_Q stand-in (a Module) for the attacker reconstruct path."""

    def forward(self, x, x_A):
        return torch.zeros((x.shape[0], 1))


def test_quality_new_checkpoint_reloads_cpu(tmp_path):
    cfg = QualityConfig(estimator_backbone="resnet18", quality_pretrained=False, epochs=1)
    m = build_quality(cfg)  # Trainee holding ModelSim + backbone + fc

    # `epochs` is kept OUT of ignore -> pickled as a primitive hparam.
    assert "epochs" in dict(m.hparams)
    # The two modules ARE excluded from hparams (§6.1a).
    assert "model" not in dict(m.hparams)
    assert "model_sim" not in dict(m.hparams)

    path = tmp_path / "fq.ckpt"
    _save_lightning_ckpt(m, path)

    # NEW reload path: reconstruct submodules from config; do NOT re-pass epochs.
    reloaded = Trainee.load_from_checkpoint(
        str(path),
        map_location="cpu",
        model=build_quality_backbone(cfg),
        model_sim=build_modelsim(),
    )
    assert reloaded.model is not None
    assert reloaded.model_sim is not None
    assert reloaded.epochs == 1

    # A forward works on CPU with tiny synthetic input. F_Q is used frozen/eval in
    # practice, so eval() it (the backbone's BatchNorm needs eval mode for batch=1).
    reloaded.eval()
    x = torch.rand(2, 3, 64, 64)
    out = reloaded(x, x)
    assert out.shape == (2, 1)
    assert torch.all((out >= 0) & (out <= 1))  # sigmoid head


def test_attacker_new_checkpoint_reloads_cpu(tmp_path):
    acfg = AttackerConfig()  # encoder_weights=None
    unet = build_attacker_unet(acfg)
    m = TraineeAttacker(model_attacker=unet, model_augmentfactor=StubFQ(), cfg=acfg)

    # Both ctor modules excluded from hparams (§6.1a).
    hp = dict(m.hparams)
    assert "model_attacker" not in hp
    assert "model_augmentfactor" not in hp

    path = tmp_path / "fa.ckpt"
    _save_lightning_ckpt(m, path)

    # NEW reload path: rebuild the U-Net, pass a stub F_Q for model_augmentfactor.
    reloaded = TraineeAttacker.load_from_checkpoint(
        str(path),
        map_location="cpu",
        model_attacker=build_attacker_unet(acfg),
        model_augmentfactor=StubFQ(),
    )
    assert reloaded.model_attacker is not None

    # A forward works on CPU (tiny single tile).
    x = torch.rand(1, 3, 64, 64)
    out = reloaded(x)
    assert out.shape == (1, 3, 64, 64)
    assert torch.all((out >= 0) & (out <= 1))  # sigmoid head
