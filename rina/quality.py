"""F_Q — the FMQ quality estimator (paper §4.2.1, trained per §5.3).

F_Q predicts the "augmentation factor" of an (original, augmented) pair: a scalar in
``[0, 1]`` measuring how heavily the second image was distorted. It is the frozen Sobel bank
F_S (:class:`~rina.simulation.ModelSim`) → a timm backbone → a bias-free linear head →
sigmoid. F_S emits 36 channels per image; the backbone's 72-channel input is
``cat(F_S(orig), F_S(aug))``, concatenated in :meth:`Trainee.forward`.

Once trained, F_Q is frozen and reused as the adversarial loss term when training the
attacker F_A: :class:`~rina.attacker.TraineeAttacker` holds a whole ``Trainee`` and calls it
as ``fq(orig, aug)``. :func:`build_quality` is that factory.
"""

from __future__ import annotations

import lightning as L
import timm
import torch
from torch import nn

from .config import QualityConfig
from .simulation import build_modelsim


def _probe_output_shape(model, resolution=224, device=None):
    """Output shape of an F_Q backbone on a dummy forward pass.

    Parameters
    ----------
    model : torch.nn.Module
        The backbone to probe.
    resolution : int, optional
        Spatial size of the dummy input.
    device : torch.device or None, optional
        Device for the dummy input; defaults to the model's own device.

    Returns
    -------
    torch.Size
        The backbone's output shape. The probe input has 72 channels — the backbone's
        ``in_chans``, i.e. ``cat(F_S(x), F_S(x_aug))``.
    """
    if device is None:
        device = next(model.parameters()).device
    x = torch.randn(1, 72, resolution, resolution).to(device)
    return model(x).shape


class Trainee(L.LightningModule):
    """The FMQ quality estimator F_Q.

    The two ``nn.Module`` arguments are excluded from ``save_hyperparameters`` and
    reconstructed by the builders at load time, so ``epochs`` / ``lr`` / ``loss`` keep their
    defaults to let old checkpoints (whose hparams predate those fields) still load.

    Parameters
    ----------
    model : torch.nn.Module
        The 72-channel timm backbone (built by :func:`build_quality_backbone`).
    model_sim : torch.nn.Module
        The frozen F_S bank, shared across both images.
    epochs : int
        Total training epochs; places the ``MultiStepLR`` milestones.
    lr : float, optional
        AdamW learning rate.
    loss : {"l1", "mse"}, optional
        Label-loss selector.
    """

    def __init__(self, model, model_sim, epochs, lr=1e-3, loss="l1"):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "model_sim"])
        self.epochs = epochs
        self.model = model
        self.model_sim = model_sim
        output_shape = _probe_output_shape(model)
        assert len(output_shape) == 2, f"expected 2D backbone output, got {output_shape}"
        self.fc = nn.Linear(output_shape[-1], 1, bias=False)
        self.criterion = nn.MSELoss() if loss == "mse" else nn.L1Loss()
        self.mse = nn.MSELoss()

    def forward(self, im_orig, im_aug):
        """Score an (original, augmented) pair.

        Parameters
        ----------
        im_orig, im_aug : torch.Tensor
            Image batches of shape ``(N, 3, H, W)``.

        Returns
        -------
        torch.Tensor
            Augmentation-factor scores of shape ``(N, 1)`` in ``[0, 1]``.
        """
        x = torch.cat([self.model_sim(im_orig), self.model_sim(im_aug)], dim=1)
        return torch.sigmoid(self.fc(self.model(x)))

    def configure_optimizers(self):
        """Build the AdamW optimizer and ``MultiStepLR`` schedule (milestones at 0.8/0.9/0.95)."""
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[
                int(self.epochs * 0.8),
                int(self.epochs * 0.9),
                int(self.epochs * 0.95),
            ],
        )
        return [optimizer], [scheduler]

    def _step(self, batch, idx, stage):
        """Shared train/val step: forward the pair, log MAE loss and MSE."""
        im_orig, im_aug, y = batch
        o = self(im_orig, im_aug).squeeze()
        loss = self.criterion(o, y)
        self.log_dict(
            {f"{stage}_loss_mae": loss.item(), f"{stage}_mse": self.mse(o, y).item()},
            on_epoch=True,
        )
        return loss

    def training_step(self, batch, idx):
        return self._step(batch, idx, stage="train")

    def validation_step(self, batch, idx):
        self._step(batch, idx, stage="val")


def build_quality_backbone(cfg: QualityConfig) -> nn.Module:
    """Build the bare 72-channel timm backbone for F_Q (headless, ``num_classes=0``).

    With ``pretrained=True`` timm downloads weights and adapts the input conv from 3 to 72
    channels. This is not a ``Trainee`` — it is the raw backbone fed to :func:`build_quality`.

    Parameters
    ----------
    cfg : QualityConfig
        ``cfg.estimator_backbone`` is the timm model name; ``cfg.quality_pretrained`` selects
        pretrained weights.

    Returns
    -------
    nn.Module
        A headless timm backbone accepting 72 input channels.
    """
    return timm.create_model(
        cfg.estimator_backbone,
        pretrained=cfg.quality_pretrained,
        num_classes=0,
        in_chans=72,
    )


def build_quality(cfg: QualityConfig) -> Trainee:
    """Assemble the whole F_Q (F_S + backbone + linear head), callable as ``fq(orig, aug)``.

    Parameters
    ----------
    cfg : QualityConfig
        Drives the backbone, F_S post-processing, and the ``epochs`` / ``lr`` / ``loss`` hparams.

    Returns
    -------
    Trainee
        The complete F_Q. This is exactly what ``TraineeAttacker.model_augmentfactor`` holds.
    """
    return Trainee(
        model=build_quality_backbone(cfg),
        model_sim=build_modelsim(),
        epochs=cfg.epochs,
        lr=cfg.lr,
        loss=cfg.loss,
    )
