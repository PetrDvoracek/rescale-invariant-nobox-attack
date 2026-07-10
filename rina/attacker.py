"""F_A — the U-Net attacker (paper §4.1, trained per §5.1).

:class:`TraineeAttacker` is the Lightning wrapper around an ``smp.Unet``. Training minimizes
the Eq.(8) compound loss (:func:`rina.losses.compound_loss`) and logs its three scalar
components plus the active-term ratio under every logger backend (image grids are logged on
wandb only). :func:`build_attacker_unet` rebuilds the U-Net from config so NEW checkpoints
are import-path-independent.

The validation step runs a frozen torchvision ``resnet18`` on single 224 ImageNet-val crops
(no tiling) and logs ``surrogate_val_top1``. This is a training-time *monitoring proxy* to
watch attack strength live — it is not F_N, not a timm victim, and not the
resolution-invariant tiled evaluation. The paper's no-box accuracy comes only from
``rina evaluate`` on the tiled victims (:mod:`rina.victims`). To keep the two sites
independent, the surrogate lives here and this module never imports ``rina.victims``.
"""

from __future__ import annotations

import lightning as L
import segmentation_models_pytorch as smp
import torch
import torchvision

from rina import metrics
from rina.config import AttackerConfig
from rina.losses import compound_loss

# ImageNet-1k normalization for the monitoring proxy. Kept here so victims.py need not be
# imported.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def log_images(logger, name: str, grid) -> None:
    """Log one image grid, a no-op under non-wandb backends.

    ``logger.log_image`` is wandb-specific and crashes under ``CSVLogger`` / ``None``; gating
    on the capability lets the scalar metrics log everywhere while grids drop off wandb.

    Parameters
    ----------
    logger : object or None
        The active Lightning logger (``self.logger``); may be ``None`` or a backend without
        ``log_image`` (e.g. ``CSVLogger``).
    name : str
        Metric key for the grid (e.g. ``"train_ims_orig"``).
    grid : torch.Tensor
        A ``uint8`` image grid as produced by :meth:`TraineeAttacker._ims2grid`.
    """
    if logger is not None and hasattr(logger, "log_image"):
        logger.log_image(name, [grid])


def build_attacker_unet(cfg: AttackerConfig) -> smp.Unet:
    """Build the F_A U-Net from config (the reconstruct factory).

    NEW checkpoints rebuild the U-Net here at load time, so the ``.ckpt`` is
    import-path-independent.

    Parameters
    ----------
    cfg : AttackerConfig
        Supplies ``unet_backbone`` (the ``smp`` encoder name) and ``encoder_weights``.

    Returns
    -------
    smp.Unet
        A U-Net with ``classes=3`` and a sigmoid head (output in ``[0, 1]``).
    """
    return smp.Unet(
        encoder_name=cfg.unet_backbone,
        encoder_weights=cfg.encoder_weights,
        classes=3,
        activation="sigmoid",
    )


class TraineeAttacker(L.LightningModule):
    """Lightning training wrapper for the F_A U-Net attacker (paper §4.1).

    Parameters
    ----------
    model_attacker : torch.nn.Module
        The ``smp.Unet`` attacker (see :func:`build_attacker_unet`).
    model_augmentfactor : torch.nn.Module
        The frozen F_Q estimator, called as ``fq(orig, aug) -> [0, 1]``. Its parameters are
        frozen and it is set to ``eval()`` in ``__init__``.
    cfg : AttackerConfig or None, optional
        The attacker config. Defaults to ``AttackerConfig()`` so OLD checkpoints (whose
        hparams predate the field) still load.
    """

    # The validation monitor's single fixed blend factor. Kept as a tuple so adding probes
    # preserves the per-alpha breakdown.
    _MONITOR_ALPHAS = (1.0,)

    def __init__(self, model_attacker, model_augmentfactor, cfg: AttackerConfig | None = None):
        super().__init__()
        self.cfg = cfg or AttackerConfig()
        self.model_attacker = model_attacker

        # F_Q is a fixed loss term, never trained through F_A.
        self.model_augmentfactor = model_augmentfactor
        for param in self.model_augmentfactor.parameters():
            param.requires_grad_(False)
        self.model_augmentfactor.eval()

        # Built lazily (see _get_monitor) so a train-only / CPU-smoke build needs no download.
        self._monitor = None

        self.save_hyperparameters(ignore=["model_attacker", "model_augmentfactor"])

    def _get_monitor(self) -> torch.nn.Module:
        """Return the frozen torchvision ``resnet18`` monitoring proxy, built lazily and cached.

        Built on first use (not in ``__init__``) so train-only / CPU-smoke runs need no
        pretrained-weight download.

        Returns
        -------
        torch.nn.Module
            The cached, frozen ``resnet18`` on the module's current device.
        """
        if self._monitor is None:
            monitor = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
            monitor.eval()
            for param in monitor.parameters():
                param.requires_grad_(False)
            self._monitor = monitor.to(self.device)
        return self._monitor

    def forward(self, x):
        """Run the attacker.

        Parameters
        ----------
        x : torch.Tensor
            Image batch of shape ``(N, 3, H, W)`` in ``[0, 1]``.

        Returns
        -------
        torch.Tensor
            The perturbed image, same shape, in ``[0, 1]``.
        """
        return self.model_attacker(x)

    def configure_optimizers(self):
        """AdamW + per-step cosine warm restarts (paper §5.1).

        ``T_0`` is in steps (``cfg.lr_T_0_steps``), so the scheduler steps once per optimizer
        step rather than once per epoch.
        """
        optimizer = torch.optim.AdamW(self.model_attacker.parameters(), lr=self.cfg.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.cfg.lr_T_0_steps,
            T_mult=self.cfg.lr_T_mult,
            eta_min=self.cfg.lr_eta_min,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def _ims2grid(self, ims, n=32):
        """Tile up to ``n`` images into a single ``uint8`` grid."""
        return (torchvision.utils.make_grid(ims[:n], nrow=4) * 255).to(torch.uint8)

    def _similarity_active_flag(self, L_S, L_A) -> float:
        """Flag whether the similarity term wins the ``max()``.

        Uses the same scaling as :func:`rina.losses.compound_loss` (β on the adversarial
        term); with ``on_epoch`` averaging the per-step flag becomes the fraction of steps
        the similarity term wins.

        Parameters
        ----------
        L_S, L_A : torch.Tensor
            The unscaled similarity and adversarial loss terms.

        Returns
        -------
        float
            ``1.0`` if the similarity term wins, else ``0.0``.
        """
        return 1.0 if L_S >= (self.cfg.beta * L_A) else 0.0

    def _log_dict_if_attached(self, log_dict) -> None:
        """Log ``log_dict`` only when a Trainer is attached, so a bare step call still works."""
        if self._trainer is not None:
            self.log_dict(log_dict, on_epoch=True, prog_bar=True)

    def _step(self, batch, idx, stage):
        """Forward the crops, compute the Eq.(8) loss, and log the scalars (and grids at idx==1).

        Parameters
        ----------
        batch : torch.Tensor
            A batch of 224 crops.
        idx : int
            Batch index; the image grids are logged only at ``idx == 1``.
        stage : str
            ``"train"`` or ``"val"``, used as the metric-key prefix.

        Returns
        -------
        torch.Tensor
            The Eq.(8) gate loss.
        """
        ims = batch.squeeze()

        if idx == 1:
            log_images(self.logger, f"{stage}_ims_orig", self._ims2grid(ims.detach().cpu()))

        out = self(ims)
        loss, L_S, L_A = compound_loss(ims, out, self.model_augmentfactor, self.cfg)

        if idx == 1:
            log_images(self.logger, f"{stage}_ims_adv", self._ims2grid(out.detach().cpu()))
            log_images(
                self.logger,
                f"{stage}_diff",
                self._ims2grid(torch.abs(ims.detach().cpu() - out.detach().cpu())),
            )

        self._log_dict_if_attached(
            {
                f"{stage}_loss": loss,
                "loss_similarity": L_S,
                "loss_adversarial": L_A,
                "loss_similarity_active": self._similarity_active_flag(L_S, L_A),
            }
        )
        return loss

    def training_step(self, batch, idx):
        return self._step(batch, idx, stage="train")

    def validation_step(self, batch, idx):
        """Monitor attack strength via the frozen resnet18 proxy (not the no-box result).

        Runs on single 224 ImageNet-val crops (no tiling): builds ``adv_pattern = F_A(x) - x``,
        blends at the fixed ``alpha = 1.0``, ImageNet-normalizes, and logs ``surrogate_val_top1``.

        Parameters
        ----------
        batch : tuple of torch.Tensor
            An ``(x_orig, y_label)`` pair of crops and integer labels.
        idx : int
            Batch index; the image grids are logged only at ``idx == 1``.
        """
        x_orig, y_lbl = batch
        x_adv = self(x_orig)
        adv_pattern = x_adv - x_orig

        if idx == 1:
            x_orig_cpu = x_orig.detach().cpu()
            x_adv_cpu = x_adv.detach().cpu()
            log_images(self.logger, "val_ims_orig", self._ims2grid(x_orig_cpu))
            log_images(self.logger, "val_ims_adv", self._ims2grid(x_adv_cpu))
            log_images(self.logger, "val_ims_diff", self._ims2grid(torch.abs(x_orig_cpu - x_adv_cpu)))

        monitor = self._get_monitor()
        log_dict = {}
        for alpha in self._MONITOR_ALPHAS:
            x = x_orig + alpha * adv_pattern
            x = torchvision.transforms.functional.normalize(x, mean=_IMAGENET_MEAN, std=_IMAGENET_STD)
            y_pred = monitor(x)
            key = "surrogate_val_top1" if alpha == 1.0 else f"surrogate_val_top1_{alpha}*adv"
            log_dict[key] = metrics.top1(y_pred, y_lbl)

        self._log_dict_if_attached(log_dict)
