"""Typed configuration — the single source of truth for hyperparameters.

The dataclasses below carry the paper's training values as their field defaults
(:class:`QualityConfig` from §5.3, :class:`AttackerConfig` from §5.1, :class:`AttackConfig`
from the §5.1 inference blend). :func:`paper_config` assembles the canonical bundle; CLI
``--flag`` overrides compose on top for ablations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional


@dataclass
class Paths:
    """Dataset/output roots. Every field is machine-specific and required at the CLI.

    Kept separate from the model configs because required (no-default) fields cannot
    follow defaulted ones in a single dataclass.
    """

    coco_test2017: Path
    imagenet_val: Path
    models_dir: Path
    output_root: Path


@dataclass
class QualityConfig:
    """F_Q (FMQ quality-estimator) training config; defaults from paper §5.3."""

    estimator_backbone: str = "tf_mobilenetv3_large_100"
    quality_pretrained: bool = True
    epochs: int = 20
    lr: float = 1e-3
    loss: str = "l1"
    enlarge: int = 10
    num_workers: int = 6
    seed: int = 42
    deterministic: bool = False
    logger: Literal["wandb", "csv", "none"] = "wandb"


@dataclass
class AttackerConfig:
    """F_A (U-Net attacker) training config; defaults from paper §5.1.

    ``lr_T_0_steps`` is derived: it resolves to ``max_steps // 3`` so the cosine-restart
    period tracks ``max_steps`` (the paper specifies warm restarts but not the period). With
    ``lr_T_mult=2`` that gives exactly two restarts. An explicit value overrides the derivation.
    """

    unet_backbone: str = "se_resnet50"
    encoder_weights: Optional[str] = None
    beta: float = 15.0  # Eq.(8) weight on the adversarial term L_A
    train_augment: Literal["coco_geom_photo"] = "coco_geom_photo"
    lr: float = 1e-3
    max_steps: int = 387_000
    lr_T_0_steps: Optional[int] = None  # derived from max_steps in __post_init__
    lr_T_mult: int = 2
    lr_eta_min: float = 1e-6
    batch_size: int = 16
    num_workers: int = 8
    seed: int = 42
    deterministic: bool = False
    logger: Literal["wandb", "csv", "none"] = "wandb"

    def __post_init__(self) -> None:
        if self.lr_T_0_steps is None:
            self.lr_T_0_steps = self.max_steps // 3


@dataclass
class AttackConfig:
    """Offline dataset-attack config (paper §5.1 inference; the Eq.(9) blend).

    ``alphas`` defaults to the paper's ``(0.85, 1.0, 1.15)`` so ``attack`` emits exactly the
    folders ``evaluate`` expects; ``0.0`` (identity) and ``1.3`` (overshoot) stay available
    as explicit probes.
    """

    tile: int = 224
    alphas: tuple[float, ...] = (0.85, 1.0, 1.15)
    pad_mode: str = "reflect"  # paper §5.1 / Fig. 8: avoids border artefacts
    compile: bool = False


def paper_config() -> tuple[QualityConfig, AttackerConfig, AttackConfig]:
    """The paper-faithful configuration bundle (the repo's only configuration)."""
    return QualityConfig(), AttackerConfig(), AttackConfig()
