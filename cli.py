"""RINA command-line interface — a Typer app with five commands:

    train-quality   train F_Q (the FMQ quality estimator)        -> F_Q .ckpt
    train-attacker  train F_A (the U-Net attacker)               -> F_A .ckpt   (needs F_Q)
    attack          generate adversarial folders (Eq.(9) blend)  -> <base>_adversarial-<a>_tag-<t>/
    evaluate        classifier accuracy on those folders (timm victims = F_N)
    measure-ssim    SSIM between an adversarial folder and the originals

Config comes from the typed dataclasses in :mod:`rina.config`; individual ``--flag`` overrides
compose on top of the paper defaults. Reproducing the paper's Table-1 numbers means running the
whole pipeline in order; the defaults reproduce the *method*, not a turnkey path to the numbers.
Heavy imports are deferred into each command so ``--help`` stays fast.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Optional

import typer

from rina.config import AttackConfig, QualityConfig, paper_config

app = typer.Typer(add_completion=False, help=__doc__)


def _override(cfg, **kw):
    """Return a copy of dataclass ``cfg`` with the non-None overrides applied."""
    changes = {k: v for k, v in kw.items() if v is not None}
    return dataclasses.replace(cfg, **changes) if changes else cfg


def _make_logger(kind: str, project: str, name: str, group: str):
    """Map the ``logger`` config (wandb|csv|none) to a Lightning logger, or ``False`` for none."""
    import lightning as L

    if kind == "wandb":
        return L.pytorch.loggers.WandbLogger(project=project, name=name, group=group, log_model=False)
    if kind == "csv":
        return L.pytorch.loggers.CSVLogger(save_dir="./log", name=name)
    return False


def _seed(cfg) -> None:
    import lightning as L

    L.pytorch.seed_everything(cfg.seed, workers=True)


@app.command("train-quality")
def train_quality(
    coco_test2017: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True, help="COCO test2017 image dir"),
    models_dir: Path = typer.Option(Path("./models"), help="checkpoint output dir"),
    backbone: Optional[str] = typer.Option(None, help="timm estimator backbone (overrides the paper default)"),
    pretrained: Optional[bool] = typer.Option(None, help="ImageNet-pretrained init"),
    epochs: Optional[int] = typer.Option(None),
    enlarge: Optional[int] = typer.Option(None),
    batch_size: int = typer.Option(16),
    num_workers: Optional[int] = typer.Option(None),
    logger: Optional[str] = typer.Option(None, help="wandb|csv|none"),
    seed: Optional[int] = typer.Option(None),
    deterministic: Optional[bool] = typer.Option(None),
    devices: int = typer.Option(1),
    accelerator: str = typer.Option("auto"),
):
    """Train F_Q (the FMQ quality estimator). Prerequisite for ``train-attacker``."""
    import lightning as L
    import torch

    from rina.augment import DSAugmentFactor, make_worker_init_fn
    from rina.quality import build_quality

    qcfg, _, _ = paper_config()
    qcfg = _override(
        qcfg,
        estimator_backbone=backbone,
        quality_pretrained=pretrained,
        epochs=epochs,
        enlarge=enlarge,
        num_workers=num_workers,
        logger=logger,
        seed=seed,
        deterministic=deterministic,
    )
    _seed(qcfg)
    models_dir.mkdir(parents=True, exist_ok=True)

    ds = DSAugmentFactor(root=str(coco_test2017), enlarge=qcfg.enlarge, seed=qcfg.seed)
    gen = torch.Generator().manual_seed(qcfg.seed)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, num_workers=qcfg.num_workers, pin_memory=True,
        shuffle=True, generator=gen, worker_init_fn=make_worker_init_fn(qcfg.seed),
    )

    model = build_quality(qcfg)
    callbacks = [
        L.pytorch.callbacks.ModelCheckpoint(
            dirpath=str(models_dir), filename=f"fq_{qcfg.estimator_backbone}_{{epoch}}", save_last=True
        )
    ]
    fq_logger = _make_logger(qcfg.logger, "adversarial", f"fq_{qcfg.estimator_backbone}", "augsim")
    if fq_logger is not False:  # LearningRateMonitor needs a logger
        callbacks.append(L.pytorch.callbacks.LearningRateMonitor())

    trainer = L.Trainer(
        max_epochs=qcfg.epochs, accelerator=accelerator, devices=devices,
        logger=fq_logger, callbacks=callbacks,
        gradient_clip_val=1.0, precision="32", log_every_n_steps=10,
        deterministic=qcfg.deterministic,
    )
    trainer.fit(model, loader)
    out = models_dir / "fq.ckpt"
    trainer.save_checkpoint(str(out))
    typer.echo(f"F_Q checkpoint: {out}")


@app.command("train-attacker")
def train_attacker(
    coco_test2017: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True),
    quality_ckpt: Optional[Path] = typer.Option(None, help="trained F_Q checkpoint (required)"),
    imagenet_val: Optional[Path] = typer.Option(None, exists=True, file_okay=False, dir_okay=True, help="ImageNet val root (optional surrogate monitor)"),
    models_dir: Path = typer.Option(Path("./models")),
    backbone: Optional[str] = typer.Option(None, help="F_Q backbone to reconstruct; must match the trained --quality-ckpt (default: paper backbone)"),
    max_steps: Optional[int] = typer.Option(None),
    lr: Optional[float] = typer.Option(None),
    batch_size: Optional[int] = typer.Option(None),
    num_workers: Optional[int] = typer.Option(None),
    logger: Optional[str] = typer.Option(None),
    seed: Optional[int] = typer.Option(None),
    deterministic: Optional[bool] = typer.Option(None),
    devices: int = typer.Option(1),
    accelerator: str = typer.Option("auto"),
    precision: str = typer.Option("16-mixed"),
):
    """Train F_A (the U-Net attacker). Refuses to start without an existing ``--quality-ckpt``."""
    import lightning as L
    import torch

    from rina.attacker import TraineeAttacker, build_attacker_unet
    from rina.data import DSAttack
    from rina.quality import Trainee, build_modelsim, build_quality_backbone

    # No silent random-weights F_Q: train it first.
    if quality_ckpt is None or not Path(quality_ckpt).exists():
        raise typer.BadParameter(
            "F_Q checkpoint required. Train it first: "
            "`rina train-quality --backbone tf_mobilenetv3_large_100 --pretrained` "
            "(see REPRODUCE.md step 1)."
        )

    _, acfg, _ = paper_config()
    acfg = _override(
        acfg, max_steps=max_steps, lr=lr, batch_size=batch_size, num_workers=num_workers,
        logger=logger, seed=seed, deterministic=deterministic,
    )
    _seed(acfg)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Reconstruct + load the frozen F_Q. Its architecture must match the trained --quality-ckpt;
    # pretrained init is skipped because the checkpoint weights overwrite it anyway.
    qcfg = QualityConfig(
        estimator_backbone=backbone or QualityConfig().estimator_backbone,
        quality_pretrained=False,
    )
    fq = Trainee.load_from_checkpoint(
        str(quality_ckpt), map_location="cpu",
        model=build_quality_backbone(qcfg), model_sim=build_modelsim(),
    )
    typer.echo(f"loaded F_Q ({qcfg.estimator_backbone}) from {quality_ckpt}")

    # F_A training augmentation (geometric + photometric).
    import albumentations as A
    import cv2
    from albumentations.pytorch import ToTensorV2

    from rina.augment import make_worker_init_fn

    res = 224
    aug = A.Compose([
        A.ShiftScaleRotate(p=0.3, shift_limit=0.0, scale_limit=0.3, rotate_limit=0, border_mode=cv2.BORDER_CONSTANT),
        A.RandomCrop(always_apply=True, p=1.0, height=res, width=res),
        A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.Perspective(p=0.5),
        A.Blur(p=0.2, blur_limit=(3, 3)),
        A.PadIfNeeded(p=1.0, min_height=res, min_width=res),
        A.ToFloat(max_value=255), ToTensorV2(),
    ])
    ds = DSAttack(str(coco_test2017), resolution=res, augment=aug)
    gen = torch.Generator().manual_seed(acfg.seed)
    train_loader = torch.utils.data.DataLoader(
        ds, batch_size=acfg.batch_size, num_workers=acfg.num_workers, pin_memory=True,
        shuffle=True, generator=gen, worker_init_fn=make_worker_init_fn(acfg.seed),
    )

    val_loader = None
    if imagenet_val is not None:
        import torchvision
        import torchvision.transforms.functional as TF

        class _T(torch.nn.Module):
            def forward(self, img):
                img = TF.resize(img, [res], interpolation=TF.InterpolationMode.BILINEAR)
                img = TF.center_crop(img, [res])
                if not isinstance(img, torch.Tensor):
                    img = TF.pil_to_tensor(img)
                return TF.convert_image_dtype(img, torch.float)

        val_ds = torchvision.datasets.ImageNet(root=str(imagenet_val), split="val", transform=_T())
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=acfg.batch_size * 6, num_workers=acfg.num_workers, pin_memory=True, shuffle=False
        )

    unet = build_attacker_unet(acfg)
    model = TraineeAttacker(model_attacker=unet, model_augmentfactor=fq, cfg=acfg)
    # save_last + periodic recovery points: a long run must survive a late divergence (the
    # 16-mixed NaN blowup) without losing hours of compute.
    callbacks = [
        L.pytorch.callbacks.ModelCheckpoint(
            dirpath=str(models_dir), filename="fa_{step}", save_last=True,
            every_n_train_steps=50_000, save_top_k=-1,
        )
    ]
    fa_logger = _make_logger(acfg.logger, "adversarial", f"fa_{acfg.unet_backbone}", "attack")
    if fa_logger is not False:  # LearningRateMonitor needs a logger
        callbacks.append(L.pytorch.callbacks.LearningRateMonitor())

    trainer = L.Trainer(
        max_steps=acfg.max_steps, accelerator=accelerator, devices=devices,
        logger=fa_logger, callbacks=callbacks,
        gradient_clip_val=1.0, precision=precision, log_every_n_steps=10,
        num_sanity_val_steps=0, deterministic=acfg.deterministic,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    out = models_dir / "fa.ckpt"
    trainer.save_checkpoint(str(out))
    typer.echo(f"F_A checkpoint: {out}")


@app.command("attack")
def attack(
    dataset_root: Path = typer.Argument(..., exists=True),
    ckpt: Path = typer.Argument(..., exists=True, dir_okay=False),
    dataset: str = typer.Option("in1k", help="in1k|coco|cityscapes"),
    tag: str = typer.Option(""),
    device: str = typer.Option("cpu"),
    alphas: Optional[list[float]] = typer.Option(None, "--alpha", "-a", help="repeatable; defaults to AttackConfig.alphas"),
    tile: Optional[int] = typer.Option(None),
    compile: bool = typer.Option(False),
):
    """Generate adversarial folders. ``--alpha`` defaults to ``AttackConfig.alphas`` = (0.85, 1.0, 1.15)."""
    from rina.inference import attack_dataset

    kcfg = AttackConfig()
    use_alphas = tuple(alphas) if alphas else kcfg.alphas
    attack_dataset(
        ckpt=str(ckpt), dataset_root=str(dataset_root), dataset=dataset,
        alphas=use_alphas, tag=tag,
        tile=tile or kcfg.tile, device=device, compile=compile, pad_mode=kcfg.pad_mode,
    )
    typer.echo(f"attacked {dataset_root} for alphas {use_alphas}")


@app.command("evaluate")
def evaluate(
    dataset_root: Path = typer.Argument(..., exists=True, help="same dataset root passed to `attack`"),
    dataset: str = typer.Option("in1k", help="in1k|coco|cityscapes (must match `attack`)"),
    tag: str = typer.Option(""),
    alphas: Optional[list[float]] = typer.Option(None, "--alpha", "-a"),
    device: str = typer.Option("cpu"),
    batch_size: int = typer.Option(64),
    smoke: bool = typer.Option(False, help="use a single tiny victim (resnet18) for a fast CPU check"),
    amp: bool = typer.Option(False),
    torchcompile: bool = typer.Option(False),
):
    """Classifier accuracy on the adversarial folders (timm victims = F_N). One CSV per alpha.

    Derives each per-alpha folder from the same ``dataset_root`` + ``--dataset`` as ``attack``,
    via the identical :func:`rina.data.adversarial_dir`, so the produced and evaluated folders
    cannot drift apart.
    """
    from rina.data import adversarial_dir
    from rina.victims import run_validation

    kcfg = AttackConfig()
    use_alphas = tuple(alphas) if alphas else kcfg.alphas  # same source as `attack`
    for a in use_alphas:
        folder = adversarial_dir(dataset_root, dataset, a, tag)
        results = Path(f"./eval_adversarial-{a}_tag-{tag}.csv")
        run_validation(
            data_dir=str(folder), results_file=str(results),
            device=device, batch_size=batch_size, amp=amp, torchcompile=torchcompile, smoke=smoke,
        )
        typer.echo(f"alpha={a} -> {results}")


@app.command("measure-ssim")
def measure_ssim(
    corrupted_root: Path = typer.Argument(..., exists=True, file_okay=False),
    dataset: str = typer.Option("in1k"),
    alpha: float = typer.Option(1.0),
    tag: str = typer.Option(""),
    wild: str = typer.Option("**/*.JPEG"),
    subset: Optional[int] = typer.Option(1000, help="fixed seeded subset size; None for all"),
    seed: int = typer.Option(42),
):
    """SSIM between an adversarial folder and the paired originals (pairing via ``data.original_path``)."""
    import glob
    import random

    import cv2
    import numpy as np

    from rina.data import original_path
    from rina.metrics import ssim_gray

    corrupted = sorted(glob.glob(str(corrupted_root / wild), recursive=True))
    if subset is not None and subset < len(corrupted):
        rng = random.Random(seed)
        corrupted = sorted(rng.sample(corrupted, subset))  # fixed seeded subset

    vals = []
    for cp in corrupted:
        op = original_path(Path(cp), dataset, alpha, tag)
        if not op.exists():
            typer.echo(f"warning: original missing for {cp} -> {op}", err=True)
            continue
        a = cv2.imread(cp)
        b = cv2.imread(str(op))
        if a is None or b is None:
            raise IOError(f"cannot read pair {cp} / {op}")
        vals.append(ssim_gray(a, b))
    typer.echo(f"dataset_name SSIM\n{corrupted_root.name} {np.mean(vals):.4f}" if vals else "no pairs found")


if __name__ == "__main__":
    app()
