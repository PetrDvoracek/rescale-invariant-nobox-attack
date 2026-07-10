"""F_N — the no-box target classifiers (paper §4.4; results in §6).

F_N is the bank of off-the-shelf timm pretrained ImageNet-1k classifiers the attacker is
evaluated against offline. They are never seen during training — that is what "no-box"
means: the attacker is trained against the frozen F_Q quality estimator with no query access
to any classifier, and only *monitored* against a frozen resnet18 proxy (:mod:`rina.attacker`,
logged as ``surrogate_val_top1``). F_N is the held-out victim set scored offline, producing
the paper's Table 1 / Fig. 9 numbers. It is the only one of the four named networks with no
trained-by-us artefact — this module is data plus a thin validation loop, not a model we own.

The reported accuracies depend on timm's per-model preprocessing (crop_pct, interpolation,
input_size, mean/std) and AMP, which a hand-rolled loop mis-scores by ~0.5–2% top-1 — the
same order as the paper's headline. So the vendored ``timm_validate_imagenet.py`` stays the
authoritative validator: :func:`run_validation` shells out to it, and that script imports the
:data:`VICTIMS` / :data:`VICTIMS_SMOKE` lists below to choose which classifiers to score.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Final

# The F_N model set, mirroring the hardcoded list in timm_validate_imagenet.py's main(),
# grouped by family. Entries are timm model-name patterns: the efficientnet wildcard is
# expanded by timm's list_models (which drops the in21k/in22k variants), the rest are
# concrete tagged weights.
VICTIMS: Final[list[str]] = [
    # ViT
    "vit_tiny_patch16_224.augreg_in21k_ft_in1k",
    "vit_small_patch16_224.augreg_in21k_ft_in1k",
    "vit_large_patch16_224.augreg_in21k_ft_in1k",
    "vit_base_patch16_224.augreg_in21k_ft_in1k",
    "vit_base_patch16_224.augreg2_in21k_ft_in1k",
    # DeiT
    "deit_base_patch16_224.fb_in1k",
    "deit_small_patch16_224.fb_in1k",
    "deit_tiny_patch16_224.fb_in1k",
    # Swin
    "swin_base_patch4_window7_224.ms_in1k",
    "swin_large_patch4_window7_224.ms_in22k",
    "swin_small_patch4_window7_224.ms_in1k",
    "swin_tiny_patch4_window7_224.ms_in1k",
    # ResNet
    "resnet18.a1_in1k",
    "resnet34.a1_in1k",
    "resnet50.a1_in1k",
    "resnet101.a1_in1k",
    # VGG
    "vgg11.tv_in1k",
    "vgg13.tv_in1k",
    "vgg16.tv_in1k",
    "vgg19.tv_in1k",
    # EfficientNet (wildcard; expanded by timm's list_models)
    "tf_efficientnet_b*.in1k",
]

# A single tiny CPU-friendly victim for the smoke test. The bare name keeps it cheap and
# weight-tag-agnostic (the tagged resnet18.a1_in1k is already in VICTIMS).
VICTIMS_SMOKE: Final[list[str]] = ["resnet18"]

# The vendored validator (repo root, sibling of the rina/ package).
_VALIDATOR_SCRIPT: Final[Path] = Path(__file__).resolve().parent.parent / "timm_validate_imagenet.py"


def _validate_smoke(data_dir: Path, results_file: Path, device: str = "cpu") -> Path:
    """Fast, offline, GPU-free wiring check for the pipeline smoke test.

    Loads a single weight-free resnet18 (no download), scores the images in ``data_dir``
    (ImageNet ``<class>/<file>`` layout, folder index as target — accuracy is meaningless,
    this checks wiring), and writes the same 9-field CSV the vendored validator emits. The
    authoritative, paper-number path is the vendored validator (``smoke=False``).

    Parameters
    ----------
    data_dir : Path
        Adversarial image folder, in ``<class>/<file>`` layout.
    results_file : Path
        Destination CSV path.
    device : str, optional
        Torch device (default ``"cpu"``).

    Returns
    -------
    Path
        ``results_file``.
    """
    import csv
    import glob

    import cv2
    import torch
    import torchvision

    from rina.metrics import top1, top5

    paths = sorted(glob.glob(str(Path(data_dir) / "*" / "*")))
    if not paths:
        raise FileNotFoundError(f"no images under {data_dir}")
    wnids = sorted({Path(p).parent.name for p in paths})
    label_of = {w: i for i, w in enumerate(wnids)}
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    model = torchvision.models.resnet18(weights=None).eval().to(device)
    n_classes = 1000

    imgs, targets = [], []
    for p in paths:
        im = cv2.imread(p)
        if im is None:
            raise IOError(f"cannot read {p}")
        im = cv2.cvtColor(cv2.resize(im, (224, 224)), cv2.COLOR_BGR2RGB)
        imgs.append(torch.from_numpy(im).permute(2, 0, 1).float() / 255.0)
        targets.append(label_of[Path(p).parent.name])

    x = ((torch.stack(imgs) - mean) / std).to(device)
    y = torch.tensor(targets, device=device)
    with torch.no_grad():
        logits = model(x)
    # Pad logits to 1000 classes so the metrics' num_classes=1000 helper is satisfied.
    if logits.shape[1] < n_classes:
        pad = torch.full((logits.shape[0], n_classes - logits.shape[1]), -1e4, device=device)
        logits = torch.cat([logits, pad], dim=1)
    t1 = float(top1(logits, y)) * 100
    t5 = float(top5(logits, y)) * 100

    results = {
        "model": "resnet18",
        "top1": round(t1, 4),
        "top1_err": round(100 - t1, 4),
        "top5": round(t5, 4),
        "top5_err": round(100 - t5, 4),
        "param_count": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
        "img_size": 224,
        "crop_pct": 0.875,
        "interpolation": "bilinear",
    }
    with open(results_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results.keys()))
        writer.writeheader()
        writer.writerow(results)
    return Path(results_file)


def run_validation(
    data_dir: Path,
    results_file: Path,
    device: str = "cpu",
    batch_size: int = 64,
    amp: bool = False,
    torchcompile: bool = False,
    smoke: bool = False,
) -> Path:
    """Score an adversarial folder and return the CSV path.

    A thin wrapper: it invokes the preserved ``timm_validate_imagenet.py`` once as a
    subprocess and does not reimplement timm's per-model pipeline. That script imports the
    victim list from here (:data:`VICTIMS`, or :data:`VICTIMS_SMOKE` when
    ``RINA_VICTIMS_SMOKE=1``).

    Parameters
    ----------
    data_dir : Path
        The adversarial image folder to score, in ImageNet ``val/<class>/`` layout.
    results_file : Path
        Destination CSV path (passed as ``--results-file``).
    device : str, optional
        Accelerator passed as ``--device`` (default ``"cpu"``).
    batch_size : int, optional
        Per-step batch size passed as ``--batch-size`` (default 64).
    amp : bool, optional
        Pass ``--amp`` (mixed-precision inference) when True.
    torchcompile : bool, optional
        Pass ``--torchcompile`` when True.
    smoke : bool, optional
        Take the offline GPU-free resnet18 path (:func:`_validate_smoke`) instead.

    Returns
    -------
    Path
        ``results_file`` (the CSV the validator wrote).
    """
    if smoke:
        return _validate_smoke(Path(data_dir), Path(results_file), device=device)

    cmd: list[str] = [
        sys.executable,
        str(_VALIDATOR_SCRIPT),
        # main() ignores --model when --checkpoint is not a dir and validates its hardcoded
        # list; the placeholder is for parity with eval_imagenet.sh.
        "--model", "hardcoded_in_script",
        "--data-dir", str(data_dir),
        "--results-file", str(results_file),
        "--device", device,
        "--batch-size", str(batch_size),
    ]
    if amp:
        cmd.append("--amp")
    if torchcompile:
        cmd.append("--torchcompile")

    subprocess.run(cmd, check=True)
    return results_file
