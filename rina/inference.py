"""F_A inference and the offline tiled dataset attack (paper §5.1).

The model side of the resolution-invariant path: :class:`AttackerInference` wraps a loaded
F_A U-Net, :func:`perturb_image` tiles an image and reassembles the full-resolution
perturbation, :func:`blend` applies the Eq.(9) blend, and :func:`attack_dataset` writes the
adversarial folders. The pure-numpy tiling helpers live in :mod:`rina.data` and are not
duplicated here; the import direction is one-way (this module imports ``rina.data`` and
``rina.attacker``, never the reverse).

Only **F_A attacker** checkpoints load here — quality-estimator (``Trainee``) checkpoints do
not. The pipeline stays in RGB end-to-end (F_A was trained on RGB): read BGR → convert to
RGB → tile / run F_A / diff / reassemble → blend → convert back to BGR → write. Adversarial
folders are created as siblings of the input split via :func:`rina.data.adversarial_name`.
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
import tqdm

from rina import data
from rina.attacker import TraineeAttacker

__all__ = ["AttackerInference", "perturb_image", "blend", "attack_dataset"]


class AttackerInference:
    """A NumPy-in / NumPy-out wrapper around a loaded F_A U-Net.

    ``__call__`` runs the model under ``no_grad`` on ``(N, tile, tile, C)`` tiles; the
    ``swapaxes`` calls bridge the channels-last NumPy convention and the channels-first layout
    the U-Net expects.

    Parameters
    ----------
    torch_model : torch.nn.Module
        A loaded F_A attacker; moved to ``device`` and set to eval here.
    device : str or torch.device
        Device to run on (e.g. ``"cpu"`` / ``"cuda"``).
    compile : bool, optional
        Whether to ``torch.compile(..., dynamic=True)`` the model. Default ``False``
        (unnecessary and flaky on CPU smoke runs).
    """

    def __init__(self, torch_model, device, compile: bool = False):
        # Move to device before eval/compile: new-format checkpoints rebuild the U-Net on CPU
        # and load weights in place, so without this the inputs (on device) and weights (on
        # CPU) would diverge on cuda.
        torch_model = torch_model.to(device)
        torch_model.eval()
        self.model = torch.compile(torch_model, dynamic=True) if compile else torch_model
        self.device = device

    def __call__(self, batch_np: np.ndarray) -> np.ndarray:
        """Run F_A on a stack of tiles.

        Parameters
        ----------
        batch_np : np.ndarray
            Tiles of shape ``(N, tile, tile, C)``, float in ``[0, 1]``, RGB.

        Returns
        -------
        np.ndarray
            The attacker output, same shape, channels-last.
        """
        batch = torch.from_numpy(batch_np.swapaxes(1, -1)).to(torch.float32).to(self.device)
        with torch.no_grad():
            out = self.model(batch).detach()
        return out.cpu().numpy().swapaxes(1, -1)


def perturb_image(
    model: AttackerInference,
    im_rgb: np.ndarray,
    tile: int,
    pad_mode: str = "reflect",
) -> np.ndarray:
    """Compute the full-resolution perturbation ``diff`` for one image.

    Pads to a multiple of ``tile``, tiles, runs ``model`` on the ``[0, 1]`` tiles, takes the
    per-tile ``out - x`` diff, reassembles, and crops back to the original size.

    Parameters
    ----------
    model : AttackerInference
        The wrapped F_A model.
    im_rgb : np.ndarray
        RGB ``uint8`` image of shape ``(H, W, C)``.
    tile : int
        Tile / F_A input size (224).
    pad_mode : str, optional
        The numpy pad mode passed to :func:`rina.data.reflect_pad_to_multiple` (default
        ``"reflect"``, the paper's choice).

    Returns
    -------
    np.ndarray
        The signed RGB float perturbation of shape ``(H, W, C)`` over the unpadded region.
    """
    padded, orig_hw = data.reflect_pad_to_multiple(im_rgb, tile, mode=pad_mode)
    padded_hw = (padded.shape[0], padded.shape[1])

    tiles = data.partition(padded, tile)
    batch = tiles.astype(np.float32) / 255.0
    diff_tiles = model(batch) - batch
    return data.reassemble(diff_tiles, padded_hw, orig_hw)


def blend(im_rgb_uint8: np.ndarray, canvas_diff: np.ndarray, alpha: float) -> np.ndarray:
    """Eq.(9) blend: ``uint8(clip(im/255 + alpha * canvas_diff, 0, 1) * 255)``.

    The clip is applied after the blend, so overshoot needs no special case.

    Parameters
    ----------
    im_rgb_uint8 : np.ndarray
        Original RGB ``uint8`` image, shape ``(H, W, C)``.
    canvas_diff : np.ndarray
        Signed float perturbation, same shape, from :func:`perturb_image`.
    alpha : float
        Strength: ``0`` reproduces the original (exact in float, before encoding), ``1`` the
        full learned perturbation, ``>1`` overshoot.

    Returns
    -------
    np.ndarray
        Adversarial RGB ``uint8`` image, shape ``(H, W, C)``.
    """
    adv = np.clip(im_rgb_uint8.astype(np.float32) / 255.0 + alpha * canvas_diff, 0.0, 1.0)
    return (adv * 255).astype(np.uint8)


def _get_paths(dataset_root: Path, dataset: str) -> list[str]:
    """Sorted input image paths for ``dataset`` under ``dataset_root``."""
    root = str(dataset_root)
    if dataset == "in1k":
        paths = glob.glob(f"{root}/val/**/*")
    elif dataset == "coco":
        paths = glob.glob(f"{root}/test2017/*")
    elif dataset == "cityscapes":
        paths = glob.glob(f"{root}/*/*")
    else:
        raise ValueError(f"unknown dataset {dataset!r}")
    return sorted(paths)


class _NoopFQ(torch.nn.Module):
    """Placeholder F_Q for inference-time loads — F_Q is never called at inference."""

    def forward(self, x, x_A):  # pragma: no cover
        return torch.zeros((x.shape[0], 1))


def _load_attacker_for_inference(ckpt, device):
    """Load an F_A attacker checkpoint for inference (old pickled or new ``ignore=`` format).

    Old checkpoints (pickled ``lib.*`` submodules) load directly via the ``lib`` shim. New
    checkpoints carry no module hparams, so the U-Net is rebuilt from the saved
    ``AttackerConfig`` and F_Q is a no-op placeholder. Only the ``model_attacker.*`` weights
    are loaded into the U-Net, and a key mismatch there is a hard error (inference depends on
    nothing else).
    """
    try:
        return TraineeAttacker.load_from_checkpoint(ckpt, map_location=device)
    except TypeError:
        from rina.attacker import build_attacker_unet
        from rina.config import AttackerConfig

        loaded = torch.load(ckpt, map_location=device)
        cfg = loaded.get("hyper_parameters", {}).get("cfg") or AttackerConfig()
        prefix = "model_attacker."
        unet_sd = {k[len(prefix):]: v for k, v in loaded["state_dict"].items() if k.startswith(prefix)}
        unet = build_attacker_unet(cfg)
        missing, unexpected = unet.load_state_dict(unet_sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"attacker U-Net weights do not match the checkpoint "
                f"(missing={list(missing)[:5]}..., unexpected={list(unexpected)[:5]}...); "
                f"check unet_backbone={cfg.unet_backbone!r}"
            )
        model = TraineeAttacker(model_attacker=unet, model_augmentfactor=_NoopFQ(), cfg=cfg)
        model.eval()
        return model


def attack_dataset(
    ckpt: str,
    dataset_root: str,
    dataset: str,
    alphas: Iterable[float],
    tag: str,
    tile: int = 224,
    device: str = "cpu",
    compile: bool = False,
    pad_mode: str = "reflect",
) -> None:
    """Run the offline tiled attack and write one adversarial folder per ``alpha``.

    Output folders are siblings of the input split, derived from each original path by
    :func:`rina.data.adversarial_name`.

    Parameters
    ----------
    ckpt : str
        Path to an **F_A attacker** checkpoint (``TraineeAttacker``).
    dataset_root : str
        Root holding the input split.
    dataset : {"in1k", "coco", "cityscapes"}
        Layout selector for globbing inputs and naming outputs.
    alphas : Iterable[float]
        Blend strengths; one folder is written per alpha.
    tag : str
        Run label baked into the output folder names.
    tile : int, optional
        Tile / F_A input size (default 224).
    device : str, optional
        Torch device (default ``"cpu"``).
    compile : bool, optional
        Whether to ``torch.compile`` F_A (default ``False``).
    pad_mode : str, optional
        Pad mode forwarded to :func:`perturb_image` (default ``"reflect"``).
    """
    alphas = list(alphas)
    trainee = _load_attacker_for_inference(ckpt, device)
    model = AttackerInference(trainee, device, compile=compile)

    for p in tqdm.tqdm(_get_paths(Path(dataset_root), dataset)):
        im = cv2.imread(p)
        if im is None:
            raise IOError(f"cannot read {p}")
        im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        canvas_diff = perturb_image(model, im_rgb, tile, pad_mode=pad_mode)

        for alpha in alphas:
            adv = blend(im_rgb, canvas_diff, alpha)
            out = data.adversarial_name(Path(p), dataset, alpha, tag)
            out.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(out), cv2.cvtColor(adv, cv2.COLOR_RGB2BGR)):
                raise IOError(f"failed to write {out}")
