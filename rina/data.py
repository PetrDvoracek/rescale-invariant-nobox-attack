"""F_A data, tiling, and corrupted↔original pairing (paper §5.1).

Three concerns:

1. **Pure-numpy tiling** (:func:`reflect_pad_to_multiple`, :func:`partition`,
   :func:`reassemble`) — the resolution-invariance machinery: reflect-pad up to a multiple
   of the tile, split into fixed 224x224 tiles, reassemble. These import no torch model, so
   they stay testable on CPU. The import rule is one-way: :mod:`rina.inference` imports this
   module, never the reverse.
2. The **F_A training dataset** :class:`DSAttack` (random 224 crops of COCO test2017).
3. **Per-dataset pairing** (:func:`adversarial_name` forward, :func:`original_path` inverse)
   by pathlib *component* surgery — never ``str.replace`` on the whole path, so a substring
   inside a filename is never rewritten.
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Final, Literal

import cv2
import numpy as np
import torch
import tqdm

__all__: Final = [
    "reflect_pad_to_multiple",
    "partition",
    "reassemble",
    "DSAttack",
    "adversarial_dir",
    "adversarial_name",
    "original_path",
]

# The source path component each dataset's adversarial folder is named after. For in1k the
# rewritten component is the literal "val", not the dataset-root basename.
_BASE_COMPONENT: Final[dict[str, str]] = {
    "in1k": "val",
    "coco": "test2017",
    "cityscapes": "val",
}

Dataset: Final = Literal["in1k", "coco", "cityscapes"]


# --------------------------------------------------------------------------- #
# Pure-numpy tiling helpers (no torch, no model import).
# --------------------------------------------------------------------------- #
def reflect_pad_to_multiple(
    img: np.ndarray, tile: int, mode: str = "reflect"
) -> tuple[np.ndarray, tuple[int, int]]:
    """Pad an image up to a multiple of ``tile`` in H and W.

    Uses ``(-h) % tile`` so an exact multiple gets no extra band. Reflect cannot pad by more
    than ``dim - 1``, which sub-tile dimensions can need, so it is clamped to that limit and
    the remainder edge-filled; other modes pad directly. The U-Net's receptive field reaches
    into the pad, so the fill genuinely influences the diff on border pixels (the paper uses
    reflect padding to avoid border artefacts, §5.1 / Fig. 8).

    Parameters
    ----------
    img : np.ndarray
        Image of shape ``(H, W, C)``.
    tile : int
        Tile size; output dims are padded up to multiples of this.
    mode : str, optional
        The numpy pad mode (default ``"reflect"``, the paper's choice).

    Returns
    -------
    padded : np.ndarray
        Padded array with both spatial dims exact multiples of ``tile``.
    orig_hw : tuple[int, int]
        The original ``(H, W)``, for cropping back in :func:`reassemble`.
    """
    h, w = img.shape[0], img.shape[1]
    h2add = (-h) % tile
    w2add = (-w) % tile

    if mode != "reflect":
        return np.pad(img, ((0, h2add), (0, w2add), (0, 0)), mode=mode), (h, w)

    # Reflect cannot exceed dim - 1; edge-fill whatever it leaves uncovered (sub-tile dims).
    refl_h = min(h2add, h - 1)
    refl_w = min(w2add, w - 1)
    out = np.pad(img, ((0, refl_h), (0, refl_w), (0, 0)), mode="reflect")
    edge_h = h2add - refl_h
    edge_w = w2add - refl_w
    if edge_h or edge_w:
        out = np.pad(out, ((0, edge_h), (0, edge_w), (0, 0)), mode="edge")
    return out, (h, w)


def partition(img: np.ndarray, tile: int) -> np.ndarray:
    """Split a padded image into a stack of square tiles.

    Parameters
    ----------
    img : np.ndarray
        Padded image ``(H, W, C)`` with ``H`` and ``W`` exact multiples of ``tile``.
    tile : int
        Tile size.

    Returns
    -------
    np.ndarray
        Tiles of shape ``(N, tile, tile, C)``, row-major: the whole top row left-to-right,
        then the next row down.
    """
    h, w, c = img.shape
    nh, nw = h // tile, w // tile
    grid = img.reshape(nh, tile, nw, tile, c).transpose(0, 2, 1, 3, 4)
    return grid.reshape(nh * nw, tile, tile, c)


def reassemble(
    tiles: np.ndarray,
    padded_hw: tuple[int, int],
    orig_hw: tuple[int, int],
) -> np.ndarray:
    """Inverse of :func:`partition`: stitch the tiles and crop the padding back off.

    Recovers the original unpadded region bit-exactly, for any fill mode.

    Parameters
    ----------
    tiles : np.ndarray
        Tiles of shape ``(N, tile, tile, C)`` as produced by :func:`partition`.
    padded_hw : tuple[int, int]
        The padded ``(H', W')`` the tiles were partitioned from.
    orig_hw : tuple[int, int]
        The original ``(H, W)`` to crop back to.

    Returns
    -------
    np.ndarray
        Array of shape ``(orig_h, orig_w, C)``.
    """
    ph, pw = padded_hw
    oh, ow = orig_hw
    _, tile, _, c = tiles.shape
    nh, nw = ph // tile, pw // tile
    grid = tiles.reshape(nh, nw, tile, tile, c).transpose(0, 2, 1, 3, 4)
    canvas = grid.reshape(ph, pw, c)
    return canvas[:oh, :ow, :]


# --------------------------------------------------------------------------- #
# F_A training dataset (torch + albumentations allowed here).
# --------------------------------------------------------------------------- #
class DSAttack(torch.utils.data.Dataset):
    """F_A training set: in-memory RGB images, optionally augmented per fetch.

    Loads every file under ``root`` (BGR→RGB), drops images smaller than ``resolution`` in
    either dimension, and optionally repeats the list ``enlarge`` times.

    Parameters
    ----------
    root : str
        Directory of images (e.g. COCO ``test2017``).
    resolution : int
        Minimum side length; smaller images are skipped.
    augment : optional
        An albumentations transform called as ``augment(image=im)["image"]``.
    enlarge : int, optional
        Repeat the loaded list this many times.
    """

    def __init__(self, root, resolution, augment=None, enlarge=None):
        self.augment = augment
        self.images = []
        for path in tqdm.tqdm(glob.glob(f"{root}/*"), desc="loading images"):
            im = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            if im.shape[0] < resolution or im.shape[1] < resolution:
                continue
            self.images.append(im)
        if enlarge is not None:
            self.images = self.images * enlarge

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        im = self.images[idx]
        if self.augment is not None:
            im = self.augment(image=im)["image"]
        return im


# --------------------------------------------------------------------------- #
# Per-dataset corrupted↔original pairing (component surgery, not str.replace).
# --------------------------------------------------------------------------- #
def _adversarial_component(base: str, alpha: float, tag: str) -> str:
    """The adversarial folder name: ``<base>_adversarial-<alpha>_tag-<tag>``."""
    return f"{base}_adversarial-{alpha}_tag-{tag}"


def adversarial_dir(dataset_root: Path, dataset: Dataset, alpha: float, tag: str) -> Path:
    """The adversarial folder ``attack`` writes for one ``alpha`` (sibling of the input split).

    For in1k a root containing ``val/`` gives ``<root>/val_adversarial-<alpha>_tag-<tag>``.
    ``attack`` and ``evaluate`` both derive their folder through this one function from the
    same ``dataset_root``, so produced and evaluated folders cannot drift apart.

    Parameters
    ----------
    dataset_root : Path
        The dataset root containing the input split.
    dataset : {"in1k", "coco", "cityscapes"}
        Selects the base component to swap.
    alpha : float
        Blend strength; part of the folder name.
    tag : str
        Free-form run label; part of the folder name.

    Returns
    -------
    Path
        ``<dataset_root>/<base>_adversarial-<alpha>_tag-<tag>``.
    """
    return Path(dataset_root) / _adversarial_component(_BASE_COMPONENT[dataset], alpha, tag)


def adversarial_name(original_path: Path, dataset: Dataset, alpha: float, tag: str) -> Path:
    """Map an original image path to its adversarial path.

    The single path *component* equal to the dataset base (``val`` for in1k/cityscapes,
    ``test2017`` for coco) becomes ``<base>_adversarial-<alpha>_tag-<tag>``; the
    ``<wnid>/<file>`` tail is untouched.

    Parameters
    ----------
    original_path : Path
        Path to the original image.
    dataset : {"in1k", "coco", "cityscapes"}
        Selects which component is swapped.
    alpha : float
        Blend strength; part of the folder name.
    tag : str
        Free-form run label; part of the folder name.

    Returns
    -------
    Path
        The corresponding adversarial image path.
    """
    base = _BASE_COMPONENT[dataset]
    return _swap_component(Path(original_path), base, _adversarial_component(base, alpha, tag), dataset)


def original_path(corrupted_path: Path, dataset: Dataset, alpha: float, tag: str) -> Path:
    """Inverse of :func:`adversarial_name`: map an adversarial path back to its original.

    Parameters
    ----------
    corrupted_path : Path
        Path to the adversarial image.
    dataset : {"in1k", "coco", "cityscapes"}
        Selects which component is swapped.
    alpha : float
        Blend strength used when the folder was created.
    tag : str
        Run label used when the folder was created.

    Returns
    -------
    Path
        The corresponding original image path.
    """
    base = _BASE_COMPONENT[dataset]
    return _swap_component(Path(corrupted_path), _adversarial_component(base, alpha, tag), base, dataset)


def _swap_component(path: Path, src: str, dst: str, dataset: Dataset) -> Path:
    """Replace the single path *component* equal to ``src`` with ``dst``.

    Searches from the right but never touches the final filename, so the swap only ever lands
    on a directory component (and a filename containing ``src`` as a substring is left alone).

    Raises
    ------
    ValueError
        If no component exactly equals ``src``.
    """
    parts = list(path.parts)
    for i in range(len(parts) - 2, -1, -1):
        if parts[i] == src:
            parts[i] = dst
            return Path(*parts)
    raise ValueError(f"no path component {src!r} in {str(path)!r} for dataset {dataset!r}")
