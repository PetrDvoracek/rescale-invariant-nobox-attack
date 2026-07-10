"""The distortion bank and ``DSAugmentFactor`` dataset for F_Q (paper §4.2.1, Fig. 6).

This is the data side of the F_Q pipeline: it produces ``(original, augmented, label)``
triples where ``label = 1 - d / N_DISTORTIONS`` quantifies how heavily an image was
distorted (clean → 1.0, the Fig. 6 convention; the §4.2.1 text writes the monotone
reflection ``d / 16``). It is kept apart from :mod:`rina.quality` (the model) so the
distortion bank lives in one place.

The sampler draws a count ``d ~ U{0..16}``, picks ``d`` distinct augmentations uniformly,
and applies all of them (the paper's ``d ~ U(0, 16)``). The bank is reproduced from the
original code; editing it would silently rescale every label, so ``N_DISTORTIONS`` is pinned
and guarded by an assert.
"""

from __future__ import annotations

import glob
import random
from typing import Final

import albumentations as A
import cv2
import numpy as np
import torch
import tqdm

#: The paper's distortion-pipeline size (§4.2.1). The F_Q label divisor; pinning it here
#: (rather than ``len(augs_all)``) locks the convention against edits to the bank.
N_DISTORTIONS: Final[int] = 16


def _build_augs_all() -> list[A.BasicTransform]:
    """Build the 16-transform distortion bank (each ``p=1.0``).

    Rebuilt per call because the ``CoarseDropout`` fill draws fresh random colors at build
    time, so a cached instance would freeze them.

    Returns
    -------
    list of albumentations.BasicTransform
        The 16 distortion transforms, in canonical order.
    """
    return [
        A.ShiftScaleRotate(p=1.0, shift_limit=0.2, scale_limit=0, rotate_limit=0, border_mode=cv2.BORDER_CONSTANT),
        A.ShiftScaleRotate(p=1.0, shift_limit=0.0, scale_limit=1.0, rotate_limit=0, border_mode=cv2.BORDER_CONSTANT),
        A.ShiftScaleRotate(p=1.0, shift_limit=0.0, scale_limit=0.0, rotate_limit=360, border_mode=cv2.BORDER_CONSTANT),
        A.CoarseDropout(
            p=1.0,
            max_holes=16, max_height=64, max_width=64,
            min_holes=1, min_height=8, min_width=8,
            fill_value=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
        ),
        A.RandomToneCurve(p=1.0),
        A.Emboss(p=1.0),
        A.ChannelShuffle(p=1.0),
        A.Perspective(p=1.0),
        A.Blur(p=1.0, blur_limit=(11, 11)),
        A.Sharpen(p=1.0),
        A.HueSaturationValue(p=1.0, hue_shift_limit=(-30, 30), sat_shift_limit=(0, 0), val_shift_limit=(0, 0)),
        A.HueSaturationValue(p=1.0, hue_shift_limit=(0, 0), sat_shift_limit=(-40, 40), val_shift_limit=(0, 0)),
        A.HueSaturationValue(p=1.0, hue_shift_limit=(0, 0), sat_shift_limit=(0, 0), val_shift_limit=(-40, 40)),
        A.ImageCompression(p=1.0, quality_lower=50, quality_upper=80, compression_type=0),
        A.GaussNoise(p=1.0, var_limit=(10.0, 50.0), per_channel=True, mean=0.0),
        A.Transpose(p=1.0),
    ]


# The label ``1 - d / N_DISTORTIONS`` only holds if the bank is exactly N_DISTORTIONS long.
assert len(_build_augs_all()) == N_DISTORTIONS


def make_worker_init_fn(seed: int):
    """Build a ``worker_init_fn`` seeding each worker from ``(seed, worker_id)``.

    Derives the per-worker seed explicitly (ignoring Lightning's base seed) to give each
    worker a deterministic, disjoint RNG stream for ``random``, numpy, and albumentations.

    Parameters
    ----------
    seed : int
        The base seed (e.g. ``QualityConfig.seed``).

    Returns
    -------
    callable
        A function ``fn(worker_id: int) -> None`` for ``DataLoader(worker_init_fn=...)``.
    """

    def _init(worker_id: int) -> None:
        worker_seed = (int(seed) + worker_id) % (2**32)
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    return _init


class DSAugmentFactor(torch.utils.data.Dataset):
    """F_Q training set: ``(orig, augmented, label)`` distortion-factor triples (paper §4.2.1).

    Loads a flat image directory, drops images smaller than 224 in either dimension, and
    optionally repeats the path list ``enlarge`` times to lengthen an epoch. Each item is a
    random 224x224 crop, a distortion-augmented copy of it, and the label
    ``1 - d / N_DISTORTIONS``.

    Parameters
    ----------
    root : str
        Directory of source images (a flat dir, e.g. COCO ``test2017``).
    enlarge : int, optional
        Repeat the size-filtered path list this many times.
    seed : int, optional
        Recorded for reference; the per-sample augmentation RNG is seeded by
        :func:`make_worker_init_fn` at the DataLoader level, not here.
    """

    def __init__(self, root, enlarge=None, seed=42):
        self.seed = seed
        self.im_paths = []
        for path in tqdm.tqdm(glob.glob(f"{root}/*"), desc="loading images"):
            im = cv2.imread(path)  # shape only here; no BGR->RGB needed
            if im.shape[0] >= 224 and im.shape[1] >= 224:
                self.im_paths.append(path)
        if enlarge is not None:
            self.im_paths = self.im_paths * enlarge

    def __len__(self):
        return len(self.im_paths)

    def __getitem__(self, idx):
        """Return ``(im_orig, im_aug, y)``: two float32 CHW tensors in ``[0, 1]`` and the label."""
        im_path = self.im_paths[idx % len(self.im_paths)]
        im_orig = cv2.cvtColor(cv2.imread(im_path), cv2.COLOR_BGR2RGB)
        im_orig = self.random_crop(im_orig)
        im_aug, y = self.augment_img(im_orig)

        im_orig = (torch.from_numpy(im_orig).swapaxes(0, -1) / 255).to(torch.float32)
        im_aug = (torch.from_numpy(im_aug).swapaxes(0, -1) / 255).to(torch.float32)
        return im_orig, im_aug, y

    @staticmethod
    def random_crop(img):
        """Return a random 224x224 crop of ``img`` (HWC, at least 224x224)."""
        t = A.Compose([A.RandomCrop(always_apply=True, p=1.0, height=224, width=224)])
        return t(image=img)["image"]

    @staticmethod
    def augment_img(img):
        """Apply a distortion subset and return ``(augmented_float_array, label)`` (paper §4.2.1).

        Draws ``d ~ U{0..16}``, applies ``d`` distinct augmentations, and labels the result
        ``1 - d / N_DISTORTIONS`` (clean → 1.0).

        Parameters
        ----------
        img : numpy.ndarray
            Source crop, HWC (cast to ``uint8`` internally).

        Returns
        -------
        tuple of (numpy.ndarray, float)
            The augmented image as a ``float`` array and the scalar distortion-factor label.
        """
        img = img.astype("uint8")
        augs_all = _build_augs_all()
        d = random.randint(0, N_DISTORTIONS)
        chosen = [augs_all[i] for i in random.sample(range(len(augs_all)), d)]
        out = A.Compose(chosen)(image=img)["image"]
        return out.astype("float"), 1 - len(chosen) / N_DISTORTIONS
