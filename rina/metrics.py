"""Metric math: classifier accuracy and SSIM (paper Table 1).

Only the computations live here — no paths, no models, no dataset access. ``top1`` /
``top5`` are shared by the training-time monitor (:mod:`rina.attacker`) and the offline
victim evaluation (:mod:`rina.victims`); ``ssim_gray`` backs the image-similarity column
the paper reports alongside accuracy. Pairing corrupted images with their originals is
:mod:`rina.data`'s job, not this module's.
"""

from __future__ import annotations

from typing import Final

import cv2
import numpy as np
import torch
import torchmetrics.functional
from skimage.metrics import structural_similarity

NUM_CLASSES: Final[int] = 1000  # ImageNet-1k


def top1(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Top-1 multiclass accuracy.

    Parameters
    ----------
    logits : torch.Tensor
        Classifier outputs of shape ``(N, NUM_CLASSES)`` (logits or probabilities).
    target : torch.Tensor
        Integer labels of shape ``(N,)`` in ``[0, NUM_CLASSES)``.

    Returns
    -------
    torch.Tensor
        Scalar fraction of samples whose top-1 prediction is correct.
    """
    return torchmetrics.functional.accuracy(
        logits, target, num_classes=NUM_CLASSES, task="multiclass", top_k=1
    )


def top5(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Top-5 multiclass accuracy (companion to :func:`top1`, with ``top_k=5``).

    Parameters
    ----------
    logits : torch.Tensor
        Classifier outputs of shape ``(N, NUM_CLASSES)``.
    target : torch.Tensor
        Integer labels of shape ``(N,)`` in ``[0, NUM_CLASSES)``.

    Returns
    -------
    torch.Tensor
        Scalar fraction of samples whose label is among the 5 highest-scoring classes.
    """
    return torchmetrics.functional.accuracy(
        logits, target, num_classes=NUM_CLASSES, task="multiclass", top_k=5
    )


def ssim_gray(im_a: np.ndarray, im_b: np.ndarray) -> float:
    """Grayscale SSIM between two equally-shaped uint8 BGR images.

    The pair is assumed already matched and aligned. ``data_range`` is pinned to 255 so the
    score does not depend on the values that happen to be present in a given pair.

    Parameters
    ----------
    im_a, im_b : numpy.ndarray
        ``(H, W, 3)`` uint8 images in BGR channel order (as read by ``cv2.imread``).

    Returns
    -------
    float
        Mean structural similarity in ``[-1, 1]`` (``1.0`` for identical inputs).
    """
    gray_a = cv2.cvtColor(im_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(im_b, cv2.COLOR_BGR2GRAY)
    return float(structural_similarity(gray_a, gray_b, data_range=255))
