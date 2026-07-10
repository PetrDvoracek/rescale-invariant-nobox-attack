"""The Eq.(8) compound loss for the U-Net attacker F_A (paper §4.3, §5.4)::

    loss = max(L_S, beta * L_A)

``L_S`` is the MSE similarity (image-quality) term, ``L_A = mean(F_Q(x, x_A))`` is the
adversarial term, and ``beta`` scales the adversarial term (Eq.(8)).

All terms are computed on **[0, 1] float** tensors: the attacker output is a sigmoid and the
input is ``im / 255`` — not ``[0, 255]`` and not ImageNet-normalized.

F_Q is passed in as a callable, so this module has no model dependency and is unit-testable
on CPU with a stub.
"""

from __future__ import annotations

import torch
from torch import Tensor


def loss_similarity(x: Tensor, x_A: Tensor) -> Tensor:
    """Similarity term ``L_S`` — mean squared error over ``[0, 1]`` (paper §5.4).

    Parameters
    ----------
    x : torch.Tensor
        Original image, ``[0, 1]`` float (``im / 255``).
    x_A : torch.Tensor
        Attacker output, ``[0, 1]`` float (sigmoid).

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """
    return torch.mean((x - x_A) ** 2)


def loss_adversarial(x: Tensor, x_A: Tensor, fq) -> Tensor:
    """Adversarial term ``L_A = mean(F_Q(x, x_A))``.

    Parameters
    ----------
    x : torch.Tensor
        Original image, ``[0, 1]`` float.
    x_A : torch.Tensor
        Attacker output, ``[0, 1]`` float.
    fq : callable
        The frozen F_Q estimator, called as ``fq(orig, aug)`` and returning per-sample scores
        in ``[0, 1]``. Passed in so this module has no model dependency.

    Returns
    -------
    torch.Tensor
        Scalar mean of the F_Q scores.
    """
    return torch.mean(fq(x, x_A))


def compound_loss(x: Tensor, x_A: Tensor, fq, cfg) -> tuple[Tensor, Tensor, Tensor]:
    """Eq.(8) gate ``max(L_S, beta * L_A)``.

    Returns the two *unscaled* inputs alongside the gate value because callers log all three
    each step; a single return tensor would force them to recompute the terms.

    Parameters
    ----------
    x : torch.Tensor
        Original image, ``[0, 1]`` float.
    x_A : torch.Tensor
        Attacker output, ``[0, 1]`` float.
    fq : callable
        Frozen F_Q estimator (see :func:`loss_adversarial`).
    cfg : AttackerConfig
        Supplies ``beta``.

    Returns
    -------
    tuple of torch.Tensor
        ``(loss, L_S, L_A)`` — the gate value and both unscaled ``max()`` inputs.
    """
    L_S = loss_similarity(x, x_A)
    L_A = loss_adversarial(x, x_A, fq)
    loss = torch.maximum(L_S, cfg.beta * L_A)
    return loss, L_S, L_A
