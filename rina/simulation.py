"""F_S — the frozen directional-Sobel bank (paper §4.2).

A fixed, non-trainable bank of 12 directional 5x5 Sobel-like kernels, one every 30°
(Fig. 3), applied depthwise (``groups=3``, ``padding=2``) so each kernel convolves each RGB
channel independently. This is the paper's "single non-trainable convolutional layer". Each
kernel is normalized by ``sum(|w|)``, so responses are bounded and need no learned
normalization.

Channel bookkeeping is the one thing to get right: ``ModelSim`` emits **36 channels per
image** (12 kernels × 3 RGB channels). The 72 channels the F_Q backbone takes are the
per-image concatenation ``cat(F_S(orig), F_S(aug))`` = 2 × 36, formed in
:mod:`rina.quality`, not here. (The paper writes this as ``2k = 24`` for ``k = 12`` filters;
the code realizes each filter per RGB channel, so its counts are 36 and 72.)
"""

from __future__ import annotations

from typing import Final

import cv2
import numpy as np
import torch

#: Channels emitted by ``ModelSim.forward`` per image (12 kernels × 3 RGB channels).
FS_OUT_CHANNELS: Final[int] = 36


class ModelSim(torch.nn.Module):
    """The frozen directional-Sobel bank (F_S).

    ``forward`` returns 36 channels per image; the 72-channel F_Q input is the concatenation
    of two such outputs, formed in :mod:`rina.quality`.
    """

    def __init__(self):
        super().__init__()
        self.kernels = self.init_kernels()

        # Each conv is created 1x1 and then has its weight replaced by the real 5x5 kernel
        # below. It keeps the (frozen) bias from init; building 1x1 leaves that bias init
        # exactly as the original network drew it, so checkpoints stay compatible.
        convs = []
        for kernel in self.kernels:
            conv = torch.nn.Conv2d(3, 3, kernel_size=1, padding=2, groups=3)
            conv.weight = torch.nn.Parameter(kernel.to(conv.weight.dtype))
            convs.append(conv)
        self.depthwise_convs = torch.nn.ModuleList(convs)
        self.depthwise_convs.requires_grad_(False)

    def forward(self, x):
        """Apply the frozen bank.

        Parameters
        ----------
        x : torch.Tensor
            Input batch of shape ``(N, 3, H, W)``.

        Returns
        -------
        torch.Tensor
            Shape ``(N, 36, H, W)``.
        """
        return torch.cat([conv(x) for conv in self.depthwise_convs], dim=1)

    @staticmethod
    def init_kernels():
        """Build the 12 directional Sobel kernels, each normalized by ``sum(|w|)``.

        Returns
        -------
        torch.Tensor
            Shape ``(12, 3, 1, 5, 5)`` — one 5x5 plane broadcast across the 3 channels, for
            each orientation 0°, 30°, ..., 330°, laid out as depthwise ``Conv2d`` weights.
        """
        kernels = []
        kx = np.array(
            [
                [8, 12, 16, 12, 8],
                [6, 9, 12, 9, 6],
                [0, 0, 0, 0, 0],
                [-6, -9, -12, -9, -6],
                [-8, -12, -16, -12, -8],
            ]
        )
        ky = cv2.rotate(kx, cv2.ROTATE_90_CLOCKWISE)
        for angle in range(0, 360, 30):
            wx = np.cos(angle / 180 * np.pi)
            wy = np.sin(angle / 180 * np.pi)
            tmp = kx * wx + ky * wy
            tmp /= np.sum(np.abs(tmp))
            kernels.append(np.expand_dims(np.stack([tmp, tmp, tmp], axis=-1), -1))
        return torch.tensor(np.array(kernels)).permute(0, 3, 4, 1, 2)


def build_modelsim() -> ModelSim:
    """Construct the frozen directional-Sobel bank F_S.

    Returns
    -------
    ModelSim
        The frozen directional-Sobel bank.
    """
    return ModelSim()
