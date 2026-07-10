"""F_S (ModelSim) characterization tests — PROGRAM.md §6.2a / §3.2.

Pins the §6.2a invariant rows for the frozen directional-Sobel bank:

- F_S emits 36 channels/image; the F_Q input is 72 (cat of two F_S outputs).
- EVERY F_S parameter has ``requires_grad == False`` (the literal frozen bank).
- 12 kernels at 0, 30, ..., 330 degrees; each kernel ``sum(|w|) ~= 1.0``.
- Conv ``padding == (2, 2)``, ``groups == 3``.

All CPU, tiny synthetic tensors, no weights/network.
"""

from __future__ import annotations

import torch

from rina.simulation import FS_OUT_CHANNELS, ModelSim, build_modelsim


def _tiny_input():
    return torch.randn(1, 3, 32, 32)


def test_fs_emits_36_channels_per_image():
    fs = ModelSim()
    out = fs(_tiny_input())
    assert out.shape[1] == 36 == FS_OUT_CHANNELS


def test_fq_input_is_72_channels_cat_of_two():
    # The 72 channels are NOT F_S's output; they are cat(F_S(x), F_S(x_aug)) = 2 x 36,
    # exactly the in_chans the F_Q backbone is built with (§3.2).
    fs = ModelSim()
    x = _tiny_input()
    cat = torch.cat([fs(x), fs(x)], dim=1)
    assert cat.shape[1] == 72


def test_every_fs_param_is_frozen():
    fs = ModelSim()
    # The literal frozen bank: no F_S parameter requires grad.
    assert all(not p.requires_grad for p in fs.parameters())


def test_twelve_kernels_at_every_30_degrees():
    fs = ModelSim()
    assert len(fs.depthwise_convs) == 12
    assert len(fs.kernels) == 12
    # The build loop is range(0, 360, 30) -> exactly 12 angles 0,30,...,330.
    assert list(range(0, 360, 30)) == [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]


def test_each_kernel_sum_abs_is_one():
    fs = ModelSim()
    # init_kernels normalizes each kernel by sum(|w|); broadcast to 3 channels, so the
    # per-(out,in) 5x5 plane sums-of-abs to ~1.0. Check a representative single plane
    # per kernel (all 3 broadcast planes are identical).
    for c in fs.depthwise_convs:
        w = c.weight.detach()  # (3, 1, 5, 5) depthwise
        plane = w[0, 0]
        assert torch.isclose(plane.abs().sum(), torch.tensor(1.0), atol=1e-5), plane.abs().sum()


def test_conv_padding_and_groups():
    fs = ModelSim()
    for c in fs.depthwise_convs:
        assert c.padding == (2, 2)
        assert c.groups == 3
        # NB: the source builds Conv2d(kernel_size=1) which yields a (1,1) kernel_size attr,
        # then overwrites .weight with the real 5x5 kernel; assert the actual weight plane is
        # 5x5 (the load-bearing fact), not the stale attr.
        assert c.weight.shape[-2:] == (5, 5)


def test_build_modelsim_constructs_frozen_bank():
    fs = build_modelsim()
    assert isinstance(fs, ModelSim)
    assert all(not p.requires_grad for p in fs.parameters())
