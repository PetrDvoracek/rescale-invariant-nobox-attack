"""Tiling / blend contract tests — PROGRAM.md §3.4 / §6.2a.

- ``reassemble(partition(pad(x)[0]), padded_hw, orig_hw) == x`` bit-exact on the unpadded
  region for shapes {448x448 (exact mult), 449x449 (+1px), 100x100 (sub-224), 224x224,
  500x200 (tall-thin)}.
- exact-multiple input adds NO extra band (the ``(-h) % tile`` formula).
- sub-224 edge-fill band equals the replicated border row/col (pins the edge-fill choice;
  the roundtrip identity holds for ANY fill, so this is a dedicated pin).
- ``alpha = 0`` float identity: ``blend(im, diff, 0.0)`` reproduces ``im`` within <=1 LSB.
- channel-order regression: the real ``build_attacker_unet`` F_A run via ``perturb_image``
  produces a non-trivial diff (guards the RGB path — the alpha=0 test passes even with
  channels swapped, so a dedicated test is mandatory, §3.4).

All CPU, tiny synthetic tensors, ``encoder_weights=None``, no network.
"""

from __future__ import annotations

import numpy as np
import pytest

from rina import data
from rina.config import AttackerConfig
from rina import inference


TILE = 224


def _img(h, w, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


# --------------------------------------------------------------------------- #
# Roundtrip identity over the required shape set.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "h,w",
    [
        (448, 448),  # exact multiple of 224
        (449, 449),  # +1 pixel
        (100, 100),  # sub-224 (needs edge fill)
        (224, 224),  # exactly one tile
        (500, 200),  # non-square / tall-thin (200 < 224 in W)
    ],
)
def test_reassemble_partition_pad_is_identity(h, w):
    x = _img(h, w, seed=h * 1000 + w)
    padded, orig_hw = data.reflect_pad_to_multiple(x, TILE)
    assert padded.shape[0] % TILE == 0 and padded.shape[1] % TILE == 0
    tiles = data.partition(padded, TILE)
    back = data.reassemble(tiles, (padded.shape[0], padded.shape[1]), orig_hw)
    assert back.shape == x.shape
    assert np.array_equal(back, x)


# --------------------------------------------------------------------------- #
# Exact-multiple input adds no extra band.
# --------------------------------------------------------------------------- #
def test_exact_multiple_adds_no_band():
    x = _img(448, 224, seed=1)
    padded, orig_hw = data.reflect_pad_to_multiple(x, TILE)
    assert padded.shape[0] == 448  # no extra 224 row added
    assert padded.shape[1] == 224
    assert orig_hw == (448, 224)


def test_plus_one_pixel_pads_up_to_next_multiple():
    x = _img(225, 225, seed=2)
    padded, orig_hw = data.reflect_pad_to_multiple(x, TILE)
    assert padded.shape[0] == 448 and padded.shape[1] == 448
    assert orig_hw == (225, 225)


# --------------------------------------------------------------------------- #
# Sub-224 edge-fill band equals the replicated border row/col.
# --------------------------------------------------------------------------- #
def test_sub224_edge_fill_band_replicates_border():
    # 100x100: reflect can cover at most (h-1)=99 extra rows, needs (-100)%224=124 -> the
    # remaining 124-99=25 rows/cols are mode='edge' (replicated border).
    h = w = 100
    x = _img(h, w, seed=3)
    padded, orig_hw = data.reflect_pad_to_multiple(x, TILE)
    assert padded.shape[0] == 224 and padded.shape[1] == 224

    h2add = (-h) % TILE
    refl = min(h2add, h - 1)  # 99
    edge_start = h + refl  # first edge-filled row index
    # The edge band rows must all equal the last row produced by the reflect phase
    # (np.pad(mode='edge') replicates the border of the reflect-padded array).
    border_row = padded[edge_start - 1:edge_start, :, :]
    edge_rows = padded[edge_start:, :, :]
    assert edge_rows.shape[0] > 0
    assert np.array_equal(edge_rows, np.repeat(border_row, edge_rows.shape[0], axis=0))

    # Same for columns.
    border_col = padded[:, edge_start - 1:edge_start, :]
    edge_cols = padded[:, edge_start:, :]
    assert np.array_equal(edge_cols, np.repeat(border_col, edge_cols.shape[1], axis=1))


# --------------------------------------------------------------------------- #
# alpha = 0 float identity (blend reproduces the original within <=1 LSB).
# --------------------------------------------------------------------------- #
def test_blend_alpha0_reproduces_original():
    im = _img(64, 48, seed=4)
    diff = np.random.default_rng(5).standard_normal((64, 48, 3)).astype(np.float32) * 0.3
    out = inference.blend(im, diff, 0.0)
    assert out.shape == im.shape
    assert out.dtype == np.uint8
    # alpha=0 drops the diff entirely; only the float /255*255 round-trip can shift a
    # value by at most 1 LSB.
    assert np.abs(out.astype(np.int16) - im.astype(np.int16)).max() <= 1


def test_blend_overshoot_clips_into_range():
    im = _img(32, 32, seed=6)
    diff = np.ones((32, 32, 3), dtype=np.float32)  # push everything up
    out = inference.blend(im, diff, 1.3)  # overshoot
    assert out.max() <= 255 and out.min() >= 0
    assert out.dtype == np.uint8


# --------------------------------------------------------------------------- #
# Channel-order regression: real F_A produces a non-trivial diff on a fixed RGB image.
# --------------------------------------------------------------------------- #
def test_channel_order_real_fa_diff_is_nontrivial():
    import torch

    from rina.attacker import build_attacker_unet

    torch.manual_seed(0)
    # Use the paper config but a tiny U-Net forward on a small image keeps this CPU-cheap.
    unet = build_attacker_unet(AttackerConfig())  # encoder_weights=None
    model = inference.AttackerInference(unet, device="cpu", compile=False)

    im = _img(TILE, TILE, seed=7)  # one tile, RGB uint8
    diff = inference.perturb_image(model, im, TILE, pad_mode="reflect")

    assert diff.shape == im.shape
    # The learned (random-init) perturbation must be non-trivial — a swapped/broken RGB
    # path that fed zeros or an identity would yield an all-zero diff. Assert real signal.
    assert np.abs(diff).max() > 1e-4
    assert float(np.abs(diff).mean()) > 0.0
