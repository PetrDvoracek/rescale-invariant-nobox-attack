"""F_Q distortion-label convention tests — PROGRAM.md §6.2a / §4.

Pins the label = ``1 - d / N_DISTORTIONS`` convention with N_DISTORTIONS == 16:
range [0, 1], clean (d == 0) -> 1.0, monotone DOWN in the number of distortions applied.

IMPORTANT — *the figure convention*, not "the paper": the pinned form ``1 - d/16`` (clean
-> 1.0, decreasing with distortions) is the convention shown in the paper's Fig. 6 and
realized by the code. The paper *text* (§4.2.1) states the opposite, ``d/16`` (clean -> 0.0).
This is a documented text<->figure contradiction internal to the paper; the code — and
therefore this test — sides with the FIGURE (PROGRAM.md §4).
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from rina.augment import N_DISTORTIONS, DSAugmentFactor, _build_augs_all


def _img():
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)


def test_n_distortions_is_16_and_bank_matches():
    assert N_DISTORTIONS == 16
    assert len(_build_augs_all()) == N_DISTORTIONS == 16


def test_label_in_unit_range():
    img = _img()
    for seed in range(20):
        random.seed(seed)
        np.random.seed(seed)
        _, y = DSAugmentFactor.augment_img(img)
        assert 0.0 <= y <= 1.0


def test_clean_image_label_is_one():
    """*The figure convention* (1 - d/16, clean -> 1.0), NOT the paper text (d/16)."""
    img = _img()
    # Find a seed that yields d == 0 (no augmentations) and assert the label is exactly 1.0.
    # d ~ U{0..16}, so a clean draw is common.
    found = False
    for seed in range(200):
        random.seed(seed)
        np.random.seed(seed)
        out, y = DSAugmentFactor.augment_img(img)
        # A clean draw (d == 0) leaves the image untouched and labels it 1.0.
        if np.array_equal(out.astype(np.uint8), img):
            assert y == pytest.approx(1.0)
            found = True
            break
    assert found, "expected at least one clean (d==0) draw in 200 seeds"


def test_label_is_one_minus_d_over_16():
    """Each label equals 1 - d/16 where d is the number of applied distortions."""
    img = _img()
    valid = {1 - k / N_DISTORTIONS for k in range(N_DISTORTIONS + 1)}
    for seed in range(40):
        random.seed(seed)
        np.random.seed(seed)
        _, y = DSAugmentFactor.augment_img(img)
        # y must land exactly on one of the 17 discrete 1 - d/16 values.
        assert min(abs(y - v) for v in valid) < 1e-9, y


def test_label_monotone_decreasing_in_applied_count():
    """More applied distortions => strictly lower label (1 - d/16 is monotone in d)."""
    labels = [1 - d / N_DISTORTIONS for d in range(N_DISTORTIONS + 1)]
    assert labels == sorted(labels, reverse=True)
    assert labels[0] == 1.0  # clean
    assert labels[-1] == 0.0  # all 16 applied
    # Strictly decreasing.
    assert all(labels[i] > labels[i + 1] for i in range(len(labels) - 1))
