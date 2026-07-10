"""Shared pytest fixtures for the CPU-only RINA test suite (PROGRAM.md §10).

Everything here is CPU-only, network-free, and uses no real datasets or checkpoints.
Two synthetic dataset trees are provided:

- ``tiny_coco``: a flat dir of >=4 synthetic >=256x256 JPEGs (a fake COCO ``test2017``),
  consumed by ``DSAugmentFactor`` / ``DSAttack`` (both filter <224 images, so the
  synthetic images are deliberately larger than 224).
- ``tiny_imagenet_val``: a ``val/<wnid>/*.JPEG`` tree (>=2 wnids, a few imgs each), the
  layout ``rina.inference._get_paths`` globs for ``in1k`` (``<root>/val/**/*``). NOT a
  ``torchvision.datasets.ImageNet`` tree — no devkit, so it works without ILSVRC files.

The repo root is added to ``sys.path`` so ``import cli`` / ``import rina`` / ``import lib``
resolve when pytest is invoked from the repo root.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Make the repo root importable (cli.py + rina/ + lib.py live there).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def pytest_configure(config):
    # Register the gpu marker programmatically too (belt-and-suspenders with pytest.ini),
    # so `-m 'not gpu'` and `@pytest.mark.gpu` never warn even if config discovery misses
    # the ini for some invocation.
    config.addinivalue_line(
        "markers",
        "gpu: tests that require CUDA; skipped on CPU-only boxes and excluded from the CPU CI job.",
    )


def _write_jpeg(path: Path, h: int, w: int, seed: int) -> None:
    """Write a deterministic synthetic RGB JPEG of size (h, w) to ``path``."""
    import cv2

    rng = np.random.default_rng(seed)
    # Smooth-ish content so JPEG encodes cleanly and SSIM is well-defined; a few
    # structured stripes give the Sobel bank something non-trivial to respond to.
    img = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    img[:, ::8, :] = 0
    img[::8, :, :] = 255
    path.parent.mkdir(parents=True, exist_ok=True)
    # cv2 writes BGR; the array is treated as BGR on disk (consistent round-trip).
    assert cv2.imwrite(str(path), img), f"failed to write {path}"


@pytest.fixture
def tiny_coco(tmp_path) -> Path:
    """A flat dir of 4 synthetic 256x256 JPEGs (a fake COCO ``test2017``)."""
    root = tmp_path / "coco_test2017"
    root.mkdir(parents=True, exist_ok=True)
    for i in range(4):
        _write_jpeg(root / f"img_{i:04d}.jpg", 256, 256, seed=100 + i)
    return root


@pytest.fixture
def tiny_imagenet_val(tmp_path) -> Path:
    """A ``val/<wnid>/*.JPEG`` tree with 2 wnids x 3 imgs each (fake ImageNet val).

    Returns the dataset *root* (the parent of ``val``), which is what the ``attack``
    in1k globber (``<root>/val/**/*``) and the ``original_path`` pairing expect.
    """
    root = tmp_path / "imagenet"
    val = root / "val"
    for wi, wnid in enumerate(["n00000001", "n00000002"]):
        for k in range(3):
            _write_jpeg(val / wnid / f"ILSVRC2012_val_{wi}_{k:03d}.JPEG", 256, 256, seed=200 + wi * 10 + k)
    return root
