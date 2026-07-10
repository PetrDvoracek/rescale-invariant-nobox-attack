"""Backward-compatibility shim (PROGRAM.md §6.1a).

The code that used to live here has moved into the ``rina/`` package (one module per paper
symbol). This file is intentionally kept as a thin compat surface and is the *only* sanctioned
``from lib import *`` post-refactor — it is NOT the home of the code.

It serves two distinct purposes:

1. **Pickle-path resolution (load-bearing).** Old Lightning checkpoints were written with
   ``save_hyperparameters()`` (no ``ignore=``), which pickled the ``nn.Module`` graph under its
   then-current import paths: ``lib.TraineeAttacker``, ``lib.Trainee``, ``lib.ModelSim``. After the
   move those classes live in ``rina.*``; registering ``sys.modules["lib"] = <this module>`` plus
   re-exporting the three pickled classes lets the old refs resolve to their ``rina.*`` homes so
   legacy ``.ckpt`` files still load. (The pickled ``smp.Unet`` / ``timm`` submodules need no entry
   — they live under their own stable third-party package paths.)

2. **Source compatibility (not pickle).** The dataset/helper re-exports (``DSAugmentFactor``,
   ``DSAttack``, ``AttackerInference``) exist only so legacy ``from lib import *`` source keeps
   importing. They are never in a saved hyperparameter graph.

Shim invariant (§6.1a): this module is a *leaf* — it depends on ``rina.*`` and no ``rina.*`` module
imports ``lib``. The re-exports must run BEFORE the ``sys.modules`` alias so unpickling never sees a
half-initialised ``lib``. CI/import-lint forbids ``import lib`` inside ``rina/``.
"""

import sys

# --- pickle-path-bearing classes (must resolve for old .ckpt unpickling) ---
from rina.attacker import TraineeAttacker
from rina.quality import Trainee
from rina.simulation import ModelSim

# --- source-compat re-exports (never pickled) ---
from rina.augment import DSAugmentFactor
from rina.data import DSAttack
from rina.inference import AttackerInference

# Belt-and-suspenders for pickled "lib.*" references. Runs only after the imports above so the
# alias never exposes a partially-initialised module to pickle.
sys.modules["lib"] = sys.modules[__name__]

__all__ = [
    "TraineeAttacker",
    "Trainee",
    "ModelSim",
    "DSAugmentFactor",
    "DSAttack",
    "AttackerInference",
]
