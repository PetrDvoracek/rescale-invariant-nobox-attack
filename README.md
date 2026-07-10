# RINA — Resolution-invariant no-box attack (reproduction)

Reproduction code for **Dvořáček, Hurtík, Števuliáková, _Resolution-invariant no-box attack_**,
*Computer Vision and Image Understanding* 269 (2026) 104767
([DOI 10.1016/j.cviu.2026.104767](https://doi.org/10.1016/j.cviu.2026.104767); PDF in
`1-s2.0-S1077314226001347-main.pdf`).

A U-Net **attacker** (`F_A`) learns to output a perturbed version of any input image. The
perturbation pattern is universal (image-independent), tiles across arbitrary resolutions, and
degrades ImageNet-classifier accuracy while staying perceptually similar to the original.
*No-box* means the attacker is trained without query access to the victim: it is supervised by a
frozen **quality estimator** (`F_Q`) built on a fixed directional-Sobel bank (`F_S`), and is
evaluated against off-the-shelf timm classifiers (`F_N`) it never saw during training.

This is a legibility-first rebuild of the original research code: the goal is that a reader
holding the PDF can map each equation/figure/symbol to a named piece of code in seconds. The
design rationale, every deviation from the paper, and the deferred items live in
**[PROGRAM.md](PROGRAM.md)** (the rebuild plan) and **[MATCH.md](MATCH.md)** (the deviation
ledger). Symbol→code anchors are in **[PAPER_MAP.md](PAPER_MAP.md)**; the exact reproduction
pipeline is in **[REPRODUCE.md](REPRODUCE.md)**.

## Configuration

A single paper-faithful configuration: `paper_config()` bundles `QualityConfig` /
`AttackerConfig` / `AttackConfig`, reproducing the paper *as written* — Eq.(8) with β=15 on the
adversarial term, MSE similarity, MobileNetV3-Large pretrained `F_Q`, lr 1e-3, 387k iterations,
`d ~ U(0,16)`. Every hyperparameter is a typed field on a config dataclass; individual `--flag`
overrides compose on top for ablations.

Defaults reproduce the paper's **method**, not its numbers from a clean clone — `F_Q` is
*trained*, not shipped, so reproduction is a multi-step pipeline (see REPRODUCE.md and the
mandatory `train-quality` prerequisite in PROGRAM.md §4.1).

## Module map — one file per paper symbol

Everything lives in a flat `rina/` package; `cli.py` is a thin Typer driver; `lib.py` is a compat
shim that re-exports moved classes so legacy pickled checkpoints still load.

| Paper symbol | File | Contents |
|---|---|---|
| **F_S** | `rina/simulation.py` | `ModelSim` — frozen 12×(5×5) directional-Sobel bank, every 30°, depthwise |
| **F_Q** | `rina/quality.py` | `Trainee` — FMQ quality estimator (F_S bank + timm backbone + sigmoid head) |
| **F_A** | `rina/attacker.py` | `TraineeAttacker` — U-Net attacker + training/validation loop |
| **F_N** | `rina/victims.py` | `VICTIMS` — timm victim classifiers + validation loop |
| Eq.(8) loss | `rina/losses.py` | `compound_loss`, `loss_similarity`, `loss_adversarial` |
| Eq.(9) blend, tiling | `rina/inference.py`, `rina/data.py` | `blend`, `AttackerInference`; `reflect_pad_to_multiple` / `partition` / `reassemble` |
| distortion pipeline | `rina/augment.py` | `DSAugmentFactor`, `N_DISTORTIONS=16`, the F_Q label |
| config | `rina/config.py` | `QualityConfig` / `AttackerConfig` / `AttackConfig`, `paper_config()` |

## Quickstart

**Install** — either build the pinned Docker image, or pip-install the pinned requirements:

```bash
# Docker (CUDA base, pinned deps)
docker build -f docker/Dockerfile -t rina .

# or pip into a venv
pip install -r docker/requirements.txt
```

**Run the CPU test suite** (no GPU, no datasets, no checkpoints needed):

```bash
pytest -m 'not gpu'
```

**The five CLI commands** (`python cli.py <command> --help` for full flags):

```bash
# 1. train F_Q (prerequisite for F_A)
python cli.py train-quality --coco-test2017 /data/coco/test2017 --backbone tf_mobilenetv3_large_100 --pretrained

# 2. train F_A (requires a trained F_Q checkpoint)
python cli.py train-attacker --coco-test2017 /data/coco/test2017 --quality-ckpt ./models/fq.ckpt

# 3. generate adversarial folders (Eq.(9) blend; default alphas 0.85/1.0/1.15)
python cli.py attack /data/imagenet ./models/fa.ckpt --dataset in1k --device cuda --tag paper

# 4. evaluate classifier accuracy (timm victims = F_N; one CSV per alpha)
python cli.py evaluate /data/imagenet --dataset in1k --tag paper --device cuda

# 5. measure SSIM vs the paired originals (fixed seeded subset)
python cli.py measure-ssim /data/imagenet/val_adversarial-1.0_tag-paper --dataset in1k --alpha 1.0 --tag paper
```

For account-free / offline runs, pass `--logger csv` (or `--logger none`), or keep wandb with
`WANDB_MODE=offline` (PROGRAM.md §12). See **[REPRODUCE.md](REPRODUCE.md)** for the full ordered
pipeline, data acquisition, and the (deferred) results table.

## Further docs

- **[PAPER_MAP.md](PAPER_MAP.md)** — paper symbol/equation → `module:Symbol` anchors (CI-checked).
- **[REPRODUCE.md](REPRODUCE.md)** — exact ordered commands, data versions, determinism/offline notes.
- **[PROGRAM.md](PROGRAM.md)** — the rebuild design, decisions, and deferred items.
- **[MATCH.md](MATCH.md)** — the paper-vs-code deviation ledger.
- **[AGENTS_DEV.md](AGENTS_DEV.md)** — developer/agent notes.
- **[CLAUDE.md](CLAUDE.md)** — repo guidance for Claude Code.

## Citation

```bibtex
@article{Dvoracek2026RINA,
  title   = {Resolution-invariant no-box attack},
  author  = {Dvo{\v{r}}{\'a}{\v{c}}ek, Petr and Hurt{\'i}k, Petr and {\v{S}}tevuli{\'a}kov{\'a}, Petra},
  journal = {Computer Vision and Image Understanding},
  volume  = {269},
  pages   = {104767},
  year    = {2026},
  doi     = {10.1016/j.cviu.2026.104767}
}
```
