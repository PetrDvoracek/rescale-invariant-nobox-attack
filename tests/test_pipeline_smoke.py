"""End-to-end CLI pipeline smoke test — PROGRAM.md §10 (the integration gate, CPU, not gpu).

Exercises the four-command wiring (train-quality -> train-attacker -> attack ->
evaluate/measure-ssim) the §6.2a unit tests never touch: Typer dispatch, factory
assembly, the mandatory ``--quality-ckpt`` guard (§4.1), the §6.1a ``ignore=`` reload,
and ``--logger none`` not crashing on ``log_image`` (§12).

This is the wiring twin of §6.2a (math). It is GPU-free and fast (single step/epoch) and
is NOT a fidelity / numbers check (deferred §3.x). It pins the F_Q backbone to ``resnet18``
(``--backbone resnet18 --no-pretrained``) on both ``train-quality`` and ``train-attacker`` so
the F_Q the attacker reconstructs matches the one ``train-quality`` wrote — keeping the §6.1a
reload self-consistent and weight-free on CPU. All steps use tiny synthetic datasets (conftest
fixtures), weight-free models, and ``num_workers=0``.
"""

from __future__ import annotations

import glob
from pathlib import Path

import pytest
import torch
from typer.testing import CliRunner

from cli import app
from rina.config import AttackConfig, QualityConfig, paper_config
from rina.quality import Trainee, build_modelsim, build_quality_backbone

runner = CliRunner()

TAG = "t"


def _run(*args):
    result = runner.invoke(app, [str(a) for a in args])
    return result


@pytest.fixture
def trained_fq(tmp_path, tiny_coco):
    """Step (1): train-quality -> fq.ckpt that reloads via the §6.1a path."""
    models_dir = tmp_path / "models"
    result = _run(
        "train-quality",
        "--coco-test2017", tiny_coco,
        "--models-dir", models_dir,
        "--backbone", "resnet18",  # small, weight-free F_Q; the attacker reconstructs it
        "--no-pretrained",
        "--epochs", 1,
        "--batch-size", 2,
        "--num-workers", 0,
        "--logger", "none",
        "--accelerator", "cpu",
    )
    assert result.exit_code == 0, result.output
    ckpt = models_dir / "fq.ckpt"
    assert ckpt.exists(), result.output
    return ckpt


def test_1_train_quality_produces_reloadable_fq(trained_fq):
    qcfg = QualityConfig(estimator_backbone="resnet18", quality_pretrained=False)  # the train cfg
    reloaded = Trainee.load_from_checkpoint(
        str(trained_fq),
        map_location="cpu",
        model=build_quality_backbone(qcfg),
        model_sim=build_modelsim(),
    )
    assert reloaded.model is not None
    reloaded.eval()  # F_Q is used frozen/eval; BatchNorm needs eval for batch=1
    out = reloaded(torch.rand(2, 3, 64, 64), torch.rand(2, 3, 64, 64))
    assert out.shape == (2, 1)


@pytest.fixture
def trained_fa(tmp_path, tiny_coco, trained_fq):
    """Step (2): train-attacker --quality-ckpt <fq> -> fa.ckpt reloadable on CPU."""
    from rina.attacker import TraineeAttacker, build_attacker_unet
    from rina.quality import build_quality

    models_dir = trained_fq.parent
    result = _run(
        "train-attacker",
        "--coco-test2017", tiny_coco,
        "--quality-ckpt", trained_fq,
        "--models-dir", models_dir,
        "--backbone", "resnet18",  # reconstruct the resnet18 F_Q matching the ckpt
        "--max-steps", 1,
        "--batch-size", 2,
        "--num-workers", 0,
        "--logger", "none",
        "--accelerator", "cpu",
        "--precision", "32",
    )
    assert result.exit_code == 0, result.output
    ckpt = models_dir / "fa.ckpt"
    assert ckpt.exists(), result.output

    # Reloads via TraineeAttacker.load_from_checkpoint on CPU (§6.1a layer-2). The F_Q
    # is a registered submodule, so its weights ARE in the F_A state_dict; the
    # reconstruct must pass a real Trainee of matching architecture (resnet18 F_Q), not a
    # stub — exactly the §6.1a asymmetry (model_augmentfactor is a WHOLE Trainee).
    qcfg = QualityConfig(estimator_backbone="resnet18", quality_pretrained=False)
    _, acfg, _ = paper_config()
    reloaded = TraineeAttacker.load_from_checkpoint(
        str(ckpt),
        map_location="cpu",
        model_attacker=build_attacker_unet(acfg),
        model_augmentfactor=build_quality(qcfg),
    )
    assert reloaded.model_attacker is not None
    return ckpt


def test_2_train_attacker_produces_reloadable_fa(trained_fa):
    assert trained_fa.exists()


def test_2_negative_train_attacker_without_quality_ckpt_fails(tmp_path, tiny_coco):
    """NEGATIVE case (§4.1): train-attacker WITHOUT --quality-ckpt exits non-zero."""
    result = _run(
        "train-attacker",
        "--coco-test2017", tiny_coco,
        "--models-dir", tmp_path / "models",
        "--max-steps", 1,
        "--batch-size", 2,
        "--num-workers", 0,
        "--logger", "none",
        "--accelerator", "cpu",
        "--precision", "32",
    )
    assert result.exit_code != 0
    # The §4.1 guard message mentions the F_Q-checkpoint requirement.
    combined = (result.output or "") + (str(result.exception) if result.exception else "")
    assert "F_Q checkpoint required" in combined


@pytest.fixture
def attacked_val(tiny_imagenet_val, trained_fa):
    """Step (3): attack the in1k val tree at alphas 0.0 and 1.0."""
    result = _run(
        "attack",
        tiny_imagenet_val,  # dataset root (globber wants <root>/val/**/*)
        trained_fa,
        "--dataset", "in1k",
        "--tag", TAG,
        "--device", "cpu",
        "-a", 0.0,
        "-a", 1.0,
    )
    assert result.exit_code == 0, result.output
    # At least one adversarial JPEG under val_adversarial-1.0_tag-t/.
    adv_folder = tiny_imagenet_val / f"val_adversarial-1.0_tag-{TAG}"
    jpegs = glob.glob(str(adv_folder / "**" / "*.JPEG"), recursive=True)
    assert jpegs, f"no adversarial JPEGs under {adv_folder}"
    return adv_folder


def test_3_attack_produces_adversarial_jpeg(attacked_val):
    assert any(attacked_val.rglob("*.JPEG"))


def test_3_attack_and_evaluate_share_alpha_source():
    # Both commands default to AttackConfig().alphas (§3.3) — assert the single source.
    assert AttackConfig().alphas == (0.85, 1.0, 1.15)


def test_4_evaluate_writes_9_field_csv(tiny_imagenet_val, attacked_val, tmp_path, monkeypatch):
    # evaluate takes the SAME dataset root + --dataset as `attack` and derives the folder via
    # rina.data.adversarial_dir (the identical function attack writes to) — no asymmetric path.
    monkeypatch.chdir(tmp_path)  # CSV is written relative to cwd
    result = _run(
        "evaluate",
        tiny_imagenet_val,
        "--dataset", "in1k",
        "--tag", TAG,
        "--device", "cpu",
        "--smoke",
        "-a", 1.0,
    )
    assert result.exit_code == 0, result.output
    csv_path = tmp_path / f"eval_adversarial-1.0_tag-{TAG}.csv"
    assert csv_path.exists(), result.output
    header = csv_path.read_text().splitlines()[0].strip()
    assert header == "model,top1,top1_err,top5,top5_err,param_count,img_size,crop_pct,interpolation"


def test_5_measure_ssim_prints_finite_number(attacked_val):
    result = _run(
        "measure-ssim",
        attacked_val,
        "--dataset", "in1k",
        "--alpha", 1.0,
        "--tag", TAG,
        "--subset", 4,
    )
    assert result.exit_code == 0, result.output
    # Output: "dataset_name SSIM\n<name> <value>"
    last = result.output.strip().splitlines()[-1]
    parts = last.split()
    val = float(parts[-1])
    assert val == val  # not NaN
    assert -1.0 <= val <= 1.0
