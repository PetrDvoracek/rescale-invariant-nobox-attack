"""Eq.(8) compound-loss tests — PROGRAM.md §4.1 / §6.2a.

Pins the loss wiring (the empirical 40%/60% active-term band needs a TRAINED F_Q and is
deferred to a @pytest.mark.gpu placeholder, per §4.1):

- A clean pair (x == x_A) drives ``loss_similarity ~= 0``.
- ``compound_loss`` returns a 3-tuple and ``loss == max(L_S, beta * L_A)``.
- β sits on the adversarial term: forcing the stub F_Q ~= 0 gives ``compound ~= L_S`` (β is
  NOT on L_S), and ``paper_config`` wires this up.

A stub ``fq`` (nn.Module returning a controllable constant in [0,1]) stands in for the
frozen F_Q so the losses are import-safe and unit-testable on CPU with no weights.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from rina.config import AttackerConfig, paper_config
from rina.losses import compound_loss, loss_adversarial, loss_similarity


class StubFQ(nn.Module):
    """F_Q stand-in: ignores its inputs and returns a constant per-sample score."""

    def __init__(self, value: float):
        super().__init__()
        self.value = float(value)

    def forward(self, x, x_A):
        n = x.shape[0]
        return torch.full((n, 1), self.value)


def _pair(value_orig=0.5, value_adv=0.5, n=2):
    x = torch.full((n, 3, 16, 16), float(value_orig))
    x_A = torch.full((n, 3, 16, 16), float(value_adv))
    return x, x_A


# --------------------------------------------------------------------------- #
# Clean pair => similarity ~= 0.
# --------------------------------------------------------------------------- #
def test_clean_pair_similarity_is_zero():
    x, x_A = _pair(0.5, 0.5)
    assert torch.isclose(loss_similarity(x, x_A), torch.tensor(0.0), atol=1e-7)


# --------------------------------------------------------------------------- #
# compound_loss is a 3-tuple equal to max(L_S, beta * L_A).
# --------------------------------------------------------------------------- #
def test_compound_returns_3_tuple_and_equals_max():
    x, x_A = _pair(0.2, 0.9)  # large diff so L_S is non-trivial
    fq = StubFQ(0.3)
    cfg = AttackerConfig(beta=15.0)
    out = compound_loss(x, x_A, fq, cfg)
    assert isinstance(out, tuple) and len(out) == 3
    loss, L_S, L_A = out
    expected = torch.maximum(L_S, 15.0 * L_A)
    assert torch.isclose(loss, expected)
    # The returned L_S / L_A are the *unscaled* terms.
    assert torch.isclose(L_S, loss_similarity(x, x_A))
    assert torch.isclose(L_A, loss_adversarial(x, x_A, fq))


# --------------------------------------------------------------------------- #
# β placement: β on L_A => similarity ungated. Force F_Q ~= 0 => compound == L_S.
# --------------------------------------------------------------------------- #
def test_beta_scales_adversarial_with_LA_zero_gives_compound_eq_LS():
    x, x_A = _pair(0.2, 0.9)
    fq = StubFQ(0.0)  # L_A ~= 0
    cfg = AttackerConfig(beta=15.0)
    loss, L_S, L_A = compound_loss(x, x_A, fq, cfg)
    assert torch.isclose(L_A, torch.tensor(0.0), atol=1e-7)
    # similarity term is ungated, so loss == L_S (NOT beta*L_S).
    assert torch.isclose(loss, L_S)
    assert not torch.isclose(loss, cfg.beta * L_S)


# --------------------------------------------------------------------------- #
# Factory wiring: paper_config puts beta on L_A.
# --------------------------------------------------------------------------- #
def test_paper_config_beta_scales_adversarial():
    _, acfg, _ = paper_config()
    assert acfg.beta == 15.0
    # With F_Q forced to 0, the gate equals the unscaled MSE similarity (β is on L_A, not L_S).
    x, x_A = _pair(0.2, 0.9)
    loss, L_S, L_A = compound_loss(x, x_A, StubFQ(0.0), acfg)
    assert torch.isclose(loss, L_S)


# --------------------------------------------------------------------------- #
# Deferred empirical 40%/60% active-term band (needs a TRAINED F_Q) — §4.1.
# --------------------------------------------------------------------------- #
@pytest.mark.gpu
def test_paper_config_active_term_ratio_near_40pct():
    """DEFERRED: the paper's ~40% L_S / ~60% L_A active-term split (§4.1).

    Cannot be asserted on CPU without trained MobileNetV3-Large F_Q weights — a
    randomly-initialized / stub F_Q produces an arbitrary split, so the only honest
    CPU assertion is the *wiring* above (the active-flag formula). This placeholder
    documents the deferred 40% +/- 15pp band check on a fixed seeded real batch with
    a trained F_Q (gated behind GPU + checkpoint availability).
    """
    pytest.skip("40% active-term band needs a trained F_Q checkpoint (deferred, §4.1)")
