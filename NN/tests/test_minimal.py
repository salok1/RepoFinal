"""Minimal tests for Heston NN components."""
from __future__ import annotations

import torch

from bs_iv import bs_call_torch, iv_call_brent_torch
from heston_torch import HestonParams, heston_cf
from model import HestonParamNet, price_with_params, rmse_loss


torch.set_default_dtype(torch.float64)


def test_heston_cf_zero_equals_one() -> None:
    params = HestonParams.from_unconstrained(
        torch.tensor(0.2),
        torch.tensor(0.1),
        torch.tensor(0.3),
        torch.tensor(0.1),
        torch.tensor(0.2),
    )
    phi_zero = heston_cf(torch.tensor(0.0, dtype=torch.float64), 1.0, 100.0, 0.01, 0.0, params)
    assert torch.allclose(phi_zero, torch.ones_like(phi_zero, dtype=torch.complex128), atol=1e-10)


def test_bs_iv_inversion_accuracy() -> None:
    S0, K, T, r, q = 100.0, 110.0, 0.5, 0.01, 0.0
    vol_true = torch.tensor(0.25)
    price = bs_call_torch(S0, K, T, r, q, vol_true)
    implied = iv_call_brent_torch(price, S0, K, T, r, q)
    assert torch.allclose(implied, vol_true, atol=1e-6)


def test_nn_forward_backward_smoke() -> None:
    net = HestonParamNet()
    batch = 3
    X = torch.randn(batch, 3, dtype=torch.float64)
    S0 = torch.full((batch,), 100.0, dtype=torch.float64)
    K = torch.full((batch,), 100.0, dtype=torch.float64)
    T = torch.full((batch,), 1.0, dtype=torch.float64)
    r = torch.full((batch,), 0.01, dtype=torch.float64)

    params = net(X)
    preds = price_with_params(S0, K, T, r, params, alpha=1.5, Nfft=2 ** 9, eta=0.25)
    loss = rmse_loss(preds, torch.zeros_like(preds))
    loss.backward()

    grad_norm = sum(p.grad.norm().item() for p in net.parameters() if p.grad is not None)
    assert grad_norm > 0.0

