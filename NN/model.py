"""Neural network model producing Heston parameters."""
from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

from heston_torch import HestonParams, carr_madan_call_torch


class HestonParamNet(nn.Module):
    """Simple feed-forward network mapping features to Heston parameters."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 5),
        )

    def forward(self, x: torch.Tensor) -> HestonParams:
        outputs = self.layers(x)
        components: Tuple[torch.Tensor, ...] = outputs.unbind(-1)
        return HestonParams.from_unconstrained(*components)


def _reshape_input(value) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    tensor = torch.as_tensor(value, dtype=torch.float64)
    original_shape = tensor.shape
    flat = tensor.reshape(-1)
    return flat, original_shape


def _expand_param(component: torch.Tensor, batch_size: int) -> torch.Tensor:
    comp = component.to(dtype=torch.float64)
    if comp.ndim == 0:
        return comp.repeat(batch_size)
    return comp.reshape(batch_size)


def price_with_params(
    S0,
    K,
    T,
    r,
    params: HestonParams,
    q: float = 0.0,
    alpha: float = 1.5,
    Nfft: int = 2 ** 12,
    eta: float = 0.25,
) -> torch.Tensor:
    """Compute option prices for each sample given predicted parameters."""
    S0_flat, original_shape = _reshape_input(S0)
    K_flat, _ = _reshape_input(K)
    T_flat, _ = _reshape_input(T)
    r_flat, _ = _reshape_input(r)
    batch = S0_flat.shape[0]

    if not (batch == K_flat.shape[0] == T_flat.shape[0] == r_flat.shape[0]):
        raise ValueError("S0, K, T, and r must share the same shape.")

    kappa = _expand_param(params.kappa, batch)
    theta = _expand_param(params.theta, batch)
    sigma = _expand_param(params.sigma, batch)
    rho = _expand_param(params.rho, batch)
    v0 = _expand_param(params.v0, batch)

    prices = []
    for idx in range(batch):
        single_params = HestonParams(
            kappa=kappa[idx],
            theta=theta[idx],
            sigma=sigma[idx],
            rho=rho[idx],
            v0=v0[idx],
        )
        price = carr_madan_call_torch(
            S0_flat[idx],
            r_flat[idx],
            q,
            T_flat[idx],
            single_params,
            K_flat[idx],
            alpha=alpha,
            Nfft=Nfft,
            eta=eta,
        )
        prices.append(price)

    price_tensor = torch.stack(prices)
    if original_shape == torch.Size([]):
        return price_tensor.squeeze()
    return price_tensor.reshape(original_shape)


def rmse_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Root-mean-square error loss."""
    return torch.sqrt(torch.mean((predicted - target) ** 2))

