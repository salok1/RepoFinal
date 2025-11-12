"""Black-Scholes utilities for option pricing and implied volatility inversion."""
from __future__ import annotations

import torch
from torch.distributions import Normal


_DTYPE = torch.float64
_NORMAL = Normal(loc=torch.tensor(0.0, dtype=_DTYPE), scale=torch.tensor(1.0, dtype=_DTYPE))


def _to_tensor(value) -> torch.Tensor:
    return torch.as_tensor(value, dtype=_DTYPE)


def _norm_cdf(x: torch.Tensor) -> torch.Tensor:
    return _NORMAL.cdf(x)


def bs_call_torch(S0, K, T, r, q, vol) -> torch.Tensor:
    """Standard Black-Scholes call price evaluated in torch.float64."""
    S0_t = _to_tensor(S0)
    K_t = _to_tensor(K)
    T_t = _to_tensor(T)
    r_t = _to_tensor(r)
    q_t = _to_tensor(q)
    vol_t = torch.clamp(_to_tensor(vol), min=1e-12)

    sqrt_T = torch.sqrt(torch.clamp(T_t, min=1e-12))
    sigma_sqrt_T = vol_t * sqrt_T
    log_term = torch.log(torch.clamp(S0_t / K_t, min=1e-16))
    drift = (r_t - q_t + 0.5 * vol_t**2) * T_t
    d1 = torch.where(
        sigma_sqrt_T > 0,
        (log_term + drift) / sigma_sqrt_T,
        torch.zeros_like(sigma_sqrt_T),
    )
    d2 = d1 - sigma_sqrt_T
    discount_domestic = torch.exp(-r_t * T_t)
    discount_foreign = torch.exp(-q_t * T_t)
    call = S0_t * discount_foreign * _norm_cdf(d1) - K_t * discount_domestic * _norm_cdf(d2)
    return call


def iv_call_brent_torch(
    C,
    S0,
    K,
    T,
    r,
    q: float = 0.0,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> torch.Tensor:
    """Invert Black-Scholes call price to implied volatility via safeguarded bisection."""
    price = _to_tensor(C)
    S0_t = _to_tensor(S0)
    K_t = _to_tensor(K)
    T_t = _to_tensor(T)
    r_t = _to_tensor(r)
    q_t = _to_tensor(q)

    intrinsic = torch.clamp(S0_t * torch.exp(-q_t * T_t) - K_t * torch.exp(-r_t * T_t), min=0.0)
    if price < intrinsic - 1e-12:
        raise ValueError("Market price is below intrinsic value; cannot compute implied volatility.")

    vol_low = torch.tensor(1e-6, dtype=_DTYPE)
    vol_high = torch.tensor(0.5, dtype=_DTYPE)
    price_high = bs_call_torch(S0_t, K_t, T_t, r_t, q_t, vol_high)
    while price_high < price and vol_high < 5.0:
        vol_high = vol_high * 2.0
        price_high = bs_call_torch(S0_t, K_t, T_t, r_t, q_t, vol_high)

    if price_high < price:
        raise ValueError("Failed to bracket implied volatility up to 5.0.")

    for _ in range(max_iter):
        vol_mid = 0.5 * (vol_low + vol_high)
        price_mid = bs_call_torch(S0_t, K_t, T_t, r_t, q_t, vol_mid)
        error = price_mid - price
        if torch.abs(error) < tol:
            return vol_mid
        if error > 0:
            vol_high = vol_mid
        else:
            vol_low = vol_mid

    return vol_mid

