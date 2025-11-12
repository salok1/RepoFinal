"""Dataset preparation utilities for Heston parameter learning."""
from __future__ import annotations

from typing import Dict, Tuple

import torch

from bs_iv import iv_call_brent_torch


def make_dataset_from_csv(df, r: float, q: float = 0.0) -> Dict[str, torch.Tensor]:
    """Convert a pandas DataFrame into tensors with implied vol features."""
    S0 = torch.tensor(df["S0"].values, dtype=torch.float64)
    K = torch.tensor(df["K"].values, dtype=torch.float64)
    T = torch.tensor(df["T"].values, dtype=torch.float64)
    C = torch.tensor(df["C_mkt"].values, dtype=torch.float64)

    sigmas = []
    for price, s0_val, k_val, t_val in zip(C, S0, K, T):
        sigma = iv_call_brent_torch(price, s0_val, k_val, t_val, r, q)
        sigmas.append(sigma)
    sigma_tensor = torch.stack(sigmas)

    X = torch.stack(
        [
            torch.clamp(S0 / K, min=1e-6),
            T,
            sigma_tensor,
        ],
        dim=1,
    )
    y = C
    r_tensor = torch.full_like(S0, fill_value=r, dtype=torch.float64)
    data = {"S0": S0, "K": K, "T": T, "r": r_tensor, "X": X, "y": y}
    return data


def split_train_val(
    data: Dict[str, torch.Tensor],
    val_frac: float = 0.2,
    seed: int = 123,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Deterministically split dataset into train/validation parts."""
    total = data["X"].shape[0]
    if total < 2:
        raise ValueError("Need at least two samples to perform a train/validation split.")

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(total, generator=generator)
    n_val = max(1, int(round(total * val_frac)))
    if n_val >= total:
        n_val = total - 1

    val_idx = permutation[:n_val]
    train_idx = permutation[n_val:]

    def _gather(idx: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {key: value[idx] for key, value in data.items()}

    return _gather(train_idx), _gather(val_idx)

