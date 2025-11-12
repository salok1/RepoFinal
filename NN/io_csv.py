"""Utilities for reading market option CSV files."""
from __future__ import annotations

from typing import List

import pandas as pd


EXPECTED_COLUMNS: List[str] = ["S0", "K", "C_mkt", "T"]


def read_market_csv(path: str) -> pd.DataFrame:
    """Read a CSV file containing option data and validate required headers."""
    df = pd.read_csv(path)
    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"CSV file {path} is missing required columns: {missing}")
    return df[EXPECTED_COLUMNS].copy()

