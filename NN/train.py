"""Training script for Heston parameter neural network."""
from __future__ import annotations

import argparse
from typing import Dict

import torch
from torch.optim import AdamW

from dataset import make_dataset_from_csv, split_train_val
from io_csv import read_market_csv
from model import HestonParamNet, price_with_params, rmse_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NN to predict Heston parameters.")
    parser.add_argument("--csv", type=str, required=True, help="Path to option market CSV file.")
    parser.add_argument("--r", type=float, required=True, help="Constant risk-free rate.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--alpha", type=float, default=1.5, help="Carr-Madan damping factor.")
    parser.add_argument("--nfft", type=int, default=2 ** 12, help="Number of FFT points.")
    parser.add_argument("--eta", type=float, default=0.25, help="Frequency spacing for FFT.")
    parser.add_argument("--val_frac", type=float, default=0.2, help="Validation fraction.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--save", type=str, default=None, help="Optional path to save model weights.")
    return parser.parse_args()


def _print_validation_table(val_data: Dict[str, torch.Tensor], preds: torch.Tensor) -> None:
    n_rows = min(5, val_data["y"].shape[0])
    header = f"{'S0':>8} {'K':>8} {'T':>6} {'C_mkt':>12} {'C_pred':>12} {'abs_err':>10} {'rel_err%':>10}"
    print(header)
    for idx in range(n_rows):
        s0 = val_data["S0"][idx].item()
        strike = val_data["K"][idx].item()
        maturity = val_data["T"][idx].item()
        c_true = val_data["y"][idx].item()
        c_pred = preds[idx].item()
        abs_err = abs(c_pred - c_true)
        rel_err = abs_err / max(abs(c_true), 1e-6) * 100.0
        print(
            f"{s0:8.3f} {strike:8.3f} {maturity:6.3f} {c_true:12.6f} "
            f"{c_pred:12.6f} {abs_err:10.6f} {rel_err:10.3f}"
        )


def main() -> None:
    args = parse_args()
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(args.seed)

    df = read_market_csv(args.csv)
    data = make_dataset_from_csv(df, r=args.r)
    train_data, val_data = split_train_val(data, val_frac=args.val_frac, seed=args.seed)

    model = HestonParamNet()
    optimizer = AdamW(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        train_params = model(train_data["X"])
        train_prices = price_with_params(
            train_data["S0"],
            train_data["K"],
            train_data["T"],
            train_data["r"],
            train_params,
            alpha=args.alpha,
            Nfft=args.nfft,
            eta=args.eta,
        )
        train_loss = rmse_loss(train_prices, train_data["y"])
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_params = model(val_data["X"])
            val_prices = price_with_params(
                val_data["S0"],
                val_data["K"],
                val_data["T"],
                val_data["r"],
                val_params,
                alpha=args.alpha,
                Nfft=args.nfft,
                eta=args.eta,
            )
            val_loss = rmse_loss(val_prices, val_data["y"])

        print(f"Epoch {epoch:03d} | Train RMSE {train_loss.item():.6f} | Val RMSE {val_loss.item():.6f}")
        _print_validation_table(val_data, val_prices)

    if args.save:
        torch.save(model.state_dict(), args.save)
        print(f"Model weights saved to {args.save}")


if __name__ == "__main__":
    main()

