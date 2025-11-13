"""Dataset loading and common preprocessing utilities.

All experimental variations *must* rely on this file for data access so that preprocessing
remains identical across runs.
"""
from __future__ import annotations

import math
import random
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# ----------------------------------------------------------------------------- #
#                               Synthetic datasets                              #
# ----------------------------------------------------------------------------- #


def _branin(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., 0], x[..., 1]
    a = 1.0
    b = 5.1 / (4 * math.pi ** 2)
    c = 5 / math.pi
    r = 6.0
    s = 10.0
    t = 1 / (8 * math.pi)
    return a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * torch.cos(x1) + s


def _low_fidelity_branin(x: torch.Tensor) -> torch.Tensor:
    # Simple smooth transformation to emulate low-fidelity artefacts
    return _branin(x) * 0.5 + 10 * torch.sin(x[..., 0])


class SyntheticFunctionDataset(Dataset):
    def __init__(self, n_samples: int, fidelity: int, function_name: str = "branin", noise_std: float = 0.0):
        super().__init__()
        self.fidelity = fidelity  # 0 = low, 1 = high
        self.function_name = function_name
        self.noise_std = noise_std

        # Random sampling within domain for Branin problems
        x1 = torch.FloatTensor(n_samples).uniform_(-5, 10)
        x2 = torch.FloatTensor(n_samples).uniform_(0, 15)
        self.x = torch.stack([x1, x2], dim=-1)

        if function_name == "branin":
            if fidelity == 0:
                self.y = _low_fidelity_branin(self.x)
            else:
                self.y = _branin(self.x)
        else:
            raise ValueError(f"Unknown synthetic function: {function_name}")

        if noise_std > 0:
            self.y += noise_std * torch.randn_like(self.y)

        # Ensure y has shape [N, 1] for broadcasting convenience
        if self.y.ndim == 1:
            self.y = self.y.unsqueeze(-1)

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return {"x": self.x[idx], "y": self.y[idx], "fidelity": self.fidelity}


# ----------------------------------------------------------------------------- #
#                               Loader facade                                   #
# ----------------------------------------------------------------------------- #

def load_dataset(dataset_cfg: Dict, batch_size: int = 32) -> Tuple[Dict, int]:
    """Load dataset according to configuration and return dataloaders per fidelity.

    Returns a dict of the form {split: {fid: DataLoader}} and the input dimension.
    """

    name = dataset_cfg["name"].lower()

    if name == "synthetic":
        function_name = dataset_cfg.get("function", "branin")
        n_low = dataset_cfg.get("n_samples_low", 200)
        n_high = dataset_cfg.get("n_samples_high", 200)
        noise_low = dataset_cfg.get("noise_low", 0.0)
        noise_high = dataset_cfg.get("noise_high", 0.0)

        ds_low = SyntheticFunctionDataset(
            n_low, fidelity=0, function_name=function_name, noise_std=noise_low
        )
        ds_high = SyntheticFunctionDataset(
            n_high, fidelity=1, function_name=function_name, noise_std=noise_high
        )

        # Simple 80/10/10 split per fidelity
        def split_dataset(ds):
            n = len(ds)
            idxs = list(range(n))
            random.shuffle(idxs)
            n_train = int(0.8 * n)
            n_val = int(0.1 * n)
            train_idx = idxs[:n_train]
            val_idx = idxs[n_train : n_train + n_val]
            test_idx = idxs[n_train + n_val :]
            return (
                torch.utils.data.Subset(ds, train_idx),
                torch.utils.data.Subset(ds, val_idx),
                torch.utils.data.Subset(ds, test_idx),
            )

        splits_low = split_dataset(ds_low)
        splits_high = split_dataset(ds_high)

        dataloaders = {
            "train": {
                0: DataLoader(splits_low[0], batch_size=batch_size, shuffle=True),
                1: DataLoader(splits_high[0], batch_size=batch_size, shuffle=True),
            },
            "val": {
                0: DataLoader(splits_low[1], batch_size=batch_size, shuffle=False),
                1: DataLoader(splits_high[1], batch_size=batch_size, shuffle=False),
            },
            "test": {
                0: DataLoader(splits_low[2], batch_size=batch_size, shuffle=False),
                1: DataLoader(splits_high[2], batch_size=batch_size, shuffle=False),
            },
        }
        x_dim = ds_low[0]["x"].numel()
        return dataloaders, x_dim

    # If other dataset names are provided, raise explicit error.
    raise NotImplementedError(
        f"Dataset '{name}' is not implemented in this experimental framework."
    )