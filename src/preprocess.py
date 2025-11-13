"""Dataset loading and common preprocessing utilities.

All experimental variations *must* rely on this file for data access so that preprocessing
remains identical across runs. Implements simple synthetic two–fidelity benchmarks that
are inexpensive yet sufficiently rich to validate the complete experimental pipeline.
"""
from __future__ import annotations

import math
import random
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# ----------------------------------------------------------------------------- #
#                               Synthetic functions                              #
# ----------------------------------------------------------------------------- #


def _branin(x: torch.Tensor) -> torch.Tensor:
    """High-fidelity Branin function (2-D)."""
    x1, x2 = x[..., 0], x[..., 1]
    a = 1.0
    b = 5.1 / (4 * math.pi ** 2)
    c = 5 / math.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * math.pi)
    return a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * torch.cos(x1) + s


def _low_fidelity_branin(x: torch.Tensor) -> torch.Tensor:
    """Ad-hoc cheaper Branin variant – smooths and adds bias/artefacts."""
    return 0.5 * _branin(x) + 10.0 * torch.sin(x[..., 0])


def _levy(x: torch.Tensor) -> torch.Tensor:
    """High-fidelity 2-D Levy function."""
    # Standard Levy implementation adapted for two dimensions
    w = 1 + (x - 1) / 4.0  # shape (..., 2)
    term1 = torch.sin(math.pi * w[..., 0]) ** 2
    term3 = (w[..., 1] - 1) ** 2 * (1 + torch.sin(2 * math.pi * w[..., 1]) ** 2)
    term2 = (w[..., 0] - 1) ** 2 * (
        1 + 10 * torch.sin(math.pi * w[..., 0] + 1) ** 2
    )
    return term1 + term2 + term3


def _low_fidelity_levy(x: torch.Tensor) -> torch.Tensor:
    """Low-fidelity Levy – down-scaled + periodic bias."""
    return 0.4 * _levy(x) + 5.0 * torch.cos(0.5 * x[..., 0])


# Mapping for convenience ------------------------------------------------------- #
_SYNTHETIC_FUNCS = {
    "branin": (_low_fidelity_branin, _branin, (-5.0, 10.0, 0.0, 15.0)),
    "levy": (_low_fidelity_levy, _levy, (-10.0, 10.0, -10.0, 10.0)),
}


# ----------------------------------------------------------------------------- #
#                           Synthetic dataset wrapper                            #
# ----------------------------------------------------------------------------- #


class SyntheticFunctionDataset(Dataset):
    """Generates random (x, y) pairs on-the-fly for a given fidelity."""

    def __init__(
        self,
        n_samples: int,
        fidelity: int,
        function_name: str = "branin",
        noise_std: float = 0.0,
    ):
        super().__init__()
        if function_name.lower() not in _SYNTHETIC_FUNCS:
            raise ValueError(f"Unsupported synthetic function: {function_name}")

        low_f, high_f, domain = _SYNTHETIC_FUNCS[function_name.lower()]
        self.fidelity = fidelity  # 0 = low, 1 = high
        self.noise_std = noise_std

        # Sample uniformly in the function-specific domain
        x1_min, x1_max, x2_min, x2_max = domain
        x1 = torch.FloatTensor(n_samples).uniform_(x1_min, x1_max)
        x2 = torch.FloatTensor(n_samples).uniform_(x2_min, x2_max)
        self.x = torch.stack([x1, x2], dim=-1)

        if fidelity == 0:
            self.y = low_f(self.x)
        else:
            self.y = high_f(self.x)

        if noise_std > 0:
            self.y += noise_std * torch.randn_like(self.y)

        # Normalise outputs to [0, 1] for numerical stability ------------------ #
        y_min, y_max = self.y.min(), self.y.max()
        self.y = (self.y - y_min) / (y_max - y_min + 1e-8)

        # Scale inputs to [0, 1] as well for both axes ------------------------- #
        self.x[..., 0] = (self.x[..., 0] - x1_min) / (x1_max - x1_min)
        self.x[..., 1] = (self.x[..., 1] - x2_min) / (x2_max - x2_min)

    # ------------------------------ PyTorch API ---------------------------------- #

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return {
            "x": self.x[idx],
            "y": self.y[idx],
            "fidelity": torch.tensor(self.fidelity, dtype=torch.long),
        }


# ----------------------------------------------------------------------------- #
#                               Public loader                                   #
# ----------------------------------------------------------------------------- #

def load_dataset(dataset_cfg: Dict, batch_size: int = 32) -> Tuple[Dict, int]:
    """Create train/val/test DataLoaders per fidelity.

    The returned dictionary has structure:
        {
            "train": {0: DataLoader, 1: DataLoader},
            "val":   {0: DataLoader, 1: DataLoader},
            "test":  {0: DataLoader, 1: DataLoader},
        }
    and the second return value is the input dimensionality (always 2 here).
    """

    name = dataset_cfg["name"].lower()

    if name == "synthetic":
        function_name = dataset_cfg.get("function", "branin")
        n_low = int(dataset_cfg.get("n_samples_low", 500))
        n_high = int(dataset_cfg.get("n_samples_high", 500))
        noise_std = float(dataset_cfg.get("noise_std", 0.0))

        ds_low = SyntheticFunctionDataset(
            n_low, fidelity=0, function_name=function_name, noise_std=noise_std
        )
        ds_high = SyntheticFunctionDataset(
            n_high, fidelity=1, function_name=function_name, noise_std=noise_std
        )

        # 80 / 10 / 10 random split per fidelity -------------------------------- #
        def _split(ds):
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

        splits_low = _split(ds_low)
        splits_high = _split(ds_high)

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

    # ------------------------------------------------------------------------- #
    # If we ever reach here the user requested an unsupported dataset ---------- #
    raise NotImplementedError(f"Dataset '{name}' is not implemented in preprocess.py.")