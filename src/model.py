"""Model architectures used across experimental variations.

This file contains fully-implemented surrogates for all experiment variants:
    • DNN-MFBO baseline (hard-wired low-fidelity mean as extra feature)
    • RA-DNN-MFBO (learnable gate with KL regularisation)
    • RA-DNN-MFBO-noKL (same architecture, KL coefficient will be 0 in the config)
    • RA-DNN-MFBO-fixed-alpha1 (degenerates to baseline)
    • RA-DNN-MFBO-fixed-alpha0 (ignores low-fidelity feature altogether)
Every model exposes .forward_f1 and .forward_f2 for low & high fidelities so that the
training loop in src/train.py remains identical across variants.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

# ----------------------------------------------------------------------------- #
#                             Shared building block                              #
# ----------------------------------------------------------------------------- #


class FidelityBlock(nn.Module):
    """Simple 2-layer MLP producing a Gaussian mean & homoscedastic variance."""

    def __init__(self, in_dim: int, hidden_dim: int = 50):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.log_var = nn.Parameter(torch.zeros(1))  # log σ² – unconstrained

    def forward(self, x):
        mean = self.net(x)
        var = torch.exp(self.log_var) + 1e-6  # numeric stability
        return mean, var


# ----------------------------------------------------------------------------- #
#                          Baseline: DNN-MFBO surrogate                          #
# ----------------------------------------------------------------------------- #


class DNNMFBO(nn.Module):
    """Original stacked surrogate from the DNN-MFBO paper (two fidelities)."""

    def __init__(self, x_dim: int, hidden_dim: int = 50):
        super().__init__()
        self.f1 = FidelityBlock(x_dim, hidden_dim)
        self.f2 = FidelityBlock(x_dim + 1, hidden_dim)

    # ------------------------- Low-fidelity ----------------------------------- #
    def forward_f1(self, x):
        return self.f1(x)

    # ------------------------ High-fidelity ----------------------------------- #
    def forward_f2(self, x):
        with torch.no_grad():
            mean1, _ = self.f1(x)
        inp = torch.cat([x, mean1], dim=-1)
        return self.f2(inp)

    def forward(self, x, fidelity: int):
        return self.forward_f1(x) if fidelity == 0 else self.forward_f2(x)


# ----------------------------------------------------------------------------- #
#                     Reliability-Aware DNN-MFBO surrogate                       #
# ----------------------------------------------------------------------------- #


class RADNNMFBO(nn.Module):
    """RA-DNN-MFBO with learnable gate α ∈ (0, 1)."""

    def __init__(self, x_dim: int, hidden_dim: int = 50):
        super().__init__()
        self.f1 = FidelityBlock(x_dim, hidden_dim)
        # α parameterised in logit space so unconstrained optimisation in ℝ
        self.alpha_raw = nn.Parameter(torch.zeros(1))
        self.f2 = FidelityBlock(x_dim + 1, hidden_dim)

    def forward_f1(self, x):
        return self.f1(x)

    def forward_f2(self, x):
        with torch.no_grad():
            mean1, _ = self.f1(x)
        alpha = torch.sigmoid(self.alpha_raw)
        inp = torch.cat([x, alpha * mean1], dim=-1)
        return self.f2(inp)

    def forward(self, x, fidelity: int):
        return self.forward_f1(x) if fidelity == 0 else self.forward_f2(x)


# ----------------------------------------------------------------------------- #
#                          Fixed-α ablation models                               #
# ----------------------------------------------------------------------------- #


class FixedAlphaOneDNNMFBO(DNNMFBO):
    """Uses α = 1 – identical to the baseline but kept for clarity."""

    # Inherits everything from baseline; no modifications necessary.
    pass


class FixedAlphaZeroDNNMFBO(nn.Module):
    """α = 0 → ignore low-fidelity mean for the high-fidelity network."""

    def __init__(self, x_dim: int, hidden_dim: int = 50):
        super().__init__()
        self.f1 = FidelityBlock(x_dim, hidden_dim)
        # Note: f2 takes *only* x (no extra feature)
        self.f2 = FidelityBlock(x_dim, hidden_dim)

    def forward_f1(self, x):
        return self.f1(x)

    def forward_f2(self, x):
        # Still compute mean1 so computational graph / call signature identical
        # but deliberately *ignore* it when forming inputs to f2.
        _mean1, _ = self.f1(x)  # noqa: F841 – value intentionally unused
        return self.f2(x)

    def forward(self, x, fidelity: int):
        return self.forward_f1(x) if fidelity == 0 else self.forward_f2(x)


# ----------------------------------------------------------------------------- #
#                               Factory function                                 #
# ----------------------------------------------------------------------------- #

def build_model(model_name: str, x_dim: int, model_cfg: Dict | None = None):
    """Instantiate a model by name so YAML stays human-friendly."""

    model_cfg = model_cfg or {}
    name = model_name.lower()

    if name in {"dnn_mfbo", "baseline"}:
        return DNNMFBO(x_dim=x_dim, **model_cfg)

    if name in {"radnn", "radnn_mfbo", "ra_dnn_mfbo"}:
        return RADNNMFBO(x_dim=x_dim, **model_cfg)

    if name in {"radnn_fixed_alpha1", "fixed_alpha1"}:
        return FixedAlphaOneDNNMFBO(x_dim=x_dim, **model_cfg)

    if name in {"radnn_fixed_alpha0", "fixed_alpha0"}:
        return FixedAlphaZeroDNNMFBO(x_dim=x_dim, **model_cfg)

    raise ValueError(f"Unknown model name: {model_name}")