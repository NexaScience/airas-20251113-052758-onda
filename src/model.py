"""Model architectures used across experimental variations.

Includes baseline DNN-MFBO, Reliability-Aware variants and a simple Random-Feature
GP surrogate (mf_gp_ei) that follows the same two-fidelity stacking interface.
"""
from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn

# ----------------------------------------------------------------------------- #
#                           Shared building blocks                              #
# ----------------------------------------------------------------------------- #


class FidelityBlock(nn.Module):
    """Standard MLP block with homoscedastic Gaussian likelihood."""

    def __init__(self, in_dim: int, hidden_dim: int = 50):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.log_var = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = self.net(x)
        var = torch.exp(self.log_var) + 1e-6
        return mean, var


# ----------------------------------------------------------------------------- #
#                   Random Fourier Features approximation to GP                 #
# ----------------------------------------------------------------------------- #


class RandomFeatureBlock(nn.Module):
    """Approximate RBF-kernel GP using random Fourier features.

    We keep the random basis functions fixed and learn only the linear weights
    of the Bayesian ridge model. This allows very fast training while providing
    mean/variance estimates similar in form to an exact GP.
    """

    def __init__(self, in_dim: int, num_features: int = 200, lengthscale: float = 1.0):
        super().__init__()
        self.in_dim = in_dim
        self.num_features = num_features
        self.lengthscale = lengthscale

        # Random frequencies and phases are sampled once and kept fixed (not trainable)
        self.register_buffer(
            "omega",
            torch.randn(in_dim, num_features) / lengthscale,
            persistent=False,
        )
        self.register_buffer(
            "phase", 2 * math.pi * torch.rand(num_features), persistent=False
        )

        # Linear weights that will be learnt
        self.w = nn.Parameter(torch.randn(num_features, 1) * 0.01)
        self.log_var = nn.Parameter(torch.zeros(1))  # observation noise

    def _phi(self, x):
        # x: [N, D] -> phi: [N, M]
        proj = x @ self.omega + self.phase  # [N, M]
        return math.sqrt(2.0 / self.num_features) * torch.cos(proj)

    def forward(self, x):
        phi_x = self._phi(x)
        mean = phi_x @ self.w  # [N, 1]
        var = torch.exp(self.log_var) + 1e-6
        return mean, var


# ----------------------------------------------------------------------------- #
#                            Baseline DNN-MFBO                                  #
# ----------------------------------------------------------------------------- #


class DNNMFBO(nn.Module):
    """Two-fidelity baseline surrogate with hard-wired low-fidelity prediction."""

    def __init__(self, x_dim: int, hidden_dim: int = 50):
        super().__init__()
        self.f1 = FidelityBlock(x_dim, hidden_dim)
        self.f2 = FidelityBlock(x_dim + 1, hidden_dim)

    # Low fidelity
    def forward_f1(self, x):
        return self.f1(x)

    # High fidelity
    def forward_f2(self, x):
        with torch.no_grad():
            mean1, _ = self.f1(x)
        inp = torch.cat([x, mean1], dim=-1)
        return self.f2(inp)

    def forward(self, x, fidelity: int):
        return self.forward_f1(x) if fidelity == 0 else self.forward_f2(x)


# ----------------------------------------------------------------------------- #
#                   Reliability-Aware DNN-MFBO (single gate)                    #
# ----------------------------------------------------------------------------- #


class RADNNMFBO(nn.Module):
    """RA-DNN-MFBO with a *single* learnable gate α ∈ (0,1)."""

    def __init__(self, x_dim: int, hidden_dim: int = 50):
        super().__init__()
        self.f1 = FidelityBlock(x_dim, hidden_dim)
        self.alpha_raw = nn.Parameter(torch.zeros(1))  # scalar logit(α)
        self.f2 = FidelityBlock(x_dim + 1, hidden_dim)

    def forward_f1(self, x):
        return self.f1(x)

    def forward_f2(self, x):
        with torch.no_grad():
            mean1, _ = self.f1(x)
        alpha = torch.sigmoid(self.alpha_raw)  # (1,)
        inp = torch.cat([x, alpha * mean1], dim=-1)
        return self.f2(inp)

    def forward(self, x, fidelity: int):
        return self.forward_f1(x) if fidelity == 0 else self.forward_f2(x)


# ----------------------------------------------------------------------------- #
#        Reliability-Aware DNN with *per-dimension* gates (ablation)             #
# ----------------------------------------------------------------------------- #


class RADNNPerDimGates(nn.Module):
    """RA-DNN-MFBO variant with one gate per input dimension (plus bias).

    The low-fidelity prediction is replicated across input dimensions and scaled
    by individual gates α_j.
    """

    def __init__(self, x_dim: int, hidden_dim: int = 50):
        super().__init__()
        self.x_dim = x_dim
        self.f1 = FidelityBlock(x_dim, hidden_dim)
        # One logit parameter per input dimension
        self.alpha_raw = nn.Parameter(torch.zeros(x_dim))
        self.f2 = FidelityBlock(x_dim + x_dim, hidden_dim)

    def forward_f1(self, x):
        return self.f1(x)

    def forward_f2(self, x):
        with torch.no_grad():
            mean1, _ = self.f1(x)  # [N,1]
        alpha = torch.sigmoid(self.alpha_raw)  # [D]
        scaled = (alpha * mean1).repeat(1, self.x_dim)  # replicate across dims
        inp = torch.cat([x, scaled], dim=-1)
        return self.f2(inp)

    def forward(self, x, fidelity: int):
        return self.forward_f1(x) if fidelity == 0 else self.forward_f2(x)


# ----------------------------------------------------------------------------- #
#                 Random-Feature GP surrogate (labelled mf_gp_ei)                #
# ----------------------------------------------------------------------------- #


class MFGPEI(nn.Module):
    """Light-weight surrogate standing in for MF-GP-EI using random features."""

    def __init__(self, x_dim: int, num_features: int = 200, lengthscale: float = 1.0):
        super().__init__()
        self.f1 = RandomFeatureBlock(x_dim, num_features, lengthscale)
        # High fidelity receives low-fidelity mean as additional input
        self.f2 = RandomFeatureBlock(x_dim + 1, num_features, lengthscale)

    def forward_f1(self, x):
        return self.f1(x)

    def forward_f2(self, x):
        with torch.no_grad():
            mean1, _ = self.f1(x)
        inp = torch.cat([x, mean1], dim=-1)
        return self.f2(inp)

    def forward(self, x, fidelity: int):
        return self.forward_f1(x) if fidelity == 0 else self.forward_f2(x)


# ----------------------------------------------------------------------------- #
#                               Factory function                                 #
# ----------------------------------------------------------------------------- #

def _normalise_name(name: str) -> str:
    return name.lower().replace("-", "_")


def build_model(model_name: str, x_dim: int, model_cfg: Dict | None = None):
    model_cfg = model_cfg or {}
    name = _normalise_name(model_name)

    if name in {"dnn_mfbo", "baseline", "baseline_dnn_mfbo"}:
        return DNNMFBO(x_dim=x_dim, **model_cfg)

    if name in {"ra_dnn", "radnn", "ra_dnn_mfbo"}:
        return RADNNMFBO(x_dim=x_dim, **model_cfg)

    if name in {"ra_dnn_per_dim_gates", "radnn_per_dim_gates"}:
        return RADNNPerDimGates(x_dim=x_dim, **model_cfg)

    if name in {"mf_gp_ei", "mfgpei"}:
        return MFGPEI(x_dim=x_dim, **model_cfg)

    raise ValueError(f"Unknown model name: {model_name}")