import argparse
import json
import os
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src import model as model_lib
from src import preprocess as preprocess_lib

# ------------------------------- Utility functions -------------------------------- #

def set_deterministic(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def nll_gaussian(y, mean, var):
    """Negative log-likelihood of isotropic Gaussian with diagonal variance."""
    return 0.5 * torch.log(var) + 0.5 * (y - mean) ** 2 / var


def nrmse(y_true: torch.Tensor, y_pred: torch.Tensor):
    rmse = torch.sqrt(F.mse_loss(y_pred, y_true))
    return (rmse / (y_true.max() - y_true.min())).item()


# ------------------------------- Training routine --------------------------------- #


def train_single_run(run_config: dict, results_dir: Path):
    results_dir.mkdir(parents=True, exist_ok=True)
    images_dir = results_dir / "images"
    images_dir.mkdir(exist_ok=True)

    # 1. Deterministic behaviour
    seed = run_config.get("seed", 42)
    set_deterministic(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Data
    dataloaders, x_dim = preprocess_lib.load_dataset(run_config["dataset"], batch_size=run_config["training"]["batch_size"])

    # 3. Model
    model = model_lib.build_model(run_config["model_name"], x_dim=x_dim, model_cfg=run_config.get("model", {})).to(device)

    # 4. Optimiser & misc
    lr = run_config["training"].get("lr", 1e-3)
    epochs = run_config["training"].get("epochs", 100)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loss_hist, val_loss_hist, val_nrmse_hist = [], [], []
    best_val_nrmse = float("inf")

    for epoch in tqdm(range(1, epochs + 1), desc=f"{run_config['run_id']}"):
        model.train()
        epoch_losses = []

        # --- iterate low then high fidelity to guarantee low-forward availability ---
        for fidelity in [0, 1]:
            for batch in dataloaders["train"][fidelity]:
                x = batch["x"].to(device)
                y = batch["y"].to(device)

                if fidelity == 0:
                    mean, var = model.forward_f1(x)
                    loss = nll_gaussian(y, mean, var).mean()
                else:  # high fidelity
                    mean, var = model.forward_f2(x)
                    loss = nll_gaussian(y, mean, var).mean()
                    # Add KL on alpha if present (RADNN)
                    if hasattr(model, "alpha_raw"):
                        alpha = torch.sigmoid(model.alpha_raw)
                        kl_alpha = -torch.distributions.Beta(1.0, 1.0).log_prob(alpha)
                        loss = loss + run_config.get("kl_coeff", 0.01) * kl_alpha

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())

        train_loss_hist.append(np.mean(epoch_losses))

        # ---------------- Validation ----------------
        model.eval()
        with torch.no_grad():
            val_losses, preds, targets = [], [], []
            for batch in dataloaders["val"][1]:  # high fidelity only for val metrics
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                mean, var = model.forward_f2(x)
                loss = nll_gaussian(y, mean, var).mean()
                val_losses.append(loss.item())
                preds.append(mean.cpu())
                targets.append(y.cpu())
        val_loss = np.mean(val_losses)
        val_loss_hist.append(val_loss)

        preds = torch.cat(preds, dim=0)
        targets = torch.cat(targets, dim=0)
        val_nrmse_val = nrmse(targets, preds)
        val_nrmse_hist.append(val_nrmse_val)
        best_val_nrmse = min(best_val_nrmse, val_nrmse_val)

    # ---------------- Testing ----------------
    model.eval()
    with torch.no_grad():
        preds, targets = [], []
        for batch in dataloaders["test"][1]:
            x, y = batch["x"].to(device), batch["y"].to(device)
            mean, _ = model.forward_f2(x)
            preds.append(mean.cpu())
            targets.append(y.cpu())
    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)
    test_nrmse_val = nrmse(targets, preds)

    # ---------------- Save artefacts ----------------
    torch.save(model.state_dict(), results_dir / "model.pt")

    metrics = {
        "run_id": run_config["run_id"],
        "best_val_nrmse": best_val_nrmse,
        "test_nrmse": test_nrmse_val,
        "train_loss_hist": train_loss_hist,
        "val_loss_hist": val_loss_hist,
        "val_nrmse_hist": val_nrmse_hist,
        "config": run_config,
    }

    with open(results_dir / "results.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ------------------ Figures --------------------
    sns.set(style="whitegrid", font_scale=1.3)

    # 1. Training / validation loss curves
    plt.figure(figsize=(6, 4))
    plt.plot(train_loss_hist, label="Train NLL")
    plt.plot(val_loss_hist, label="Val NLL")
    plt.xlabel("Epoch")
    plt.ylabel("Negative Log-Likelihood")
    plt.title(f"Training Curve – {run_config['run_id']}")
    plt.legend()
    plt.annotate(f"Final Val = {val_loss_hist[-1]:.3f}", xy=(len(val_loss_hist) - 1, val_loss_hist[-1]))
    plt.tight_layout()
    plt.savefig(images_dir / f"training_loss_{run_config['run_id']}.pdf", bbox_inches="tight")
    plt.close()

    # 2. nRMSE curve
    plt.figure(figsize=(6, 4))
    plt.plot(val_nrmse_hist, label="Val nRMSE", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("nRMSE")
    plt.title(f"Validation nRMSE – {run_config['run_id']}")
    plt.annotate(f"Best = {best_val_nrmse:.3f}", xy=(np.argmin(val_nrmse_hist), best_val_nrmse))
    plt.legend()
    plt.tight_layout()
    plt.savefig(images_dir / f"nrmse_{run_config['run_id']}.pdf", bbox_inches="tight")
    plt.close()

    # Print JSON summary to stdout for structured logging
    print(json.dumps({"run_id": run_config["run_id"], "best_val_nrmse": best_val_nrmse, "test_nrmse": test_nrmse_val}))


# ---------------------------------------------------------------------------------- #


def cli_main():
    parser = argparse.ArgumentParser(description="Run a single experiment variation.")
    parser.add_argument("--run-config", type=str, required=True, help="Path to a JSON file containing the run configuration.")
    parser.add_argument("--results-dir", type=str, required=True, help="Directory to write outputs to.")
    args = parser.parse_args()

    with open(args.run_config, "r") as f:
        run_cfg = json.load(f)

    train_single_run(run_cfg, Path(args.results_dir))


if __name__ == "__main__":
    cli_main()