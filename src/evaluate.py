import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ------------------------------- Evaluation/visualisation ------------------------- #


def collect_results(results_root: Path):
    summaries = []
    for run_dir in results_root.iterdir():
        if not run_dir.is_dir():
            continue
        res_file = run_dir / "results.json"
        if res_file.exists():
            with open(res_file, "r") as f:
                data = json.load(f)
                summaries.append({
                    "run_id": data["run_id"],
                    "best_val_nrmse": data["best_val_nrmse"],
                    "test_nrmse": data["test_nrmse"],
                })
    return pd.DataFrame(summaries)


def generate_comparison_figures(df: pd.DataFrame, results_dir: Path):
    images_dir = results_dir / "images"
    images_dir.mkdir(exist_ok=True)

    sns.set(style="whitegrid", font_scale=1.4)

    # 1. Bar chart of test nRMSE
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(x="run_id", y="test_nrmse", data=df, palette="viridis")
    for i, p in enumerate(ax.patches):
        val = df.loc[i, "test_nrmse"]
        ax.annotate(f"{val:.3f}", (p.get_x() + p.get_width() / 2., val), ha='center', va='bottom')
    plt.ylabel("Test nRMSE (lower is better)")
    plt.title("Test nRMSE comparison across runs")
    plt.tight_layout()
    plt.savefig(images_dir / "test_nrmse_comparison.pdf", bbox_inches="tight")
    plt.close()

    # 2. Sorted line plot for clarity (optional)
    df_sorted = df.sort_values("test_nrmse")
    plt.figure(figsize=(8, 5))
    plt.plot(df_sorted["run_id"], df_sorted["test_nrmse"], marker="o")
    for i, val in enumerate(df_sorted["test_nrmse"].values):
        plt.annotate(f"{val:.3f}", (i, val))
    plt.xlabel("Run")
    plt.ylabel("Test nRMSE")
    plt.title("Test nRMSE per run (sorted)")
    plt.tight_layout()
    plt.savefig(images_dir / "test_nrmse_sorted.pdf", bbox_inches="tight")
    plt.close()


# ------------------------------ Main entry-point ---------------------------------- #


def main(results_root: str):
    root = Path(results_root)
    df = collect_results(root)
    generate_comparison_figures(df, root)

    # Print summary JSON to stdout
    print(df.to_json(orient="records"))


# ---------------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate and visualise results across runs.")
    parser.add_argument("--results-dir", type=str, required=True)
    args = parser.parse_args()
    main(args.results_dir)