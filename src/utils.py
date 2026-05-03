"""Shared utilities: seeding, plotting, and checkpoint loading."""

from __future__ import annotations

import csv
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch


def set_seeds(seed: int) -> None:
    """Set all random seeds for reproducibility.

    Args:
        seed: Integer seed value applied to Python, NumPy, PyTorch, and CUDA.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def plot_learning_curve(metrics_csv: str, output_path: str) -> None:
    """Plot evaluation reward vs. training steps from a metrics CSV file.

    The CSV must have columns: step, mean_eval_reward, max_eval_reward.

    Args:
        metrics_csv: Path to the ``metrics.csv`` produced by the training loop.
        output_path: Destination path for the saved PNG figure.
    """
    steps: list[int] = []
    mean_rewards: list[float] = []

    with open(metrics_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["step"]))
            mean_rewards.append(float(row["mean_eval_reward"]))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, mean_rewards, linewidth=1.5, color="steelblue")
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Mean evaluation reward")
    ax.set_title(os.path.basename(os.path.dirname(metrics_csv)))
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[plot] saved → {output_path}")


def load_checkpoint(path: str) -> dict:
    """Load a checkpoint saved by the training loop.

    Args:
        path: Path to the ``.pt`` checkpoint file.

    Returns:
        Dictionary with keys: step, epsilon, model_state_dict,
        target_state_dict, optimizer_state_dict, config.
    """
    return torch.load(path, map_location="cpu", weights_only=False)
