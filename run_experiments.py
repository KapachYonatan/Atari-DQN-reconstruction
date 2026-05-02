"""Run 3 independent training runs and report the final result.

Seeds used: 42, 123, 456 (fixed for reproducibility across submissions).

Usage:
    python run_experiments.py

    # Shorter runs for debugging:
    python run_experiments.py --total-steps 500000 --eval-episodes 5

Google Colab usage:
    !python run_experiments.py
"""

import argparse
import os

import numpy as np

from src.config import Config
from src.train import train
from src.utils import plot_learning_curve

SEEDS = [42, 123, 456]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 3 independent DQN training runs.")
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--buffer-capacity", type=int, default=None)
    args = parser.parse_args()

    all_final_rewards: list[float] = []

    for i, seed in enumerate(SEEDS, start=1):
        run_id = f"run{i}"
        print(f"\n{'='*60}")
        print(f"  Starting run {i}/3  seed={seed}  run_id={run_id}")
        print(f"{'='*60}\n")

        cfg = Config(seed=seed, run_id=run_id)
        if args.total_steps is not None:
            cfg.total_steps = args.total_steps
        if args.eval_episodes is not None:
            cfg.eval_episodes = args.eval_episodes
        if args.buffer_capacity is not None:
            cfg.buffer_capacity = args.buffer_capacity

        eval_rewards = train(cfg)

        # Final mean reward = mean of the last evaluation checkpoint
        final_reward = eval_rewards[-1] if eval_rewards else float("nan")
        all_final_rewards.append(final_reward)

        # Plot learning curve for this run
        csv_path = os.path.join(cfg.results_dir, run_id, "metrics.csv")
        plot_path = os.path.join(cfg.results_dir, run_id, "learning_curve.png")
        if os.path.exists(csv_path):
            plot_learning_curve(csv_path, plot_path)

    # ------------------------------------------------------------------ #
    # Summary                                                             #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("  FINAL RESULTS")
    print("=" * 60)
    for i, (seed, reward) in enumerate(zip(SEEDS, all_final_rewards), start=1):
        print(f"  Run {i} (seed={seed}):  {reward:.1f}")
    mean = float(np.mean(all_final_rewards))
    std = float(np.std(all_final_rewards))
    print(f"\n  Average over 3 runs:  {mean:.1f}  ±  {std:.1f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
