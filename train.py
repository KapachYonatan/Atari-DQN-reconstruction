"""CLI entry point for a single DQN training run.

Example usage:
    # Default config (seed=42, run_id="run1", 10M steps):
    python train.py

    # Custom run:
    python train.py --seed 123 --run-id run2 --total-steps 5000000

    # Smoke test (fast, ~5 min on GPU):
    python train.py --seed 0 --run-id smoke --total-steps 50000 --eval-episodes 1

Google Colab usage:
    !python train.py --seed 42 --run-id run1
"""

import argparse

from src.config import Config
from src.train import train


def _parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Train a DQN agent on Atari.")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", type=str, default="run1")
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--buffer-capacity", type=int, default=None)
    parser.add_argument("--min-replay-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--results-dir", type=str, default="results")

    args = parser.parse_args()

    cfg = Config(seed=args.seed, run_id=args.run_id, results_dir=args.results_dir)
    if args.total_steps is not None:
        cfg.total_steps = args.total_steps
    if args.eval_episodes is not None:
        cfg.eval_episodes = args.eval_episodes
    if args.buffer_capacity is not None:
        cfg.buffer_capacity = args.buffer_capacity
    if args.min_replay_size is not None:
        cfg.min_replay_size = args.min_replay_size
    if args.lr is not None:
        cfg.lr = args.lr

    return cfg


if __name__ == "__main__":
    cfg = _parse_args()
    train(cfg)
