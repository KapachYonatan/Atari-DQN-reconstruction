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
    _defaults = Config()
    parser = argparse.ArgumentParser(description="Train a DQN agent on Atari.")

    parser.add_argument("--seed",             type=int,   default=_defaults.seed)
    parser.add_argument("--run-id",           type=str,   default=_defaults.run_id)
    parser.add_argument("--total-steps",      type=int,   default=_defaults.total_steps)
    parser.add_argument("--eval-episodes",    type=int,   default=_defaults.eval_episodes)
    parser.add_argument("--buffer-capacity",  type=int,   default=_defaults.buffer_capacity)
    parser.add_argument("--min-replay-size",  type=int,   default=_defaults.min_replay_size)
    parser.add_argument("--lr",               type=float, default=_defaults.lr)
    parser.add_argument("--optimizer",        type=str,   default=_defaults.optimizer, choices=["rmsprop", "adam"])
    parser.add_argument("--rmsprop-alpha",    type=float, default=_defaults.rmsprop_alpha)
    parser.add_argument("--rmsprop-eps",      type=float, default=_defaults.rmsprop_eps)
    parser.add_argument("--reward-mode",      type=str,   default=_defaults.reward_mode,
                        choices=["clip", "normalize", "none"],
                        help="Reward scaling during training: clip={-1,0,+1}, normalize=/max_abs, none=raw.")
    parser.add_argument("--reward-max-abs",   type=float, default=_defaults.reward_max_abs,
                        help="Divisor for --reward-mode=normalize (default: 1000.0).")
    parser.add_argument("--results-dir",      type=str,   default=_defaults.results_dir)
    parser.add_argument("--resume",           type=str,   default=_defaults.resume_from,
                        help="Path to a run directory to resume from (e.g. results/run1).")

    args = parser.parse_args()

    return Config(
        seed=args.seed,
        run_id=args.run_id,
        total_steps=args.total_steps,
        eval_episodes=args.eval_episodes,
        buffer_capacity=args.buffer_capacity,
        min_replay_size=args.min_replay_size,
        lr=args.lr,
        optimizer=args.optimizer,
        rmsprop_alpha=args.rmsprop_alpha,
        rmsprop_eps=args.rmsprop_eps,
        reward_mode=args.reward_mode,
        reward_max_abs=args.reward_max_abs,
        results_dir=args.results_dir,
        resume_from=args.resume,
    )


if __name__ == "__main__":
    cfg = _parse_args()
    train(cfg)
