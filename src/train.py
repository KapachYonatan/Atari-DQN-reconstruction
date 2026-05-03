"""Core DQN training loop.

Drives environment interaction, replay buffer filling, agent updates,
periodic evaluation, checkpointing, and CSV metric logging.

Usage (from the project root):
    from src.config import Config
    from src.train import train

    cfg = Config(seed=42, run_id="run1")
    eval_rewards = train(cfg)
"""

from __future__ import annotations

import csv
import os
import time

import numpy as np
import torch

from src.agent import DQNAgent
from src.config import Config
from src.env import make_env
from src.evaluate import evaluate
from src.replay_buffer import ReplayBuffer
from src.utils import load_checkpoint, set_seeds


def train(config: Config) -> list[float]:
    """Run a single full training run.

    Args:
        config: Project-wide configuration instance.

    Returns:
        List of mean evaluation rewards recorded every
        ``config.eval_frequency`` steps.
    """
    set_seeds(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device}  run_id={config.run_id}  seed={config.seed}")

    # ------------------------------------------------------------------ #
    # Setup                                                                #
    # ------------------------------------------------------------------ #
    run_dir = os.path.join(config.results_dir, config.run_id)
    os.makedirs(run_dir, exist_ok=True)
    metrics_path = os.path.join(run_dir, "metrics.csv")

    env = make_env(config, eval_mode=False)
    n_actions: int = env.action_space.n  # type: ignore[attr-defined]

    agent = DQNAgent(n_actions, config, device)
    buffer = ReplayBuffer(
        capacity=config.buffer_capacity,
        state_shape=(config.n_stack, 84, 84),
        device=device,
    )

    # ------------------------------------------------------------------ #
    # Resume                                                               #
    # ------------------------------------------------------------------ #
    start_step = 0
    best_mean_reward = float("-inf")
    if config.resume_from:
        ckpt = load_checkpoint(os.path.join(config.resume_from, "checkpoint_latest.pt"))
        agent.online_net.load_state_dict(ckpt["model_state_dict"])
        agent.target_net.load_state_dict(ckpt["target_state_dict"])
        ckpt_cfg = ckpt.get("config")
        if ckpt_cfg is not None and ckpt_cfg.optimizer != config.optimizer:
            config.optimizer = ckpt_cfg.optimizer
            agent._optimizer = agent._build_optimizer()
        agent._optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = int(ckpt["step"])
        best_mean_reward = float(ckpt.get("best_mean_reward", float("-inf")))
        buf_path = os.path.join(config.resume_from, "buffer_latest.npz")
        if os.path.exists(buf_path):
                try:
                    buffer.load(buf_path)
                except Exception as exc:
                    print(
                        f"[train] failed to load replay buffer from {buf_path}: {exc}. "
                        "Continuing with an empty buffer."
                    )
        else:
            print(f"[train] no buffer file found at {buf_path} — buffer will refill from scratch")
        print(f"[train] resumed from step {start_step:,}  buffer_size={len(buffer):,}")

    # ------------------------------------------------------------------ #
    # CSV logging                                                          #
    # ------------------------------------------------------------------ #
    csv_mode = "a" if start_step > 0 else "w"
    csv_file = open(metrics_path, csv_mode, newline="")
    writer = csv.writer(csv_file)
    if start_step == 0:
        writer.writerow(["step", "mean_eval_reward", "max_eval_reward", "n_eval_episodes"])

    # ------------------------------------------------------------------ #
    # Training loop                                                        #
    # ------------------------------------------------------------------ #
    mean_eval_rewards: list[float] = []

    obs, _ = env.reset(seed=config.seed)
    episode_reward = 0.0
    episode_count = 0
    t_start = time.time()

    for step in range(start_step + 1, config.total_steps + 1):
        # ---- Epsilon schedule ---------------------------------------- #
        fraction = min(step / config.epsilon_decay_steps, 1.0)
        epsilon = config.epsilon_start + fraction * (
            config.epsilon_end - config.epsilon_start
        )

        # ---- Collect one transition ---------------------------------- #
        action = agent.select_action(np.array(obs), epsilon)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        buffer.push(np.array(obs), action, float(reward), np.array(next_obs), done)
        obs = next_obs
        episode_reward += float(reward)

        if done:
            episode_count += 1
            obs, _ = env.reset()
            episode_reward = 0.0

        # ---- Learning update ----------------------------------------- #
        if (
            len(buffer) >= config.min_replay_size
            and step % config.update_frequency == 0
        ):
            agent.learn(buffer)

        # ---- Target network sync ------------------------------------- #
        if step % config.target_update_freq == 0:
            agent.sync_target()

        # ---- Evaluation ---------------------------------------------- #
        if step % config.eval_frequency == 0:
            ep_rewards = evaluate(agent, config, seed=config.seed + step)
            mean_r = float(np.mean(ep_rewards))
            max_r = float(np.max(ep_rewards))
            mean_eval_rewards.append(mean_r)

            writer.writerow([step, f"{mean_r:.2f}", f"{max_r:.2f}", len(ep_rewards)])
            csv_file.flush()

            elapsed = time.time() - t_start
            print(
                f"  step={step:>10,}  ε={epsilon:.3f}"
                f"  eval_mean={mean_r:>8.1f}  eval_max={max_r:>8.1f}"
                f"  episodes={episode_count}  elapsed={elapsed:.0f}s"
            )

            # Save best checkpoint
            if mean_r > best_mean_reward:
                best_mean_reward = mean_r
                _save_checkpoint(agent, config, step, epsilon, best_mean_reward, run_dir, tag="best")

        # ---- Periodic checkpoint ------------------------------------- #
        if step % config.save_frequency == 0:
            _save_checkpoint(agent, config, step, epsilon, best_mean_reward, run_dir, tag="latest", buffer=buffer)

    env.close()
    csv_file.close()

    print(f"[train] done  best_mean_eval={best_mean_reward:.1f}")
    return mean_eval_rewards


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _save_checkpoint(
    agent: DQNAgent,
    config: Config,
    step: int,
    epsilon: float,
    best_mean_reward: float,
    run_dir: str,
    tag: str,
    buffer: ReplayBuffer | None = None,
) -> None:
    """Save online net, target net, optimizer state, and optionally the replay buffer."""
    path = os.path.join(run_dir, f"checkpoint_{tag}.pt")
    torch.save(
        {
            "step": step,
            "epsilon": epsilon,
            "best_mean_reward": best_mean_reward,
            "model_state_dict": agent.online_net.state_dict(),
            "target_state_dict": agent.target_net.state_dict(),
            "optimizer_state_dict": agent._optimizer.state_dict(),
            "config": config,
        },
        path,
    )
    if buffer is not None:
        buffer.save(os.path.join(run_dir, "buffer_latest.npz"))
