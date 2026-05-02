"""Evaluation loop for DQN.

Runs a fixed number of full episodes in a separate environment instance and
returns per-episode total rewards.  Intentionally decoupled from the training
loop so it can be tested independently.
"""

from __future__ import annotations

import numpy as np

from src.agent import DQNAgent
from src.config import Config
from src.env import make_env


def evaluate(
    agent: DQNAgent,
    config: Config,
    *,
    seed: int | None = None,
) -> list[float]:
    """Run ``config.eval_episodes`` full episodes and return total rewards.

    Args:
        agent: DQN agent whose online network is used for action selection.
        config: Project-wide configuration instance.
        seed: Optional fixed seed for the evaluation environment.  If None,
            the environment is not explicitly seeded (non-deterministic).

    Returns:
        List of total (unclipped) rewards, one value per episode.
    """
    env = make_env(config, eval_mode=True)
    episode_rewards: list[float] = []

    for ep in range(config.eval_episodes):
        ep_seed = seed + ep if seed is not None else None
        obs, _ = env.reset(seed=ep_seed)
        total_reward = 0.0
        terminated = truncated = False

        while not (terminated or truncated):
            action = agent.select_action(np.array(obs), config.eval_epsilon)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)

        episode_rewards.append(total_reward)

    env.close()
    return episode_rewards
