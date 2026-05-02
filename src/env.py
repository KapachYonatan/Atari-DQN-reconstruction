"""Environment factory for Atari DQN.

Builds the preprocessed Crazy Climber environment using only standard
Gymnasium wrappers, reproducing the DQN paper preprocessing pipeline:

    1. AtariPreprocessing  — noop reset (1–30), frame-skip=4, max-pool over
                             last 2 frames, grayscale, resize to 84×84.
    2. FrameStackObservation — stack 4 consecutive frames → shape (4, 84, 84).
    3. TransformReward      — clip rewards to {-1, 0, +1}.

The factory is intentionally thin: it only wires wrappers and exposes a
callable so training and evaluation code can each create their own isolated
env instance.
"""

import ale_py  # registers ALE environments with Gymnasium
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation, TransformReward

from src.config import Config


def _clip_reward(reward: float) -> float:
    """Clip a reward value to {-1, 0, +1}."""
    if reward > 0:
        return 1.0
    if reward < 0:
        return -1.0
    return 0.0


def make_env(config: Config, *, eval_mode: bool = False) -> gym.Env:
    """Create and wrap the Atari environment.

    Args:
        config: Project-wide configuration instance.
        eval_mode: When True the reward is NOT clipped (so evaluation scores
            reflect true game performance) and episode life-loss termination
            is disabled.  Training uses ``eval_mode=False``.

    Returns:
        A fully wrapped ``gymnasium.Env`` with observation shape
        ``(n_stack, 84, 84)`` and dtype ``uint8``.
    """
    # frameskip=1 disables the env's built-in frame-skip so that
    # AtariPreprocessing can apply its own (frame_skip=4) without conflict.
    env = gym.make(config.env_id, frameskip=1)

    # Paper preprocessing: noop reset, frame-skip=4, max-pool, grayscale, 84×84.
    # scale_obs=False keeps observations as uint8 (we normalise in the buffer).
    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        grayscale_obs=True,
        scale_obs=False,
    )

    # Stack 4 frames → observation shape (4, 84, 84).
    env = FrameStackObservation(env, stack_size=config.n_stack)

    # Clip rewards during training only; evaluation uses raw scores.
    if not eval_mode:
        env = TransformReward(env, _clip_reward)

    return env
