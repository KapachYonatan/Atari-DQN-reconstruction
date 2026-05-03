"""Environment factory for Atari DQN.

Builds the preprocessed Crazy Climber environment using only standard
Gymnasium wrappers, reproducing the DQN paper preprocessing pipeline:

    1. AtariPreprocessing    — noop reset (1–30), frame-skip=4, max-pool over
                               last 2 frames, grayscale, resize to 84×84.
    2. FrameStackObservation — stack 4 consecutive frames → shape (4, 84, 84).
    3. TransformReward       — reward scaling controlled by ``config.reward_mode``:
                               'clip'      → {-1, 0, +1}  (DQN paper default)
                               'normalize' → min(r / reward_max_abs, 1.0)
                                             dense rewards stay meaningful;
                                             rare spikes cap at 1.0
                               'none'      → raw game reward (requires lr retuning)

The factory is intentionally thin: it only wires wrappers and exposes a
callable so training and evaluation code can each create their own isolated
env instance.
"""

from typing import Callable

import ale_py  # registers ALE environments with Gymnasium
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation, TransformReward

from src.config import Config


_VALID_REWARD_MODES = {"clip", "normalize", "none"}


def _clip_reward(reward: float) -> float:
    """Clip a reward value to {-1, 0, +1}."""
    if reward > 0:
        return 1.0
    if reward < 0:
        return -1.0
    return 0.0


def _make_normalize_reward(max_abs: float) -> "Callable[[float], float]":
    """Return a reward function that scales r to (0, 1] by dividing by max_abs.

    Crazy Climber only emits non-negative rewards, so no lower-bound guard
    is needed.  Rare large bonuses are capped at 1.0 to prevent TD-target
    spikes.

    Args:
        max_abs: The expected maximum single-step reward (e.g. 400 for
            Building 4 climbing points in Crazy Climber).
    """
    scale = float(max_abs)

    def _normalize(reward: float) -> float:
        return min(reward / scale, 1.0)

    return _normalize


def make_env(config: Config, *, eval_mode: bool = False) -> gym.Env:
    """Create and wrap the Atari environment.

    Args:
        config: Project-wide configuration instance.
        eval_mode: When True no reward transformation is applied (raw game
            scores) so evaluation metrics are comparable to paper benchmarks.
            Training applies the transformation selected by
            ``config.reward_mode``.

    Returns:
        A fully wrapped ``gymnasium.Env`` with observation shape
        ``(n_stack, 84, 84)`` and dtype ``uint8``.

    Raises:
        ValueError: If ``config.reward_mode`` is not one of the valid options.
    """
    if config.reward_mode not in _VALID_REWARD_MODES:
        raise ValueError(
            f"Unknown reward_mode '{config.reward_mode}'. "
            f"Must be one of {sorted(_VALID_REWARD_MODES)}."
        )

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

    # Apply reward transformation during training only.
    if not eval_mode:
        if config.reward_mode == "clip":
            env = TransformReward(env, _clip_reward)
        elif config.reward_mode == "normalize":
            env = TransformReward(env, _make_normalize_reward(config.reward_max_abs))
        # 'none': no wrapper — raw rewards passed through

    return env
