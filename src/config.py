"""Central configuration dataclass for the Atari DQN project.

This is the single place to change hyperparameters between tuning runs.
All other modules import Config and read from it — never hardcode values elsewhere.
"""

from dataclasses import dataclass, field


@dataclass
class Config:
    # ------------------------------------------------------------------ #
    # Environment                                                          #
    # ------------------------------------------------------------------ #
    env_id: str = "ALE/CrazyClimber-v5"
    """Gymnasium environment ID.  Must be a NoFrameskip variant so that
    AtariPreprocessing can apply its own frame-skip internally."""

    n_stack: int = 4
    """Number of consecutive frames to stack into a single observation."""

    # ------------------------------------------------------------------ #
    # Replay buffer                                                        #
    # ------------------------------------------------------------------ #
    buffer_capacity: int = 150_000
    """Maximum number of transitions stored.

    Colab-safe default to avoid system RAM exhaustion with the current replay
    layout (state + next_state). Increase to 1,000,000 only when enough host
    RAM is available.
    """

    min_replay_size: int = 20_000
    """Training updates start only after this many transitions are stored.

    Keep this proportional to ``buffer_capacity`` when tuning on limited-RAM
    environments.
    """

    # ------------------------------------------------------------------ #
    # Training                                                             #
    # ------------------------------------------------------------------ #
    total_steps: int = 10_000_000
    """Total environment steps for one training run."""

    batch_size: int = 32
    update_frequency: int = 4
    """Perform one gradient update every this many environment steps."""

    target_update_freq: int = 10_000
    """Copy online → target network every this many environment steps."""

    gamma: float = 0.99
    """Discount factor."""

    # ------------------------------------------------------------------ #
    # Optimizer                                                            #
    # ------------------------------------------------------------------ #
    lr: float = 0.00025
    optimizer: str = "rmsprop"
    """'rmsprop' (paper) or 'adam'."""

    rmsprop_alpha: float = 0.95
    rmsprop_eps: float = 0.01
    rmsprop_momentum: float = 0.0

    grad_clip: float | None = None
    """If set, clip gradient norms to this value before each optimizer step."""

    reward_mode: str = "clip"
    """How to scale training rewards. Options:
    - 'clip'      : clip to {-1, 0, +1} (DQN paper default).
    - 'normalize' : clip(r / reward_max_abs, 1).
                    Dense rewards stay meaningful (100 → 0.25, 400 → 1.0);
                    rare large bonuses saturate at ±1 instead of exploding.
    - 'none'      : pass raw game rewards unchanged (requires retuning lr).
    """

    reward_max_abs: float = 400.0
    """Divisor used when ``reward_mode='normalize'``.
    For Crazy Climber: Building 4 awards 400 pts/row — the highest dense reward.
    Using 400 maps the most common rewards to (0, 1] and caps spikes at 1.
    """

    # ------------------------------------------------------------------ #
    # Exploration                                                          #
    # ------------------------------------------------------------------ #
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay_steps: int = 1_000_000
    """Epsilon is linearly annealed from start to end over this many steps."""

    eval_epsilon: float = 0.05
    """Epsilon used during evaluation (paper convention: not 0 to avoid loops)."""

    # ------------------------------------------------------------------ #
    # Evaluation                                                           #
    # ------------------------------------------------------------------ #
    eval_frequency: int = 10_000
    """Run an evaluation episode every this many training steps."""

    eval_episodes: int = 100
    """Number of episodes per evaluation checkpoint.
    Use 30+ for final 3 submitted runs; document this in the report."""

    # ------------------------------------------------------------------ #
    # Checkpointing & logging                                              #
    # ------------------------------------------------------------------ #
    run_id: str = "run1"
    results_dir: str = "results"
    """Root directory for per-run subdirectories."""

    save_frequency: int = 50_000
    """Save a checkpoint every this many training steps."""

    # ------------------------------------------------------------------ #
    # Reproducibility                                                      #
    # ------------------------------------------------------------------ #
    seed: int = 42

    # ------------------------------------------------------------------ #
    # Resume                                                               #
    # ------------------------------------------------------------------ #
    resume_from: str = ""
    """Path to a previous run directory to resume from (e.g. 'results/run1').
    Empty string means a fresh run.
    Expects checkpoint_latest.pt and buffer_latest.npz inside that directory."""
