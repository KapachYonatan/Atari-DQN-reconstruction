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
    buffer_capacity: int = 1_000_000
    """Maximum number of transitions stored.  States are kept as uint8 to
    limit RAM usage (~6.7 GB for 1 M × 4 × 84 × 84 frames)."""

    min_replay_size: int = 50_000
    """Training updates start only after this many transitions are stored."""

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

    eval_episodes: int = 10
    """Number of episodes per evaluation checkpoint.
    Use 30+ for final 3 submitted runs; document this in the report."""

    # ------------------------------------------------------------------ #
    # Checkpointing & logging                                              #
    # ------------------------------------------------------------------ #
    run_id: str = "run1"
    results_dir: str = "results"
    """Root directory for per-run subdirectories."""

    save_frequency: int = 100_000
    """Save a checkpoint every this many training steps."""

    # ------------------------------------------------------------------ #
    # Reproducibility                                                      #
    # ------------------------------------------------------------------ #
    seed: int = 42
