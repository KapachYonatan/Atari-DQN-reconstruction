"""Experience replay buffer for DQN.

States are stored as ``uint8`` numpy arrays (values 0–255) to keep RAM
usage manageable: 1 M transitions × 4 × 84 × 84 × 1 byte ≈ 6.7 GB.
Float32 normalisation (÷ 255) is applied lazily at sample time.
"""

from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    """Circular replay buffer storing (s, a, r, s', done) transitions.

    Args:
        capacity: Maximum number of transitions to store.
        state_shape: Shape of a single preprocessed observation, e.g. ``(4, 84, 84)``.
        device: PyTorch device to place sampled tensors on.
    """

    def __init__(
        self,
        capacity: int,
        state_shape: tuple[int, ...],
        device: torch.device,
    ) -> None:
        self._capacity = capacity
        self._device = device
        self._ptr = 0
        self._size = 0

        # Pre-allocate contiguous arrays. States stored as uint8 to save RAM.
        self._states = np.zeros((capacity, *state_shape), dtype=np.uint8)
        self._next_states = np.zeros((capacity, *state_shape), dtype=np.uint8)
        self._actions = np.zeros(capacity, dtype=np.int64)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._dones = np.zeros(capacity, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a single transition.

        Args:
            state: Preprocessed observation, shape ``state_shape``, dtype uint8.
            action: Integer action index.
            reward: Scalar reward (already clipped during training).
            next_state: Next preprocessed observation.
            done: True if the episode ended after this transition.
        """
        self._states[self._ptr] = state
        self._next_states[self._ptr] = next_state
        self._actions[self._ptr] = action
        self._rewards[self._ptr] = reward
        self._dones[self._ptr] = float(done)

        self._ptr = (self._ptr + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def sample(
        self, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a random mini-batch of transitions.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Tuple of ``(states, actions, rewards, next_states, dones)``
            as PyTorch tensors on ``self._device``.
            States and next_states are float32 normalised to [0, 1].
        """
        indices = np.random.randint(0, self._size, size=batch_size)

        states = (
            torch.from_numpy(self._states[indices]).float().div(255.0).to(self._device, non_blocking=True)
        )
        next_states = (
            torch.from_numpy(self._next_states[indices]).float().div(255.0).to(self._device, non_blocking=True)
        )
        actions = torch.from_numpy(self._actions[indices]).to(self._device, non_blocking=True)
        rewards = torch.from_numpy(self._rewards[indices]).to(self._device, non_blocking=True)
        dones = torch.from_numpy(self._dones[indices]).to(self._device, non_blocking=True)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return self._size

    @property
    def is_ready(self) -> bool:
        """True once enough transitions exist to start training (checked externally)."""
        return True  # Readiness against min_replay_size checked in the training loop.
