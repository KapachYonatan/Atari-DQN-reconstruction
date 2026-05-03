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

    def save(self, path: str) -> None:
        """Save the live portion of the buffer to a compressed npz file.

        Only the filled slice (``[:self._size]``) is saved to keep file size small.

        Args:
            path: Destination ``.npz`` file path.
        """
        np.savez_compressed(
            path,
            states=self._states[: self._size],
            next_states=self._next_states[: self._size],
            actions=self._actions[: self._size],
            rewards=self._rewards[: self._size],
            dones=self._dones[: self._size],
            ptr=np.array([self._ptr]),
            size=np.array([self._size]),
        )

    def load(self, path: str) -> None:
        """Restore buffer state from a file written by :meth:`save`.

        Args:
            path: Path to the ``.npz`` file.
        """
        data = np.load(path)

        required = ["states", "next_states", "actions", "rewards", "dones"]
        missing = [name for name in required if name not in data.files]
        if missing:
            raise ValueError(
                f"Replay buffer archive is missing required arrays: {missing}. "
                f"Found keys: {list(data.files)}"
            )

        loaded_states = data["states"]
        loaded_next_states = data["next_states"]
        loaded_actions = data["actions"]
        loaded_rewards = data["rewards"]
        loaded_dones = data["dones"]

        if "size" in data.files:
            loaded_size = int(data["size"][0])
        else:
            # Backward-compatibility with older archives without metadata.
            loaded_size = int(loaded_states.shape[0])

        size = min(loaded_size, self._capacity)
        self._size = size

        if "ptr" in data.files:
            self._ptr = int(data["ptr"][0]) % self._capacity
        else:
            self._ptr = size % self._capacity

        self._states[:size] = loaded_states[:size]
        self._next_states[:size] = loaded_next_states[:size]
        self._actions[:size] = loaded_actions[:size]
        self._rewards[:size] = loaded_rewards[:size]
        self._dones[:size] = loaded_dones[:size]

    @property
    def is_ready(self) -> bool:
        """True once enough transitions exist to start training (checked externally)."""
        return True  # Readiness against min_replay_size checked in the training loop.
