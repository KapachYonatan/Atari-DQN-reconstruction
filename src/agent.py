"""DQN agent: action selection, learning, and target network synchronisation.

Owns the online network, target network, and optimizer.
The training loop drives when ``learn()`` and ``sync_target()`` are called;
the agent itself has no step counter.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import Config
from src.model import AtariDQN
from src.replay_buffer import ReplayBuffer


class DQNAgent:
    """DQN agent wrapping the online/target networks and optimizer.

    Args:
        n_actions: Number of discrete actions available in the environment.
        config: Project-wide configuration instance.
        device: PyTorch device to run computations on.
    """

    def __init__(self, n_actions: int, config: Config, device: torch.device) -> None:
        self._n_actions = n_actions
        self._config = config
        self._device = device

        self.online_net = AtariDQN(n_actions).to(device)
        self.target_net = copy.deepcopy(self.online_net)
        self.target_net.eval()  # target network is never trained directly

        self._optimizer = self._build_optimizer()

    # ------------------------------------------------------------------ #
    # Action selection                                                     #
    # ------------------------------------------------------------------ #

    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        """ε-greedy action selection.

        Args:
            state: Single preprocessed observation of shape ``(4, 84, 84)``,
                dtype uint8.
            epsilon: Current exploration probability.

        Returns:
            Integer action index.
        """
        if np.random.random() < epsilon:
            return int(np.random.randint(self._n_actions))

        state_t = (
            torch.from_numpy(state)
            .float()
            .div(255.0)
            .unsqueeze(0)  # → (1, 4, 84, 84)
            .to(self._device, non_blocking=True)
        )
        with torch.no_grad():
            q_values = self.online_net(state_t)
        return int(q_values.argmax(dim=1).item())

    # ------------------------------------------------------------------ #
    # Learning                                                             #
    # ------------------------------------------------------------------ #

    def learn(self, buffer: ReplayBuffer) -> float:
        """Sample a mini-batch and perform one gradient update.

        Uses Double-DQN-style target computation:
            y = r + γ · Q_target(s', argmax_a Q_online(s', a))  if not done
            y = r                                                 if done

        Args:
            buffer: Replay buffer to sample from.

        Returns:
            Scalar loss value (for logging).
        """
        states, actions, rewards, next_states, dones = buffer.sample(
            self._config.batch_size
        )

        # -------------------------------------------------------------- #
        # Compute TD targets using the target network                     #
        # -------------------------------------------------------------- #
        with torch.no_grad():
            # Action selection from online net, value from target net (Double DQN)
            next_actions = self.online_net(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            targets = rewards + self._config.gamma * next_q * (1.0 - dones)

        # -------------------------------------------------------------- #
        # Compute online Q-values for taken actions                       #
        # -------------------------------------------------------------- #
        q_values = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Huber loss (smooth_l1) is equivalent to the clipped MSE in the paper
        loss = F.smooth_l1_loss(q_values, targets)

        self._optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if self._config.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.online_net.parameters(), self._config.grad_clip)

        self._optimizer.step()

        return float(loss.item())

    # ------------------------------------------------------------------ #
    # Target network                                                       #
    # ------------------------------------------------------------------ #

    def sync_target(self) -> None:
        """Hard-copy online network weights to target network."""
        self.target_net.load_state_dict(self.online_net.state_dict())

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _build_optimizer(self) -> torch.optim.Optimizer:
        cfg = self._config
        if cfg.optimizer == "rmsprop":
            return torch.optim.RMSprop(
                self.online_net.parameters(),
                lr=cfg.lr,
                alpha=cfg.rmsprop_alpha,
                eps=cfg.rmsprop_eps,
                momentum=cfg.rmsprop_momentum,
            )
        if cfg.optimizer == "adam":
            return torch.optim.Adam(self.online_net.parameters(), lr=cfg.lr)
        raise ValueError(f"Unknown optimizer '{cfg.optimizer}'. Choose 'rmsprop' or 'adam'.")
