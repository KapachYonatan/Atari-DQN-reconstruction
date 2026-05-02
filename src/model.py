"""DQN network architecture from Mnih et al. (2013/2015).

Architecture (input: 4 × 84 × 84 uint8 frames, normalised to float32):

    Conv1  :  32 filters, 8×8 kernel, stride 4  → ReLU
    Conv2  :  64 filters, 4×4 kernel, stride 2  → ReLU
    Conv3  :  64 filters, 3×3 kernel, stride 1  → ReLU
    Flatten
    FC1    :  512 units                          → ReLU
    FC_out :  n_actions units                   (linear)
"""

import torch
import torch.nn as nn


class AtariDQN(nn.Module):
    """Deep Q-Network for Atari, following the architecture in the DQN paper.

    Args:
        n_actions: Number of discrete actions in the environment.
    """

    def __init__(self, n_actions: int) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Conv output size for 84×84 input: 64 × 7 × 7 = 3136
        self.fc = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Q-values for a batch of states.

        Args:
            x: Float tensor of shape ``(B, 4, 84, 84)`` with values in [0, 1].

        Returns:
            Q-value tensor of shape ``(B, n_actions)``.
        """
        return self.fc(self.conv(x))
