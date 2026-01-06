import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """
    Centralized Q critic: Q(s_global, a_joint) -> scalar
    Input: concatenated [global_state, joint_action]
    Output: (batch, 1)
    """

    def __init__(self, input_dim: int, hidden_dims=(256, 256), use_layer_norm: bool = True):
        super().__init__()

        h1, h2 = hidden_dims

        self.fc1 = nn.Linear(input_dim, h1)
        self.ln1 = nn.LayerNorm(h1) if use_layer_norm else None

        self.fc2 = nn.Linear(h1, h2)
        self.ln2 = nn.LayerNorm(h2) if use_layer_norm else None

        self.fc3 = nn.Linear(h2, 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.fc1.weight, gain=1.0)
        nn.init.zeros_(self.fc1.bias)

        nn.init.orthogonal_(self.fc2.weight, gain=1.0)
        nn.init.zeros_(self.fc2.bias)

        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        if self.ln1 is not None:
            x = self.ln1(x)
        x = F.relu(x)

        x = self.fc2(x)
        if self.ln2 is not None:
            x = self.ln2(x)
        x = F.relu(x)

        return self.fc3(x)
