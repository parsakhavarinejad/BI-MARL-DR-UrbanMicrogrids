import torch
import torch.nn as nn
import torch.nn.functional as F


class CriticNetwork(nn.Module):
    """
    Centralized critic network for value estimation in multi-agent RL.

    This critic takes a *global state* as input and outputs a scalar state-value:
        V(s_global)

    In CTDE (Centralized Training, Decentralized Execution) settings, the global state
    is commonly formed by concatenating all agents' observations. For example:
        num_agents = 12, obs_dim = 8  -> global_state_dim = 96

    Parameters
    ----------
    global_state_dim : int
        Dimension of the global state vector passed to the critic.

    Returns
    -------
    value : torch.Tensor
        Estimated state value with shape (..., 1).
    """

    def __init__(self, global_state_dim):
        """
        Initialize the critic network.

        Parameters
        ----------
        global_state_dim : int
            Input dimension of the global state.
        """
        super().__init__()

        self.fc1 = nn.Linear(global_state_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 1)

    def forward(self, x):
        """
        Forward pass of the critic network.

        Parameters
        ----------
        x : torch.Tensor
            Global state tensor with shape (..., global_state_dim).

        Returns
        -------
        torch.Tensor
            State-value estimate with shape (..., 1).
        """
        x = F.relu(self.fc2(F.relu(self.fc1(x))))
        x = F.relu(self.fc4(F.relu(self.fc3(x))))

        value = self.fc5(x)
        return value


# ---------- Test ----------
if __name__ == "__main__":
    x = torch.rand(12 * 8)
    ciritc_network = CriticNetwork(12 * 8)

    value = ciritc_network(x)
    print("State Value: ", value.shape)
