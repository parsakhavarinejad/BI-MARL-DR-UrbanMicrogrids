import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorNetwork(nn.Module):
    """
    Gaussian actor network for continuous-action reinforcement learning.

    This network maps an input state to the parameters of a Gaussian policy:
        - mean (mu)
        - standard deviation (sigma)

    The policy is defined as:
        π(a | s) = Normal(mu(s), sigma(s))

    Notes
    -----
    - `mu` is squashed with `tanh` to keep actions bounded.
    - `sigma` is produced using `softplus` to ensure positivity.
    - The network is suitable for PPO / A2C-style actor–critic methods.
    """

    def __init__(self, state_dim, action_dim):
        """
        Initialize the actor network.

        Parameters
        ----------
        state_dim : int
            Dimension of the input state.
        action_dim : int
            Dimension of the action space.
        """
        super().__init__()

        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, action_dim)
        self.fc6 = nn.Linear(64, action_dim)

    def forward(self, x):
        """
        Forward pass of the actor network.

        Parameters
        ----------
        x : torch.Tensor
            Input state tensor with shape (..., state_dim).

        Returns
        -------
        mu : torch.Tensor
            Mean of the Gaussian policy, shape (..., action_dim).
        sigma : torch.Tensor
            Standard deviation of the Gaussian policy, shape (..., action_dim).
        """
        x = F.relu(self.fc2(F.relu(self.fc1(x))))
        x = F.relu(self.fc4(F.relu(self.fc3(x))))

        mu = torch.tanh(self.fc5(x))
        sigma = F.softplus(self.fc6(x)) + 1e-6

        return mu, sigma

    def get_action(self, state):
        """
        Sample an action from the policy given a state.

        Parameters
        ----------
        state : torch.Tensor
            Input state tensor with shape (..., state_dim).

        Returns
        -------
        sample_action : torch.Tensor
            Action sampled from the Gaussian policy.
        log_prob : torch.Tensor
            Log-probability of the sampled action.
        entropy : torch.Tensor
            Entropy of the Gaussian policy.
        """
        mu, sigma = self.forward(state)
        normal_distribution = torch.distributions.Normal(mu, sigma)

        sample_action = normal_distribution.sample()
        log_prob = normal_distribution.log_prob(sample_action)
        entropy = normal_distribution.entropy()

        return sample_action, log_prob, entropy


# ---------- Test ----------
if __name__ == "__main__":
    x = torch.rand(12, 8)
    actor_network = ActorNetwork(8, 1)

    mu, sigma = actor_network(x)
    print("mu: ", mu.shape)
    print("sigma: ", sigma.shape)

    sample_action, log_prob, entropy = actor_network.get_action(x)

    print("Sample Action: ", sample_action.shape)
    print("Log prob: ", log_prob.shape)
    print("Entropy: ", entropy.shape)
