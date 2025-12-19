import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class ActorNetwork(nn.Module):
    """
    Tanh-squashed Gaussian actor for bounded continuous actions in PPO/MAPPO.

    Produces a Normal distribution in unconstrained space (pre-tanh),
    then samples and squashes with tanh to fit actions in [-1, 1].

    Returns:
      - action (tanh-squashed)
      - log_prob (with tanh correction)
      - entropy (of the underlying Normal, pre-tanh)
    """

    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 64)

        self.mu_head = nn.Linear(64, action_dim)
        self.log_std_head = nn.Linear(64, action_dim)

        self.LOG_STD_MIN = -5.0   
        self.LOG_STD_MAX = 0.0   

    def forward(self, x):
        """
        Returns parameters of the *pre-tanh* Gaussian: mean and std.

        mu: unconstrained (no tanh here)
        std: exp(log_std), clipped via log_std bounds for stability
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        mu = self.mu_head(x)

        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)

        return mu, std

    @torch.no_grad()
    def act(self, state):
        """
        Convenience method for environment interaction:
        returns action only (no grad). Useful in rollout collection.
        """
        action, _, _ = self.get_action(state)
        return action

    def sample(self, state):
        """
        Samples pre_tanh ~ N(mu, std), then action = tanh(pre_tanh).
        Returns: action, log_prob(action|state), entropy, pre_tanh
        """
        mu, std = self.forward(state)
        dist = Normal(mu, std)

        pre_tanh = dist.rsample()
        action = torch.tanh(pre_tanh)

        log_prob = dist.log_prob(pre_tanh).sum(dim=-1)
        log_prob -= torch.sum(torch.log(1.0 - action.pow(2) + 1e-6), dim=-1)

        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy, pre_tanh

    def evaluate_pre_tanh(self, state, pre_tanh):
        """
        Evaluate log_prob + entropy for a GIVEN pre_tanh sample.
        Returns: log_prob, entropy
        """
        mu, std = self.forward(state)
        dist = Normal(mu, std)

        action = torch.tanh(pre_tanh)

        log_prob = dist.log_prob(pre_tanh).sum(dim=-1)
        log_prob -= torch.sum(torch.log(1.0 - action.pow(2) + 1e-6), dim=-1)

        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


# ---------- Test ----------
if __name__ == "__main__":
    x = torch.rand(12, 8)
    actor_network = ActorNetwork(8, 1)

    with torch.no_grad:
        mu, std = actor_network(x)
        print("mu: ", mu.shape)
        print("std: ", std.shape)

   
        sample_action, log_prob, entropy = actor_network.sample(x)
        print("Sample Action: ", sample_action.shape)
        print("Log prob: ", log_prob.shape)
        print("Entropy: ", entropy.shape)
