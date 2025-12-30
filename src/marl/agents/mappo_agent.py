import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from marl.networks.actor_network import ActorNetwork
from marl.networks.critic_network import CriticNetwork


class MAPPOAgent:
    """
    Multi-Agent PPO agent with a decentralized actor and a centralized critic.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        global_state_dim,
        actor_lr,
        critic_lr,
        epochs,
        gamma,
        clip_eps,
        entropy_coef,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = ActorNetwork(state_dim, action_dim).to(self.device)
        self.actor_old = ActorNetwork(state_dim, action_dim).to(self.device)
        self.actor_old.load_state_dict(self.actor.state_dict())

        self.critic = CriticNetwork(global_state_dim).to(self.device)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)

        self.epochs = epochs
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef

        self.memory = []

    def store(self, action, state, reward, done, log_prob, pre_tanh):
        # Arguments aligned with main.py call signature
        self.memory.append((state, action, reward, done, log_prob, pre_tanh))

    def act(self, state):
        """
        Sample a stochastic action from the actor policy.
        """
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action, log_prob, _, pre_tanh = self.actor.sample(state_tensor)
        return action.cpu().numpy(), log_prob.cpu().numpy(), pre_tanh.cpu().numpy()
    
    # Alias for compatibility
    def actions(self, state):
        return self.act(state)

    def act_deterministic(self, state):
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            mu, _ = self.actor(state_tensor)
            pre_tanh = mu
            action = torch.tanh(pre_tanh)
        log_prob = torch.zeros(action.shape[0], device=self.device)
        return action.cpu().numpy(), log_prob.cpu().numpy(), pre_tanh.cpu().numpy()

    def _compute_returns(self, rewards, dones):
        returns = []
        cumulative = 0.0
        for r, d in zip(reversed(rewards), reversed(dones)):
            if np.any(d):
                cumulative = 0.0
            cumulative = r + self.gamma * cumulative
            returns.insert(0, cumulative)
        return np.asarray(returns, dtype=np.float32)

    def update(self):
        if len(self.memory) == 0:
            return

        states, actions, rewards, dones, old_log_probs, pre_tanhs = zip(*self.memory)

        returns = self._compute_returns(rewards, dones)

        states_t = torch.as_tensor(np.array(states), dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        old_log_probs_t = torch.as_tensor(np.array(old_log_probs), dtype=torch.float32, device=self.device)
        pre_tanhs_t = torch.as_tensor(np.array(pre_tanhs), dtype=torch.float32, device=self.device)

        batch_size, num_agents, state_dim = states_t.shape
        action_dim = 1 if pre_tanhs_t.ndim == 1 else pre_tanhs_t.shape[-1]

        flat_states = states_t.reshape(-1, state_dim)
        flat_returns = returns_t.reshape(-1)
        flat_old_log_probs = old_log_probs_t.reshape(-1)
        flat_pre_tanhs = pre_tanhs_t.reshape(-1, action_dim)

        global_states = states_t.reshape(batch_size, -1)
        flat_global_states = global_states.unsqueeze(1).expand(-1, num_agents, -1).reshape(-1, num_agents * state_dim)

        with torch.no_grad():
            values_old = self.critic(flat_global_states).reshape(-1)
            advantages = flat_returns - values_old
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

        for _ in range(self.epochs):
            new_log_probs, entropy = self.actor.evaluate_pre_tanh(flat_states, flat_pre_tanhs)
            new_log_probs = new_log_probs.reshape(-1)

            ratios = torch.exp(new_log_probs - flat_old_log_probs.detach())
            clipped = torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps)

            policy_loss = -torch.min(ratios * advantages, clipped * advantages).mean()

            values = self.critic(flat_global_states).reshape(-1)
            value_loss = 0.5 * F.mse_loss(values, flat_returns)

            loss = policy_loss + value_loss - self.entropy_coef * entropy.mean()

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

        self.memory.clear()
        self.actor_old.load_state_dict(self.actor.state_dict())

    def save(self, path):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        torch.save(self.actor.state_dict(), f"{path}_actor.pth")
        torch.save(self.critic.state_dict(), f"{path}_critic.pth")

    def load(self, path):
        self.actor.load_state_dict(torch.load(f"{path}_actor.pth", map_location=self.device))
        self.critic.load_state_dict(torch.load(f"{path}_critic.pth", map_location=self.device))
        self.actor_old.load_state_dict(self.actor.state_dict())