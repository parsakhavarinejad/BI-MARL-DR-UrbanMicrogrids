import numpy as np
from marl.networks.actor_network import ActorNetwork
from marl.networks.critic_network import CriticNetwork

from torch.optim import Adam
import torch
import torch.nn.functional as F
import os

class MAPPOAgent:
    """
    MAPPO-style agent with a decentralized actor and a centralized critic.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        global_state_dim,
        ac_lr,
        cr_lr,
        K_epochs,
        gamma,
        eps_clip,
        entropy_coeff,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.actor_network = ActorNetwork(state_dim, action_dim).to(self.device)
        # We keep an old network for PPO ratio calculation
        self.old_actor_network = ActorNetwork(state_dim, action_dim).to(self.device)
        self.old_actor_network.load_state_dict(self.actor_network.state_dict())

        self.critic_network = CriticNetwork(global_state_dim).to(self.device)

        self.actor_opt = Adam(self.actor_network.parameters(), lr=ac_lr)
        self.critic_opt = Adam(self.critic_network.parameters(), lr=cr_lr)

        self.K_epochs = K_epochs
        self.gamma = gamma
        self.eps_clip = eps_clip

        self.agent_entropy_coeff = entropy_coeff
        self.buffer = []

    def store(self, action, state, reward, done, log_prob, pre_tanh):
        obs = [action, state, reward, done, log_prob, pre_tanh]
        self.buffer.append(obs)

    def actions(self, state):
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            action, log_prob, _, pre_tanh = self.actor_network.sample(state)
        return action.cpu().numpy(), log_prob.cpu().numpy(), pre_tanh.cpu().numpy()

    def actions_deterministic(self, state):
        state_t = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            mu, _std = self.actor_network(state_t)
            pre_tanh = mu
            action = torch.tanh(pre_tanh)
        logprob = torch.zeros(action.shape[0], device=self.device)
        return action.cpu().numpy(), logprob.cpu().numpy(), pre_tanh.cpu().numpy()

    def update(self):
        if not self.buffer:
            return

        _, state, rewards, done, log_prob, pre_tanhs = zip(*self.buffer)

        discounted_reward = []
        current_reward = 0

        for reward, is_terminal in zip(reversed(rewards), reversed(done)):
            if np.any(is_terminal):
                current_reward = 0
            current_reward = reward + self.gamma * current_reward
            discounted_reward.insert(0, current_reward)

        rewards_tensor = torch.FloatTensor(np.array(discounted_reward)).to(self.device)
        old_states = torch.FloatTensor(np.array(state)).to(self.device)
        old_logprobs = torch.FloatTensor(np.array(log_prob)).to(self.device)
        old_pre_tanh = torch.FloatTensor(np.array(pre_tanhs)).to(self.device)
        
        # Check dimensions
        if old_pre_tanh.ndim > 1:
            a_dim = old_pre_tanh.shape[-1]
        else:
            a_dim = 1

        b, n, d = old_states.shape
        flat_states = old_states.view(-1, d)
        flat_old_logprobs = old_logprobs.reshape(-1)
        flat_rewards = rewards_tensor.reshape(-1)
        flat_pre_tanh = old_pre_tanh.reshape(-1, a_dim)

        global_state = old_states.view(b, -1)
        flat_global_state = (
            global_state.unsqueeze(1).expand(-1, n, -1).reshape(-1, n * d)
        )

        with torch.no_grad():
            critic_value_old = self.critic_network(flat_global_state).reshape(-1)
            advantages = flat_rewards - critic_value_old
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

        for _ in range(self.K_epochs):
            new_logprob, entropy = self.actor_network.evaluate_pre_tanh(
                flat_states, flat_pre_tanh
            )
            new_logprob = new_logprob.reshape(-1)

            ratio = torch.exp(new_logprob - flat_old_logprobs.detach())

            surr1 = ratio * advantages
            surr2 = (
                torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            )

            new_value = self.critic_network(flat_global_state).reshape(-1)

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = 0.5 * F.mse_loss(new_value, flat_rewards)
            entropy_bonus = entropy.mean()

            loss = actor_loss + critic_loss - self.agent_entropy_coeff * entropy_bonus

            self.actor_opt.zero_grad()
            self.critic_opt.zero_grad()
            loss.backward()
            self.actor_opt.step()
            self.critic_opt.step()

        self.buffer = []
        self.old_actor_network.load_state_dict(self.actor_network.state_dict())

    # --- ADDED METHODS BELOW ---

    def save(self, filename):
        """
        Saves the actor and critic weights.
        """
        # Ensure the directory exists (filename might include a path)
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            
        torch.save(self.actor_network.state_dict(), filename + "_actor.pth")
        torch.save(self.critic_network.state_dict(), filename + "_critic.pth")

    def load(self, filename):
        """
        Loads the actor and critic weights.
        """
        self.actor_network.load_state_dict(torch.load(filename + "_actor.pth", map_location=self.device))
        self.critic_network.load_state_dict(torch.load(filename + "_critic.pth", map_location=self.device))
        
        # Sync old network
        self.old_actor_network.load_state_dict(self.actor_network.state_dict())