import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
import random
import os
from marl.networks.actor_network import ActorNetwork
from marl.networks.critic_network import CriticNetwork


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class MADDPGAgent:
    """
    Multi-Agent Deep Deterministic Policy Gradient agent.
    """

    def __init__(self, state_dim, action_dim, global_state_dim, actor_lr=1e-4, critic_lr=1e-3, gamma=0.99, tau=0.005):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim

        self.actor = ActorNetwork(state_dim, action_dim).to(self.device)
        self.actor_target = ActorNetwork(state_dim, action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        num_agents = global_state_dim // state_dim
        critic_input_dim = global_state_dim + num_agents * action_dim

        self.critic = CriticNetwork(critic_input_dim).to(self.device)
        self.critic_target = CriticNetwork(critic_input_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)

        self.replay_buffer = ReplayBuffer(capacity=100000)

    def act(self, state):
        """
        Compute deterministic actions from the actor network (used in training/eval).
        """
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            mu, _ = self.actor(state_tensor)
            action = torch.tanh(mu)
        return action.cpu().numpy(), None, None

    def actions(self, state):
        return self.act(state)

    def store(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self, batch_size=64):
        if len(self.replay_buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(-1)
        next_states = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(-1)

        batch_size_curr = states.shape[0]

        flat_states = states.reshape(batch_size_curr, -1)
        flat_actions = actions.reshape(batch_size_curr, -1)
        flat_next_states = next_states.reshape(batch_size_curr, -1)

        with torch.no_grad():
            target_mu, _ = self.actor_target(next_states)
            target_actions = torch.tanh(target_mu)
            flat_target_actions = target_actions.reshape(batch_size_curr, -1)

            target_critic_input = torch.cat([flat_next_states, flat_target_actions], dim=1)
            target_q = self.critic_target(target_critic_input)

            total_reward = rewards.sum(dim=1)
            done_mask = dones.any(dim=1, keepdim=True).float()
            target_value = total_reward + (1.0 - done_mask) * self.gamma * target_q

        critic_input = torch.cat([flat_states, flat_actions], dim=1)
        current_q = self.critic(critic_input)

        critic_loss = F.mse_loss(current_q, target_value)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        mu, _ = self.actor(states)
        curr_actions = torch.tanh(mu)
        flat_curr_actions = curr_actions.reshape(batch_size_curr, -1)

        actor_input = torch.cat([flat_states, flat_curr_actions], dim=1)
        actor_loss = -self.critic(actor_input).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.mul_(1.0 - self.tau)
            target_param.data.add_(self.tau * param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.mul_(1.0 - self.tau)
            target_param.data.add_(self.tau * param.data)

    def save(self, path):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        torch.save(self.actor.state_dict(), f"{path}_actor.pth")
        torch.save(self.critic.state_dict(), f"{path}_critic.pth")

    def load(self, path):
        self.actor.load_state_dict(torch.load(f"{path}_actor.pth", map_location=self.device))
        self.critic.load_state_dict(torch.load(f"{path}_critic.pth", map_location=self.device))