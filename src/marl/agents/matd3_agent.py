import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
import random
import os
from marl.networks.actor_network import ActorNetwork
from marl.networks.q_network import QNetwork


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


class MATD3Agent:
    """
    Multi-Agent Twin Delayed DDPG (MATD3).
    CTDE: Centralized Critics (x2), Decentralized Actors.
    """

    def __init__(self, state_dim, action_dim, global_state_dim, actor_lr=1e-4, critic_lr=1e-3, gamma=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.total_it = 0

        self.actor = ActorNetwork(state_dim, action_dim).to(self.device)
        self.actor_target = ActorNetwork(state_dim, action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)

        num_agents = global_state_dim // state_dim
        critic_input_dim = global_state_dim + num_agents * action_dim

        self.critic_1 = QNetwork(critic_input_dim, hidden_dims=(256, 256), use_layer_norm=True).to(self.device)
        self.critic_2 = QNetwork(critic_input_dim, hidden_dims=(256, 256), use_layer_norm=True).to(self.device)

        self.critic_1_target = QNetwork(critic_input_dim, hidden_dims=(256, 256), use_layer_norm=True).to(self.device)
        self.critic_2_target = QNetwork(critic_input_dim, hidden_dims=(256, 256), use_layer_norm=True).to(self.device)

        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        self.critic_optimizer = Adam(list(self.critic_1.parameters()) + list(self.critic_2.parameters()), lr=critic_lr)

        self.replay_buffer = ReplayBuffer(capacity=100000)

    def act(self, state):
        """
        Get deterministic action (mu) from Actor.
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
        self.total_it += 1
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
            mu_next, _ = self.actor_target(next_states)
            noise = (torch.randn_like(mu_next) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            
            next_action = torch.tanh(mu_next + noise)
            flat_next_action = next_action.reshape(batch_size_curr, -1)

            target_critic_input = torch.cat([flat_next_states, flat_next_action], dim=1)
            target_q1 = self.critic_1_target(target_critic_input)
            target_q2 = self.critic_2_target(target_critic_input)
            target_q = torch.min(target_q1, target_q2)
            
            total_reward = rewards.sum(dim=1)
            done_mask = dones.any(dim=1, keepdim=True).float()
            
            target_value = total_reward + (1.0 - done_mask) * self.gamma * target_q

        critic_input = torch.cat([flat_states, flat_actions], dim=1)
        current_q1 = self.critic_1(critic_input)
        current_q2 = self.critic_2(critic_input)

        critic_loss = F.mse_loss(current_q1, target_value) + F.mse_loss(current_q2, target_value)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.critic_1.parameters()) + list(self.critic_2.parameters()), 1.0)
        self.critic_optimizer.step()

        if self.total_it % self.policy_freq == 0:
            mu, _ = self.actor(states)
            curr_actions = torch.tanh(mu)
            flat_curr_actions = curr_actions.reshape(batch_size_curr, -1)
            
            actor_input = torch.cat([flat_states, flat_curr_actions], dim=1)
            actor_loss = -self.critic_1(actor_input).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                target_param.data.mul_(1.0 - self.tau)
                target_param.data.add_(self.tau * param.data)

            for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
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
        torch.save(self.critic_1.state_dict(), f"{path}_critic1.pth")
        torch.save(self.critic_2.state_dict(), f"{path}_critic2.pth")

    def load(self, path):
        self.actor.load_state_dict(torch.load(f"{path}_actor.pth", map_location=self.device))
        self.critic_1.load_state_dict(torch.load(f"{path}_critic1.pth", map_location=self.device))
        self.critic_2.load_state_dict(torch.load(f"{path}_critic2.pth", map_location=self.device))