import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
import random
from marl.networks.actor_network import ActorNetwork
from marl.networks.critic_network import CriticNetwork

class ReplayBuffer:
    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.buffer = []
        self.ptr = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.ptr] = (state, action, reward, next_state, done)
        self.ptr = (self.ptr + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class MADDPGAgent:
    def __init__(self, state_dim, action_dim, global_state_dim, lr=1e-3, gamma=0.99, tau=0.005):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim

        # Actor (Deterministic)
        self.actor = ActorNetwork(state_dim, action_dim).to(self.device)
        self.actor_target = ActorNetwork(state_dim, action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        # Critic (Centralized: Global State + Actions of all agents)
        # Note: 'global_state_dim' here represents (N * State_Dim).
        # We assume Critic input is (N * State) + (N * Action)
        # We need to approximate N * Action dim roughly.
        # Ideally, pass explicit dims. Here we approximate:
        critic_input_dim = global_state_dim + (global_state_dim // state_dim) * action_dim
        
        self.critic = CriticNetwork(critic_input_dim).to(self.device)
        self.critic_target = CriticNetwork(critic_input_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = Adam(self.critic.parameters(), lr=lr * 10)
        
        self.memory = ReplayBuffer(100000, state_dim)

    def actions(self, state):
        # Exploration noise added outside or here
        state_t = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            mu, _ = self.actor(state_t) # ActorNetwork returns mu, std
            action = torch.tanh(mu)     # Deterministic action
            
        # Add slight noise for exploration during training if needed
        return action.cpu().numpy(), None, None

    def store(self, action, state, reward, done, next_state):
        # We need next_state for MADDPG (Off-policy)
        self.memory.push(state, action, reward, next_state, done)

    def update(self, batch_size=64):
        if len(self.memory) < batch_size: return
        
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        
        # Convert to tensors (Batch, Agents, Dim)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(-1) # (B, N, 1)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device).unsqueeze(-1)

        B, N, S_DIM = states.shape
        
        # Flatten for global critic
        # Inputs: (Batch, N*S_DIM) and (Batch, N*A_DIM)
        flat_states = states.view(B, -1)
        flat_actions = actions.view(B, -1)
        flat_next_states = next_states.view(B, -1)
        
        # --- Critic Update ---
        with torch.no_grad():
            # Target Actions: (Batch, N, A_DIM) -> flatten
            target_mu, _ = self.actor_target(next_states)
            target_actions = torch.tanh(target_mu) 
            flat_target_actions = target_actions.view(B, -1)
            
            # Concat State + Action for critic
            target_critic_in = torch.cat([flat_next_states, flat_target_actions], dim=1)
            
            target_q = self.critic_target(target_critic_in)
            
            # Since Critic outputs scalar for the whole team or per agent?
            # Existing CriticNetwork outputs scalar (1). 
            # If we want per-agent Q, we need N outputs.
            # For simplicity in this adaptation: MADDPG usually has 1 critic per agent.
            # WE WILL USE 1 CENTRALIZED CRITIC returning 1 value (Cooperative).
            # Target = r_sum + gamma * Q_next
            
            # Sum rewards across agents for cooperative objective
            total_reward = rewards.sum(dim=1) # (B, 1)
            target_value = total_reward + (1 - dones.any(dim=1, keepdim=True).float()) * self.gamma * target_q

        # Current Q
        current_critic_in = torch.cat([flat_states, flat_actions], dim=1)
        current_q = self.critic(current_critic_in)
        
        critic_loss = F.mse_loss(current_q, target_value)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # --- Actor Update ---
        # Maximize Q(s, mu(s))
        curr_mu, _ = self.actor(states)
        curr_acts = torch.tanh(curr_mu)
        
        # We need to construct the joint action vector with ONLY the current agent's action changed?
        # In fully cooperative shared-parameter MADDPG:
        flat_curr_acts = curr_acts.view(B, -1)
        actor_critic_in = torch.cat([flat_states, flat_curr_acts], dim=1)
        
        actor_loss = -self.critic(actor_critic_in).mean()
        
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # --- Soft Update ---
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor.pth")
        torch.save(self.critic.state_dict(), filename + "_critic.pth")
        
    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor.pth"))
        self.critic.load_state_dict(torch.load(filename + "_critic.pth"))