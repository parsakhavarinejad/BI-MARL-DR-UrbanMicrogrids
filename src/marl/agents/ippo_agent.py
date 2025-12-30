import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from marl.networks.actor_network import ActorNetwork
from marl.networks.critic_network import CriticNetwork

class IPPOAgent:
    def __init__(self, state_dim, action_dim, ac_lr, cr_lr, K_epochs, gamma, eps_clip, entropy_coeff):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.actor = ActorNetwork(state_dim, action_dim).to(self.device)
        self.critic = CriticNetwork(state_dim).to(self.device) # Local state only
        
        self.actor_opt = Adam(self.actor.parameters(), lr=ac_lr)
        self.critic_opt = Adam(self.critic.parameters(), lr=cr_lr)
        
        self.K_epochs = K_epochs
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.entropy_coeff = entropy_coeff
        self.buffer = []

    def actions(self, state):
        state_t = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            action, log_prob, _, pre_tanh = self.actor.sample(state_t)
        return action.cpu().numpy(), log_prob.cpu().numpy(), pre_tanh.cpu().numpy()

    def store(self, action, state, reward, done, log_prob, pre_tanh):
        self.buffer.append([action, state, reward, done, log_prob, pre_tanh])

    def update(self):
        if not self.buffer: return
        
        actions, states, rewards, dones, log_probs, pre_tanhs = zip(*self.buffer)
        
        # 1. Monte Carlo Estimate of Returns
        discounted_rewards = []
        running_add = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            if np.any(d): running_add = 0
            running_add = r + self.gamma * running_add
            discounted_rewards.insert(0, running_add)
            
        # 2. Convert to Tensor
        rewards_t = torch.FloatTensor(np.array(discounted_rewards)).to(self.device)
        states_t = torch.FloatTensor(np.array(states)).to(self.device) #(B, N, State)
        old_logprobs_t = torch.FloatTensor(np.array(log_probs)).to(self.device)
        old_pre_tanh_t = torch.FloatTensor(np.array(pre_tanhs)).to(self.device)
        
        # Flatten for independent processing (Batch * Agents, State)
        B, N, D = states_t.shape
        flat_states = states_t.view(-1, D)
        flat_rewards = rewards_t.view(-1)
        flat_logprobs = old_logprobs_t.view(-1)
        flat_pre_tanh = old_pre_tanh_t.view(-1, old_pre_tanh_t.shape[-1])

        # 3. PPO Update
        for _ in range(self.K_epochs):
            # Evaluate old actions and values
            new_logprobs, entropy = self.actor.evaluate_pre_tanh(flat_states, flat_pre_tanh)
            new_logprobs = new_logprobs.view(-1)
            
            # Critic evaluates LOCAL state only
            state_values = self.critic(flat_states).view(-1)
            
            # Ratios
            ratios = torch.exp(new_logprobs - flat_logprobs.detach())
            
            # Advantages
            advantages = flat_rewards - state_values.detach()
            
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            loss_actor = -torch.min(surr1, surr2).mean()
            loss_critic = 0.5 * F.mse_loss(state_values, flat_rewards)
            
            loss = loss_actor + loss_critic - self.entropy_coeff * entropy.mean()
            
            self.actor_opt.zero_grad()
            self.critic_opt.zero_grad()
            loss.backward()
            self.actor_opt.step()
            self.critic_opt.step()
            
        self.buffer = []

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor.pth")
        torch.save(self.critic.state_dict(), filename + "_critic.pth")

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor.pth"))
        self.critic.load_state_dict(torch.load(filename + "_critic.pth"))