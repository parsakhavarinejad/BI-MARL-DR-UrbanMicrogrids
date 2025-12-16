import numpy as np
from src.marl.networks.actor_network import ActorNetwork
from src.marl.networks.critic_network import CriticNetwork

from torch.optim import Adam
import torch
import torch.nn.functional as F


class MAPPOAgent:
    """
    MAPPO-style agent with a decentralized actor and a centralized critic.

    This implementation follows the CTDE idea:
    - Actor takes local state per agent:      (num_agents, state_dim)
    - Critic takes a global state vector:     (num_agents * state_dim,)

    The buffer stores trajectory tuples and `update` performs K epochs of PPO-style
    optimization with clipping.

    Notes
    -----
    - This code assumes the actor's `get_action(state)` returns (action, log_prob, entropy).
    - The critic is trained to predict a scalar value for the global state.
    - Rewards are discounted and then normalized before training.
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
    ):
        """
        Initialize actor/critic networks, optimizers, and training hyperparameters.

        Parameters
        ----------
        state_dim : int
            Dimension of each agent's local state (observation features).
        action_dim : int
            Dimension of each agent's action space.
        global_state_dim : int
            Dimension of the critic input (typically num_agents * state_dim).
        ac_lr : float
            Learning rate for the actor optimizer.
        cr_lr : float
            Learning rate for the critic optimizer.
        K_epochs : int
            Number of PPO update epochs per batch of collected experience.
        gamma : float
            Discount factor.
        eps_clip : float
            PPO clipping parameter.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.actor_network = ActorNetwork(state_dim, action_dim).to(self.device)
        self.old_actor_network = ActorNetwork(state_dim, action_dim).to(self.device)
        self.old_actor_network.load_state_dict(self.actor_network.state_dict())

        self.critic_network = CriticNetwork(global_state_dim).to(self.device)

        self.actor_opt = Adam(self.actor_network.parameters(), lr=ac_lr)
        self.critic_opt = Adam(self.critic_network.parameters(), lr=cr_lr)

        self.K_epochs = K_epochs
        self.gamma = gamma
        self.eps_clip = eps_clip

        self.buffer = []

    def store(self, acton, state, reward, done, log_prob):
        """
        Store a single transition in the on-policy buffer.

        Parameters
        ----------
        acton : np.ndarray or torch.Tensor
            Action taken by the policy.
        state : np.ndarray or torch.Tensor
            Local states for all agents at the time of action.
        reward : float or np.ndarray
            Reward signal from the environment.
        done : bool or np.ndarray
            Episode termination flag(s).
        log_prob : np.ndarray or torch.Tensor
            Log-probability of the action under the behavior policy.
        """
        obs = [acton, state, reward, done, log_prob]
        self.buffer.append(obs)

    def actions(self, state):
        """
        Sample actions from the current actor for a given state.

        Parameters
        ----------
        state : np.ndarray
            Local observation/state, typically shaped (num_agents, state_dim).

        Returns
        -------
        action : np.ndarray
            Sampled action(s).
        log_prob : np.ndarray
            Log-probability of the sampled action(s).
        """
        state = torch.FloatTensor(state).to(self.device)

        with torch.no_grad():
            action, log_prob, _ = self.actor_network.get_action(state)

        return action.cpu().numpy(), log_prob.cpu().numpy()

    def update(self, state):
        """
        Update actor and critic using PPO-style clipped objective.

        This method:
        1) Builds discounted rewards from the buffer
        2) Flattens data over (batch, agent) for vectorized optimization
        3) Constructs per-agent repeated global state for centralized critic input
        4) Runs K epochs of PPO update

        Parameters
        ----------
        state : Any
            Unused in the current implementation (kept to preserve structure).
        """
        _, state, rewards, done, log_prob = zip(*self.buffer)

        discounted_reward = []
        current_reward = 0

        for reward, is_terminal in zip(reversed(rewards), reversed(done)):
            if np.any(is_terminal):
                current_reward = 0

            current_reward = reward + self.gamma * current_reward
            discounted_reward.insert(0, current_reward)

        rewards_tensor = torch.FloatTensor(np.array(discounted_reward)).to(self.device)
        old_states = torch.FloatTensor(np.array(state)).to(self.device)

        # old_actions = torch.FloatTensor(np.array(actions)).to(self.device)
        old_logprobs = torch.FloatTensor(np.array(log_prob)).to(self.device)

        b, n, d = old_states.shape
        flat_states = old_states.view(-1, d)
        # flat_actions = old_actions.view(-1, 1)
        flat_logprobs = old_logprobs.view(-1, 1)
        flat_rewards = rewards_tensor.view(-1, 1)

        flat_rewards = (flat_rewards - flat_rewards.mean()) / (
            flat_rewards.std() + 1e-6
        )
        flat_rewards = flat_rewards.unsqueeze(1).expand(-1, n, -1).reshape(-1, 1) 

        global_state = old_states.view(b, -1)
        flat_global_state = (
            global_state.unsqueeze(1).expand(-1, n, -1).reshape(-1, n * d)
        )

        for _ in range(self.K_epochs):
            _, log_prob, entropy = self.actor_network.get_action(flat_states)
            critic_value = self.critic_network(flat_global_state)

            log_prob = log_prob.squeeze()
            ratio = torch.exp(log_prob - flat_logprobs.squeeze().detach())
            advantage = flat_rewards - critic_value.detach()

            surr_1 = ratio * advantage
            surr_2 = (
                torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
            )

            loss = (
                -torch.min(surr_1, surr_2)
                + 0.5 * F.mse_loss(flat_rewards, critic_value)
                - 0.01 * entropy
            ).mean()

            self.actor_opt.zero_grad()
            self.critic_opt.zero_grad()

            loss.backward()

            self.actor_opt.step()
            self.critic_opt.step()

        self.buffer = []
        self.old_actor_network.load_state_dict(self.actor_network.state_dict())


# ---------- Test ----------
if __name__ == "__main__":
    agent = MAPPOAgent(state_dim=8, action_dim=1, global_state_dim=96,
                       ac_lr=1e-4, cr_lr=2e-4, K_epochs=2, gamma=0.99, eps_clip=0.2)

    b = 5  # number of time steps collected in the buffer

    for t in range(b):
        state = np.random.randn(12, 8).astype(np.float32)
        action, log_prob = agent.actions(state)
        reward = float(np.random.randn())
        done = bool(t == b - 1)

        agent.store(action, state, reward, done, log_prob)

    print("Buffer length before update:", len(agent.buffer))

    agent.update(state)

    print("Buffer length after update:", len(agent.buffer))
    print("Update ran successfully.")
