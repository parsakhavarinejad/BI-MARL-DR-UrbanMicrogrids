import gymnasium as gym
import numpy as np
from gymnasium import spaces
from utils.data_loader import SmartGridDataLoader


class SmartGridEnv(gym.Env):
    """
    Multi-agent smart grid environment built on daily community episodes.

    This environment assumes:
    - 12 agents (homes)
    - each agent has 8 observation features per time step
    - each episode is a single day with 96 steps (15-minute intervals)

    Action
    ------
    Continuous action per agent in [-1, 1]:
        actions[i] scales the base load of agent i as:
            actual_load[i] = base_load[i] * (1 + actions[i])

    Observation
    -----------
    A matrix with shape (num_agents, 8) taken from the preprocessed data loader.

    Reward
    ------
    Per-agent reward is the negative of:
        - energy cost under a dynamic price
        - discomfort penalty proportional to action^2

        reward_i = -(actual_load_i * price_i + actions_i^2)

    Notes
    -----
    - Training uses one randomly sampled day per episode.
    - The critic (in CTDE setups) typically uses the global state formed by
      concatenating all agent observations.
    """

    def __init__(self, data_loader):
        """
        Initialize the environment.

        Parameters
        ----------
        data_loader : SmartGridDataLaoder
            Prepared dataset loader that provides daily episodes with shape
            (num_agents, 96, 8).
        """
        super().__init__()
        self.data_loader = data_loader
        self.num_agents = 12

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_agents,), dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_agents, 8), dtype=np.float32
        )

        self.current_step = 0
        self.day_data = None

        self.expected_load = (
            self.data_loader.daily_episodes[:, :, :, 0].sum(axis=1).mean(axis=0)
        )

        self.total_steps = 96

    def reset(self, seed=None, options=None):
        """
        Reset the environment and sample a new day episode.

        Parameters
        ----------
        seed : int, optional
            Random seed passed to Gymnasium.
        options : dict, optional
            Gymnasium reset options (unused).

        Returns
        -------
        obs : np.ndarray
            Initial observation with shape (num_agents, 8).
        info : dict
            Empty info dict (placeholder for Gymnasium compatibility).
        """
        super().reset(seed=seed)

        day_idx = np.random.randint(len(self.data_loader))
        self.day_data = self.data_loader.get_episode(day_idx)

        self.current_step = 0
        return self._get_obs(), {}

    def _get_dynamic_price(self, total_grid_load, base_price, alpha=0.5):
        """
        Compute dynamic energy price based on total grid load.

        Parameters
        ----------
        total_grid_load : float
            Total community load at the current time step.
        base_price : np.ndarray
            Base price per agent at the current time step, shape (num_agents,).
        alpha : float, optional
            Strength of the price response to load (default: 0.5).

        Returns
        -------
        np.ndarray
            Dynamic price per agent, shape (num_agents,).
        """
        total_price = base_price * (
            1 + alpha * (total_grid_load / self.expected_load[self.current_step]) ** 2
        )
        return total_price

    def step(self, actions):
        """
        Apply actions for all agents and advance one time step.

        Parameters
        ----------
        actions : np.ndarray
            Array of actions in [-1, 1] with shape (num_agents,).

        Returns
        -------
        next_obs : np.ndarray
            Next observation, shape (num_agents, 8) while episode is running.
            If done, returns a zero array (kept for structure compatibility).
        rewards : np.ndarray
            Per-agent rewards, shape (num_agents,).
        done : bool
            True if the episode finished (end of day).
        truncated : bool
            Always False in this environment.
        info : dict
            Empty info dict.
        """
        obs = self._get_obs()
        current_base_load = obs[:, 0]
        current_base_price = obs[:, 1]

        actions = np.squeeze(actions)

        actions = np.clip(actions, -1.0, 1.0)

        actual_load = current_base_load * (1 + actions)
        total_grid_load = np.sum(actual_load, axis=0)

        current_price = self._get_dynamic_price(total_grid_load, current_base_price)

        scaling_factor = 1 / 2000
        rewards = []
        for agent in range(self.num_agents):
            cost = actual_load[agent] * current_price[agent]
            discomfort = (actions[agent]) ** 2
            reward = -(cost + discomfort) * scaling_factor
            rewards.append(reward)

        self.current_step += 1
        done = self.current_step >= self.total_steps

        next_obs = (
            self._get_obs()
            if not done
            else np.zeros((self.num_agents, 3), dtype=np.float32)
        )

        return next_obs, np.array(rewards, dtype=np.float32), done, False, {}

    def _get_obs(self):
        """
        Fetch the observation at the current time step.

        Returns
        -------
        np.ndarray
            Observation matrix with shape (num_agents, 8).
        """
        current_values_feature = self.day_data[:, self.current_step, :].astype(np.float32)

        obs = current_values_feature.copy()
        obs[:, 0] = obs[:, 0] / 10.0 
        obs[:, 1] = obs[:, 1] / 50.0

        return obs


# -------- Test the script --------
if __name__ == "__main__":
    dataloader = SmartGridDataLoader("data\\IDEAL\\panel_env_ready_15m.csv.gz")
    env = SmartGridEnv(dataloader)

    obs, _ = env.reset()
    action = np.random.rand(12).astype(np.float32)

    ob, rew, _, _, _ = env.step(action)

    print("Observations shape is: ", obs.shape)
    print("Reward is: ", rew.shape)
