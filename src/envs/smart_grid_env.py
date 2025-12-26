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

    def __init__(
        self,
        data_loader,
        ratio_clip,
        total_price_clip,
        state_dim,
        scaling_factor,
        discomfort_weight,
        num_agents,
        num_steps_per_day
    ):
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
        self.num_agents = num_agents

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_agents,), dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_agents, state_dim),
            dtype=np.float32,
        )

        self.current_step = 0
        self.day_data = None

        self.expected_load = (
            self.data_loader.daily_episodes[:, :, :, 0].sum(axis=1).mean(axis=0)
        )

        self.total_steps = num_steps_per_day

        self.ratio_clip = ratio_clip
        self.total_price_clip = total_price_clip
        self.state_dim = state_dim
        self.scaling_factor = scaling_factor
        self.discomfort_weight = discomfort_weight

        self.load_clip = float(self.data_loader.load_clip)
        self.price_min = float(self.data_loader.price_min)
        self.price_max = float(self.data_loader.price_max)

        self.expected_load_norm = (
            self.data_loader.daily_episodes[:, :, :, 0].sum(axis=1).mean(axis=0)
        )

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

    def _get_dynamic_price_real(self, total_grid_load_norm, base_price_real, alpha=0.5):
        """
        Calculates the REAL dynamic price (in Pence).
        
        Args:
            total_grid_load_norm: The sum of NORMALIZED loads (dimensionless 0-1 scale sum).
            base_price_real: The BASE price in REAL Pence/kWh.
        """
        den = max(self.expected_load_norm[self.current_step], 1e-3)
        
        ratio = total_grid_load_norm / den
        ratio = np.clip(ratio, self.ratio_clip[0], self.ratio_clip[1])
        
        total_price_real = base_price_real * (1 + alpha * ratio**2)
        
        total_price_real = np.clip(
            total_price_real,
            self.total_price_clip[0] * base_price_real,
            self.total_price_clip[1] * base_price_real,
        )
        return total_price_real

    def step(self, actions):
        raw_norm = self._get_obs_raw_norm()
        norm_base_load = raw_norm[:, 0]  
        norm_base_price = raw_norm[:, 1] 

        real_base_load = norm_base_load * self.load_clip
        real_base_price = norm_base_price * (self.price_max - self.price_min) + self.price_min

        actions = np.squeeze(actions)
        actions = np.clip(actions, -1.0, 1.0)

        real_actual_load = real_base_load * (1 + actions)
        
        norm_actual_load = norm_base_load * (1 + actions)
        total_grid_load_norm = np.sum(norm_actual_load, axis=0)

        real_dynamic_price = self._get_dynamic_price_real(total_grid_load_norm, real_base_price)

        rewards = []
        for i in range(self.num_agents):
            cost_pence = real_actual_load[i] * real_dynamic_price[i]
            discomfort = (actions[i] ** 2) * self.discomfort_weight

            reward = -(cost_pence + discomfort) * self.scaling_factor
            rewards.append(reward)

        self.current_step += 1
        done = self.current_step >= self.total_steps

        next_obs = (
            self._get_obs()
            if not done
            else np.zeros((self.num_agents, self.state_dim), dtype=np.float32)
        )

        return next_obs, np.array(rewards, dtype=np.float32), done, False, {}

    def _get_obs(self):
        """
        Returns Observation for Agent.
        Data is already [0, 1]. We scale it to [-1, 1] for better tanh performance.
        """
        obs = self._get_obs_raw_norm().copy()

        obs[:, 0] = 2.0 * obs[:, 0] - 1.0
        obs[:, 1] = 2.0 * obs[:, 1] - 1.0

        return obs.astype(np.float32)

    def _get_obs_raw_norm(self):
        """Returns the [0, 1] normalized data directly from the loader."""
        return self.day_data[:, self.current_step, :].astype(np.float32)


# -------- Test the script --------
if __name__ == "__main__":
    dataloader = SmartGridDataLoader("data\\IDEAL\\panel_env_ready_15m.csv.gz")
    env = SmartGridEnv(dataloader)

    obs, _ = env.reset()
    action = np.random.rand(12).astype(np.float32)

    ob, rew, _, _, _ = env.step(action)

    print("Observations shape is: ", obs.shape)
    print("Reward is: ", rew.shape)
