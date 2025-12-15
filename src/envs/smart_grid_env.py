import gymnasium as gym
import numpy as np
from gymnasium import spaces
from utils.data_loader import SmartGridDataLaoder


class SmartGridEnv(gym.Env):
    def __init__(self, data_loader):
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
        super().reset(seed=seed)
        day_idx = np.random.randint(len(self.data_loader))
        self.day_data = self.data_loader.get_episode(day_idx)

        self.current_step = 0
        return self._get_obs(), {}

    def _get_dynamic_price(self, total_grid_load, base_price, alpha=0.5):

        total_price = base_price * (
            1 + alpha * (total_grid_load / self.expected_load[self.current_step]) ** 2
        )
        return total_price

    def step(self, actions):
        obs = self._get_obs()
        current_base_load = obs[:, 0]
        current_base_price = obs[:, 1]

        actual_load = current_base_load * (1 + actions)
        total_grid_load = np.sum(actual_load, axis=0)
        current_price = self._get_dynamic_price(total_grid_load, current_base_price)

        rewards = []
        for agent in range(self.num_agents):
            cost = actual_load[agent] * current_price[agent]
            discomfort = (actions[agent]) ** 2

            reward = -(cost + discomfort)
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
        return self.day_data[:, self.current_step, :].astype(np.float32)


if __name__ == "__main__":
    dataloader = SmartGridDataLaoder("data\IDEAL\panel_env_ready_15m.csv.gz")
    env = SmartGridEnv(dataloader)
    obs, _ = env.reset()
    action = np.random.rand(12)
    ob, rew, _, _, _ = env.step(action)
    print("Observations shape is: ", obs.shape)
    print("Reward is: ", rew)
