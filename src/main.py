import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime

from utils.data_loader import SmartGridDataLoader
from envs.smart_grid_env import SmartGridEnv
from marl.agents.mappo_agent import MAPPOAgent
from utils.config_parser import Config


def parse_args():
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments containing the config path.
    """
    parser = argparse.ArgumentParser(description="Input config directory")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file",
    )
    return parser.parse_args()


class TrainingLogger:
    """
    Lightweight training logger for tracking rewards and saving artifacts.

    This logger tracks:
    - Total episode reward (sum over agents)
    - Moving average reward over a sliding window
    - Per-agent episode rewards

    It can periodically save:
    - Training curve plot (PNG)
    - Episode reward data (CSV)

    Parameters
    ----------
    save_dir : str, optional
        Directory where plots and CSV logs will be saved (default: "results").
    """

    def __init__(self, save_dir="results"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.episode_rewards = []
        self.avg_rewards = []
        self.agent_rewards = {}

    def log_episode(self, episode, total_reward, individual_rewards):
        """
        Log rewards for a completed episode.

        Parameters
        ----------
        episode : int
            Episode index (0-based).
        total_reward : float
            Total reward for the episode (typically sum over all agents and steps).
        individual_rewards : np.ndarray
            Per-agent total rewards for the episode, shape (num_agents,).
        """
        self.episode_rewards.append(total_reward)

        window = 10
        avg_r = np.mean(self.episode_rewards[-window:])
        self.avg_rewards.append(avg_r)

        for agent_id, r in enumerate(individual_rewards):
            if agent_id not in self.agent_rewards:
                self.agent_rewards[agent_id] = []
            self.agent_rewards[agent_id].append(float(r))

    def print_progress(self, episode, total_episodes, total_reward):
        """
        Print a short training progress line.

        Parameters
        ----------
        episode : int
            Current episode index (0-based).
        total_episodes : int
            Total number of training episodes.
        total_reward : float
            Total reward achieved in the current episode.
        """
        avg_r = self.avg_rewards[-1] if self.avg_rewards else float("nan")
        print(
            f"Episode {episode+1}/{total_episodes} | Total Reward: {total_reward:.2f} | MA(10): {avg_r:.2f}"
        )

    def save_plots(self):
        """
        Save the training curve plot as a PNG in `save_dir`.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(self.episode_rewards, label="Episode Reward", alpha=0.4)
        plt.plot(self.avg_rewards, label="Moving Avg (10)", color="red")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Training Progress")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, "training_curve.png"))
        plt.close()

    def save_data(self):
        """
        Save episode reward history and moving average as a CSV in `save_dir`.
        """
        df = pd.DataFrame(
            {"episode_reward": self.episode_rewards, "moving_avg": self.avg_rewards}
        )
        df.to_csv(os.path.join(self.save_dir, "training_data.csv"), index=False)


def main():
    """
    Main training entrypoint.

    Workflow:
    1) Load configuration
    2) Initialize logger with a timestamped output directory
    3) Build dataset loader and environment
    4) Initialize MAPPO agent
    5) Run training loop and periodically save plots + CSV logs
    """
    args = parse_args()
    cfg = Config(args.config)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = TrainingLogger(save_dir=f"results/run_{timestamp}")

    data_loader = SmartGridDataLoader(cfg.data_path)
    env = SmartGridEnv(data_loader)

    mappo_agent = MAPPOAgent(
        state_dim=cfg.actor_state_dim,
        action_dim=cfg.actor_action_dim,
        global_state_dim=cfg.critic_global_state_dim,
        ac_lr=cfg.actor_lr,
        cr_lr=cfg.critic_lr,
        K_epochs=cfg.k_epochs,
        gamma=cfg.gamma,
        eps_clip=cfg.eps_clip,
    )

    print(f"Start training MAPPO Agent | Saving to {logger.save_dir}")

    for episode in range(cfg.num_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        agent_episode_rewards = np.zeros(cfg.num_agents, dtype=np.float32)

        for _ in range(cfg.num_steps_per_day):
            action, logprob = mappo_agent.actions(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            mappo_agent.store(action, obs, reward, terminated, logprob)

            obs = next_obs
            episode_reward += float(np.sum(reward))
            agent_episode_rewards += np.asarray(reward, dtype=np.float32)

            if terminated or truncated:
                break

        mappo_agent.update()

        logger.log_episode(episode, episode_reward, agent_episode_rewards)

        if (episode + 1) % cfg.update_interval == 0:
            logger.print_progress(episode, cfg.num_episodes, episode_reward)
            logger.save_plots()
            logger.save_data()

    print("Training Complete")


if __name__ == "__main__":
    main()
