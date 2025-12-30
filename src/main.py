import argparse
from datetime import datetime
import numpy as np

from utils.data_loader import SmartGridDataLoader
from envs.smart_grid_env import SmartGridEnv
from marl.agents.mappo_agent import MAPPOAgent
from utils.config_parser import Config
from utils.save_config import save_config_csv
from utils.train_logger import TrainingLogger
from utils.visualization import plot_rich_vs_poor, save_paper_visualizations

import warnings
from pandas.errors import DtypeWarning

warnings.filterwarnings("ignore", category=DtypeWarning)

def parse_args():
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments containing the config path and verbosity level.
    """
    parser = argparse.ArgumentParser(description="Run experiment with config file")

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        type=int,
        choices=[0, 1],
        default=0,
        help="Whether to show logs: 0 = no logs, 1 = show logs",
    )

    return parser.parse_args()


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
    verbose = args.verbose
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"results/run_{timestamp}"
    logger = TrainingLogger(save_dir=save_dir)

    save_config_csv(cfg, logger.save_dir)

    data_loader = SmartGridDataLoader(cfg.data_path, cfg.num_agents)
    env = SmartGridEnv(
        data_loader,
        cfg.env_ratio_clip_min_max,
        cfg.env_total_price_clip_min_max,
        cfg.actor_state_dim,
        cfg.env_scaling_factor,
        cfg.env_discomfort_weight,
        cfg.num_agents,
        cfg.num_steps_per_day
    )

    mappo_agent = MAPPOAgent(
        state_dim=cfg.actor_state_dim,
        action_dim=cfg.actor_action_dim,
        global_state_dim=cfg.critic_global_state_dim,
        ac_lr=cfg.actor_lr,
        cr_lr=cfg.critic_lr,
        K_epochs=cfg.k_epochs,
        gamma=cfg.gamma,
        eps_clip=cfg.eps_clip,
        entropy_coeff=cfg.agent_entropy_coeff,
    )

    if verbose == 1:
        print(f"Start training MAPPO Agent | Saving to {logger.save_dir}")

    for episode in range(cfg.num_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        agent_episode_rewards = np.zeros(cfg.num_agents, dtype=np.float32)

        for _ in range(cfg.num_steps_per_day):
            action, logprob, pre_tanh = mappo_agent.actions(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            mappo_agent.store(action, obs, reward, terminated, logprob, pre_tanh)

            obs = next_obs
            episode_reward += float(np.sum(reward))
            agent_episode_rewards += np.asarray(reward, dtype=np.float32)

            if terminated or truncated:
                break

        logger.log_episode(episode, episode_reward, agent_episode_rewards)

        if (episode + 1) % cfg.update_interval == 0:
            mappo_agent.update()
            logger.print_progress(episode, cfg.num_episodes, episode_reward)
            logger.save_plots()
            logger.save_data()

    if verbose == 1:
        print("Training Complete. Starting Evaluation...")

    from evaluation import evaluate_agent

    stats_table = evaluate_agent(mappo_agent, env, num_episodes=100)
    save_paper_visualizations(mappo_agent, env, save_dir)
    plot_rich_vs_poor(mappo_agent, env, save_dir)

    if verbose == 1:
        print("\n" + "=" * 50)
        print("FINAL RESULTS FOR PAPER")
        print("=" * 50)
        print(stats_table.to_string(index=False, float_format="%.2f"))
        print("=" * 50)

    stats_table.to_csv(f"{logger.save_dir}/final_paper_results.csv", index=False)


if __name__ == "__main__":
    main()
