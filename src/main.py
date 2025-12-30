import argparse
import os
import numpy as np
import torch

from utils.data_loader import SmartGridDataLoader
from envs.smart_grid_env import SmartGridEnv
from utils.config_parser import Config
from utils.train_logger import TrainingLogger
from utils.save_config import save_config_csv
from utils.experiment_paths import ExperimentPaths
from utils.visualization import save_paper_visualizations, plot_rich_vs_poor

from marl.agents.mappo_agent import MAPPOAgent
from marl.agents.ippo_agent import IPPOAgent
from marl.agents.maddpg_agent import MADDPGAgent
from marl.agents.rule_based_agent import RuleBasedAgent


def parse_args():
    """
    Parse command-line arguments for selecting config, algorithm, and experiment naming.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default="configs/config.yaml")
    parser.add_argument("--algo", type=str, default="mappo", choices=["mappo", "ippo", "maddpg", "rule"])
    parser.add_argument("--exp_name", type=str, default=None, help="Parent folder name for this experiment")
    return parser.parse_args()


def build_env(cfg):
    """
    Create and return the smart grid environment from the provided config.
    """
    data_loader = SmartGridDataLoader(cfg.data_path, cfg.num_agents)
    env = SmartGridEnv(
        data_loader,
        cfg.env_ratio_clip_min_max,
        cfg.env_total_price_clip_min_max,
        cfg.actor_state_dim,
        cfg.env_scaling_factor,
        cfg.env_discomfort_weight,
        cfg.num_agents,
        cfg.num_steps_per_day,
    )
    return env


def build_agent(cfg, algo):
    """
    Instantiate and return the requested agent.
    """
    if algo == "mappo":
        return MAPPOAgent(
            cfg.actor_state_dim,
            cfg.actor_action_dim,
            cfg.critic_global_state_dim,
            cfg.actor_lr,
            cfg.critic_lr,
            cfg.k_epochs,
            cfg.gamma,
            cfg.eps_clip,
            cfg.agent_entropy_coeff,
        )
    if algo == "ippo":
        return IPPOAgent(
            cfg.actor_state_dim,
            cfg.actor_action_dim,
            cfg.actor_lr,
            cfg.critic_lr,
            cfg.k_epochs,
            cfg.gamma,
            cfg.eps_clip,
            cfg.agent_entropy_coeff,
        )
    if algo == "maddpg":
        return MADDPGAgent(cfg.actor_state_dim, cfg.actor_action_dim, cfg.critic_global_state_dim)
    if algo == "rule":
        return RuleBasedAgent()
    raise ValueError(f"Unknown algo: {algo}")


def train(cfg, args, env, agent, logger, agent_paths):
    """
    Run the training loop and persist checkpoints and training artifacts.
    """
    for episode in range(int(cfg.num_episodes)):
        obs, _ = env.reset()
        episode_reward = 0.0

        for _ in range(int(cfg.num_steps_per_day)):
            if args.algo == "maddpg":
                action, _, _ = agent.act(obs)
                noise = np.random.normal(0.0, 0.1, size=np.asarray(action).shape)
                action = np.clip(np.asarray(action) + noise, -1.0, 1.0)
            else:
                action, logprob, pre_tanh = agent.act(obs)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)

            if args.algo == "maddpg":
                agent.store(obs, action, reward, next_obs, done)
                agent.update()
            elif args.algo in {"mappo", "ippo"}:
                agent.store(action, obs, reward, done, logprob, pre_tanh)

            obs = next_obs
            episode_reward += float(np.sum(reward))

            if done:
                break

        if args.algo in {"mappo", "ippo"} and ((episode + 1) % int(cfg.update_interval) == 0):
            agent.update()
            logger.save_data()
            logger.save_plots()

        logger.log_episode(episode, episode_reward, np.zeros(int(cfg.num_agents)))

        if (episode + 1) % 100 == 0:
            logger.print_progress(episode, int(cfg.num_episodes), episode_reward)
            agent.save(os.path.join(agent_paths["model"], "checkpoint"))

    agent.save(os.path.join(agent_paths["model"], "final_model"))
    logger.save_data()
    logger.save_plots()


def run_inference(algo, agent, env, agent_paths):
    """
    Generate and save inference visualizations for the selected agent.
    """
    if algo != "rule":
        save_paper_visualizations(agent, env, agent_paths["inference"])
        plot_rich_vs_poor(agent, env, agent_paths["inference"])
        return
    save_paper_visualizations(agent, env, agent_paths["inference"])


def main():
    """
    Entry point for training or evaluating the selected multi-agent algorithm.
    """
    args = parse_args()
    cfg = Config(args.config)

    paths = ExperimentPaths(experiment_name=args.exp_name)
    agent_paths = paths.get_agent_dirs(args.algo)

    logger = TrainingLogger(save_dir=agent_paths["training"])
    save_config_csv(cfg, agent_paths["training"])

    env = build_env(cfg)
    agent = build_agent(cfg, args.algo)

    if args.algo != "rule":
        train(cfg, args, env, agent, logger, agent_paths)

    run_inference(args.algo, agent, env, agent_paths)


if __name__ == "__main__":
    main()
