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

# --- IMPORT ALL AGENTS ---
from marl.agents.mappo_agent import MAPPOAgent
from marl.agents.ippo_agent import IPPOAgent
from marl.agents.maddpg_agent import MADDPGAgent
from marl.agents.rule_based_agent import RuleBasedAgent
from marl.agents.tou_agent import TOUAgent    
from marl.agents.matd3_agent import MATD3Agent


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default="configs/config.yaml")
    parser.add_argument(
        "--algo", 
        type=str, 
        default="mappo", 
        choices=["mappo", "ippo", "maddpg", "matd3", "rule", "tou"],
        help="Algorithm to run: [mappo, ippo, maddpg, matd3, rule, tou]"
    )
    parser.add_argument("--exp_name", type=str, default=None, help="Parent folder name for this experiment")
    
    parser.add_argument("--num_episodes", type=int, default=None, help="Override number of episodes from config")
    
    return parser.parse_args()


def build_env(cfg):
    """
    Create the Smart Grid Environment.
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
    Instantiate the selected agent with config parameters.
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
        return MADDPGAgent(
            cfg.actor_state_dim, 
            cfg.actor_action_dim, 
            cfg.critic_global_state_dim,
            actor_lr=cfg.actor_lr,
            critic_lr=cfg.critic_lr
        )
    
    if algo == "matd3":
        return MATD3Agent(
            cfg.actor_state_dim, 
            cfg.actor_action_dim, 
            cfg.critic_global_state_dim,
            actor_lr=cfg.actor_lr,
            critic_lr=cfg.critic_lr
        )

    if algo == "rule":
        return RuleBasedAgent()
    
    if algo == "tou":
        return TOUAgent()

    raise ValueError(f"Unknown algo: {algo}")


def train(cfg, args, env, agent, logger, agent_paths):
    """
    Main training loop.
    """
    total_episodes = args.num_episodes if args.num_episodes is not None else int(cfg.num_episodes)
    
    off_policy_algos = ["maddpg", "matd3"]
    on_policy_algos = ["mappo", "ippo"]
    heuristics = ["rule", "tou"]

    print(f"Starting execution for {args.algo} over {total_episodes} episodes...")

    for episode in range(total_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0

        for _ in range(int(cfg.num_steps_per_day)):
            if args.algo in off_policy_algos:
                action, _, _ = agent.act(obs)
                noise = np.random.normal(0.0, 0.1, size=np.asarray(action).shape)
                action = np.clip(np.asarray(action) + noise, -1.0, 1.0)
                logprob, pre_tanh = None, None
            else:
                action, logprob, pre_tanh = agent.act(obs)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)

            if args.algo not in heuristics:
                if args.algo in off_policy_algos:
                    agent.store(obs, action, reward, next_obs, done)
                    agent.update()
                elif args.algo in on_policy_algos:
                    agent.store(action, obs, reward, done, logprob, pre_tanh)

            obs = next_obs
            episode_reward += float(np.sum(reward))

            if done:
                break

        if args.algo in on_policy_algos and ((episode + 1) % int(cfg.update_interval) == 0):
            agent.update()
            logger.save_data()
            logger.save_plots()

        logger.log_episode(episode, episode_reward, np.zeros(int(cfg.num_agents)))

        if (episode + 1) % 100 == 0:
            logger.print_progress(episode, total_episodes, episode_reward)
            if args.algo not in heuristics:
                agent.save(os.path.join(agent_paths["model"], "checkpoint"))

    if args.algo not in heuristics:
        agent.save(os.path.join(agent_paths["model"], "final_model"))
    
    logger.save_data()
    logger.save_plots()
    print("Training/Execution complete.")


def run_inference(algo, agent, env, agent_paths):
    """
    Generate plots and evaluation metrics.
    """
    print("Generating inference plots...")
    
    save_paper_visualizations(agent, env, agent_paths["inference"])
    
    plot_rich_vs_poor(agent, env, agent_paths["inference"])


def main():
    args = parse_args()
    cfg = Config(args.config)

    paths = ExperimentPaths(experiment_name=args.exp_name)
    agent_paths = paths.get_agent_dirs(args.algo)

    logger = TrainingLogger(save_dir=agent_paths["training"])
    save_config_csv(cfg, agent_paths["training"])

    env = build_env(cfg)
    agent = build_agent(cfg, args.algo)

    train(cfg, args, env, agent, logger, agent_paths)

    run_inference(args.algo, agent, env, agent_paths)


if __name__ == "__main__":
    main()