import argparse
import numpy as np
import os
import torch

from utils.data_loader import SmartGridDataLoader
from envs.smart_grid_env import SmartGridEnv
from utils.config_parser import Config
from utils.train_logger import TrainingLogger
from utils.save_config import save_config_csv
from utils.experiment_paths import ExperimentPaths
from utils.visualization import save_paper_visualizations, plot_rich_vs_poor

# Import Agents
from marl.agents.mappo_agent import MAPPOAgent
from marl.agents.ippo_agent import IPPOAgent
from marl.agents.maddpg_agent import MADDPGAgent
from marl.agents.rule_based_agent import RuleBasedAgent

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default="configs/config.yaml")
    parser.add_argument("--algo", type=str, default="mappo", choices=["mappo", "ippo", "maddpg", "rule"])
    parser.add_argument("--exp_name", type=str, default=None, help="Parent folder name for this experiment")
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = Config(args.config)
    
    # 1. Setup Directories
    # This creates: models/, training_results/, inference_images/, comparison_results/
    paths = ExperimentPaths(experiment_name=args.exp_name)
    
    # Get specific subfolders for THIS algo (e.g. training_results/mappo/)
    agent_paths = paths.get_agent_dirs(args.algo)
    
    # Initialize Logger to save inside 'training_results/{algo}'
    logger = TrainingLogger(save_dir=agent_paths["training"])
    
    # SAVE CONFIG CSV (Important: Saves config.csv to training_results/{algo}/)
    save_config_csv(cfg, agent_paths["training"])

    print(f"--- Running {args.algo.upper()} ---")
    print(f"Results will be saved to: {paths.exp_root}")

    # 2. Environment
    data_loader = SmartGridDataLoader(cfg.data_path, cfg.num_agents)
    env = SmartGridEnv(data_loader, cfg.env_ratio_clip_min_max, cfg.env_total_price_clip_min_max,
                       cfg.actor_state_dim, cfg.env_scaling_factor, cfg.env_discomfort_weight,
                       cfg.num_agents, cfg.num_steps_per_day)

    # 3. Initialize Agent
    if args.algo == "mappo":
        agent = MAPPOAgent(cfg.actor_state_dim, cfg.actor_action_dim, cfg.critic_global_state_dim,
                           cfg.actor_lr, cfg.critic_lr, cfg.k_epochs, cfg.gamma, cfg.eps_clip, cfg.agent_entropy_coeff)
    elif args.algo == "ippo":
        agent = IPPOAgent(cfg.actor_state_dim, cfg.actor_action_dim, cfg.actor_lr, cfg.critic_lr,
                          cfg.k_epochs, cfg.gamma, cfg.eps_clip, cfg.agent_entropy_coeff)
    elif args.algo == "maddpg":
        agent = MADDPGAgent(cfg.actor_state_dim, cfg.actor_action_dim, cfg.critic_global_state_dim)
    elif args.algo == "rule":
        agent = RuleBasedAgent()

    # 4. Training Loop
    if args.algo != "rule":
        for episode in range(cfg.num_episodes):
            obs, _ = env.reset()
            episode_reward = 0.0
            
            for step in range(cfg.num_steps_per_day):
                # Action
                if args.algo == "maddpg":
                    action, _, _ = agent.actions(obs)
                    noise = np.random.normal(0, 0.1, size=action.shape)
                    action = np.clip(action + noise, -1, 1)
                else:
                    action, logprob, pre_tanh = agent.actions(obs)

                # Step
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                # Store
                if args.algo == "maddpg":
                    agent.store(action, obs, reward, done, next_obs)
                    agent.update() 
                elif args.algo in ["mappo", "ippo"]:
                    agent.store(action, obs, reward, done, logprob, pre_tanh)
                
                obs = next_obs
                episode_reward += float(np.sum(reward))
                if done: break

            # On-Policy Update
            if args.algo in ["mappo", "ippo"] and (episode + 1) % cfg.update_interval == 0:
                agent.update()
                # Save Logs periodically
                logger.save_data()
                logger.save_plots()

            logger.log_episode(episode, episode_reward, np.zeros(cfg.num_agents))
            
            if (episode + 1) % 100 == 0:
                logger.print_progress(episode, cfg.num_episodes, episode_reward)
                # Save Checkpoint to 'models/{algo}/'
                agent.save(os.path.join(agent_paths["model"], "checkpoint"))
        
        # FINAL SAVES
        agent.save(os.path.join(agent_paths["model"], "final_model"))
        logger.save_data()  # Ensures training_data.csv is saved
        logger.save_plots() # Ensures training_curve.png is saved
    
    # 5. Inference Images (Post-Training)
    # Saves to 'inference_images/{algo}/'
    if args.algo != "rule": # Rule based doesn't need "training" curves, but we can do inference
        print("Generating Inference Visualizations...")
        save_paper_visualizations(agent, env, agent_paths["inference"])
        plot_rich_vs_poor(agent, env, agent_paths["inference"])
    else:
        # For Rule based, we can still generate inference plots
        save_paper_visualizations(agent, env, agent_paths["inference"])

if __name__ == "__main__":
    main()