import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os
from envs.smart_grid_env import SmartGridEnv
from utils.data_loader import SmartGridDataLoader
from utils.config_parser import Config
from evaluation import evaluate_agent
from utils.experiment_paths import ExperimentPaths

from marl.agents.mappo_agent import MAPPOAgent
from marl.agents.ippo_agent import IPPOAgent
from marl.agents.maddpg_agent import MADDPGAgent
from marl.agents.rule_based_agent import RuleBasedAgent
from marl.agents.tou_agent import TOUAgent 
from marl.agents.matd3_agent import MATD3Agent

def load_agent(algo, base_path, cfg):
    if algo == "mappo":
        agent = MAPPOAgent(cfg.actor_state_dim, cfg.actor_action_dim, cfg.critic_global_state_dim,
                           cfg.actor_lr, cfg.critic_lr, cfg.k_epochs, cfg.gamma, cfg.eps_clip, cfg.agent_entropy_coeff)
    elif algo == "ippo":
        agent = IPPOAgent(cfg.actor_state_dim, cfg.actor_action_dim, cfg.actor_lr, cfg.critic_lr,
                          cfg.k_epochs, cfg.gamma, cfg.eps_clip, cfg.agent_entropy_coeff)
    elif algo == "maddpg":
        agent = MADDPGAgent(cfg.actor_state_dim, cfg.actor_action_dim, cfg.critic_global_state_dim)
    elif algo == "matd3":
        agent = MATD3Agent(cfg.actor_state_dim, cfg.actor_action_dim, cfg.critic_global_state_dim)
    elif algo == "rule":
        return RuleBasedAgent()
    elif algo == "tou": # Add TOU case
        return TOUAgent()
        
    if algo not in ["rule", "tou"]:
        if os.path.exists(base_path + "_actor.pth"):
            print(f"Loading {algo} weights from {base_path}")
            agent.load(base_path)
        else:
            print(f"Warning: Model not found for {algo}")
            
    return agent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True)
    args = parser.parse_args()

    cfg = Config("configs/config.yaml")
    paths = ExperimentPaths(experiment_name=args.exp_name)
    comp_dir = paths.get_comparison_dirs()

    algos = ["rule", "tou", "ippo", "maddpg", "mappo", "matd3"]
    
    data_loader = SmartGridDataLoader(cfg.data_path, cfg.num_agents)
    env = SmartGridEnv(data_loader, cfg.env_ratio_clip_min_max, cfg.env_total_price_clip_min_max,
                       cfg.actor_state_dim, cfg.env_scaling_factor, cfg.env_discomfort_weight,
                       cfg.num_agents, cfg.num_steps_per_day)

    all_summaries = []

    for algo in algos:
        model_path = os.path.join(paths.models_dir, algo, "final_model")
        agent = load_agent(algo, model_path, cfg)
        
        print(f"--- Evaluating {algo} ---")
        df = evaluate_agent(agent, env, num_episodes=50)
        
        if df is not None and not df.empty:
            perf = df[["Metric", "Value"]].copy()
            perf.columns = ["Metric", algo.upper()]
            perf.set_index("Metric", inplace=True)
            all_summaries.append(perf)

    if not all_summaries:
        print("No models found to compare.")
        return

    final_table = pd.concat(all_summaries, axis=1)
    print("\nFINAL COMPARISON TABLE")
    print(final_table)
    final_table.to_csv(os.path.join(comp_dir, "final_comparison.csv"))
    
    final_table.T.plot(kind="bar", figsize=(12, 6), rot=0)
    plt.tight_layout()
    plt.savefig(os.path.join(comp_dir, "comparison_chart.png"))

if __name__ == "__main__":
    main()