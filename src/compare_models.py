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

def load_agent(algo, base_path, cfg):
    if algo == "mappo":
        agent = MAPPOAgent(cfg.actor_state_dim, cfg.actor_action_dim, cfg.critic_global_state_dim,
                           cfg.actor_lr, cfg.critic_lr, cfg.k_epochs, cfg.gamma, cfg.eps_clip, cfg.agent_entropy_coeff)
    elif algo == "ippo":
        agent = IPPOAgent(cfg.actor_state_dim, cfg.actor_action_dim, cfg.actor_lr, cfg.critic_lr,
                          cfg.k_epochs, cfg.gamma, cfg.eps_clip, cfg.agent_entropy_coeff)
    elif algo == "maddpg":
        agent = MADDPGAgent(cfg.actor_state_dim, cfg.actor_action_dim, cfg.critic_global_state_dim)
    elif algo == "rule":
        return RuleBasedAgent()
        
    if algo != "rule":
        print(f"Loading {algo} weights from {base_path}")
        agent.load(base_path)
    return agent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True, help="The folder name in results/ to analyze")
    args = parser.parse_args()

    cfg = Config("configs/config.yaml")
    paths = ExperimentPaths(experiment_name=args.exp_name)
    
    comp_dir = paths.get_comparison_dirs()

    algos = ["rule", "ippo", "maddpg", "mappo"]
    
    data_loader = SmartGridDataLoader(cfg.data_path, cfg.num_agents)
    env = SmartGridEnv(data_loader, cfg.env_ratio_clip_min_max, cfg.env_total_price_clip_min_max,
                       cfg.actor_state_dim, cfg.env_scaling_factor, cfg.env_discomfort_weight,
                       cfg.num_agents, cfg.num_steps_per_day)

    all_summaries = []

    for algo in algos:
        model_path = os.path.join(paths.models_dir, algo, "final_model")
        
        if algo != "rule" and not os.path.exists(model_path + "_actor.pth"):
            print(f"Skipping {algo}: Model not found at {model_path}")
            continue

        agent = load_agent(algo, model_path, cfg)
        
        print(f"--- Evaluating {algo} ---")
        df = evaluate_agent(agent, env, num_episodes=50)
        
        perf = df[["Metric", "MAPPO (Mean)"]].copy()
        perf.columns = ["Metric", algo.upper()]
        perf.set_index("Metric", inplace=True)
        all_summaries.append(perf)

    if not all_summaries:
        print("No models found to compare.")
        return

    final_table = pd.concat(all_summaries, axis=1)
    print("\nFINAL COMPARISON TABLE")
    print(final_table)
    
    csv_path = os.path.join(comp_dir, "final_comparison.csv")
    final_table.to_csv(csv_path)
    print(f"Saved table to: {csv_path}")
    
    final_table.T.plot(kind="bar", figsize=(12, 6), rot=0)
    plt.title("Algorithm Performance Comparison")
    plt.ylabel("Value")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    chart_path = os.path.join(comp_dir, "comparison_chart.png")
    plt.savefig(chart_path)
    print(f"Saved chart to: {chart_path}")

if __name__ == "__main__":
    main()