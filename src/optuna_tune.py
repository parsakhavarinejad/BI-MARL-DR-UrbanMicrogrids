# python src/optuna_tune.py --config configs/config.yaml --trials 30 --train_episodes 20 --eval_episodes 10 --seed 0

import argparse
import os
import random
from datetime import datetime

import numpy as np
import optuna
import torch

from utils.config_parser import Config
from utils.data_loader import SmartGridDataLoader
from envs.smart_grid_env import SmartGridEnv
from marl.agents.mappo_agent import MAPPOAgent


def set_global_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def team_step_reward(reward) -> float:
    """
    team reward = sum across agents each step.
    """
    return float(np.sum(reward))


def eval_mean_episode_return(
    agent: MAPPOAgent, env: SmartGridEnv, num_episodes: int, max_steps: int
) -> float:
    """
    Evaluate agent with deterministic actions, return mean undiscounted episode return.
    """
    returns = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        ep_return = 0.0

        for _t in range(max_steps):
            actions, _, _ = agent.actions_deterministic(obs)
            obs, reward, terminated, truncated, _ = env.step(actions)
            ep_return += team_step_reward(reward)

            if terminated or truncated:
                break

        returns.append(ep_return)

    return float(np.mean(returns))


def train_short_budget(
    cfg: Config,
    hp: dict,
    train_episodes: int,
    eval_episodes: int,
    seed: int,
    trial: optuna.Trial,
) -> float:
    """
    1) Build env + agent using cfg and sampled hyperparams hp
    2) Train for a small number of episodes
    3) Evaluate deterministic mean episode return
    4) Return score (to maximize)
    """
    set_global_seeds(seed)

    data_loader = SmartGridDataLoader(cfg.data_path)
    env = SmartGridEnv(
        data_loader,
        cfg.env_ratio_clip_min_max,
        cfg.env_total_price_clip_min_max,
        cfg.actor_state_dim,
        cfg.env_scaling_factor,
        hp["discomfort_weight"],
    )

    agent = MAPPOAgent(
        state_dim=cfg.actor_state_dim,
        action_dim=cfg.actor_action_dim,
        global_state_dim=cfg.critic_global_state_dim,
        ac_lr=hp["actor_lr"],
        cr_lr=hp["critic_lr"],
        K_epochs=hp["k_epochs"],
        gamma=hp["gamma"],
        eps_clip=hp["eps_clip"],
        entropy_coeff=hp["entropy_coeff"],
    )

    update_interval = int(hp["update_interval"])

    for episode in range(train_episodes):
        obs, _ = env.reset()
        ep_return = 0.0

        for _ in range(cfg.num_steps_per_day):
            action, logprob, pre_tanh = agent.actions(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            agent.store(action, obs, reward, terminated, logprob, pre_tanh)

            obs = next_obs
            ep_return += team_step_reward(reward)

            if terminated or truncated:
                break

        if (episode + 1) % update_interval == 0:
            agent.update()

        trial.report(ep_return, step=episode)
        if trial.should_prune():
            raise optuna.TrialPruned()

    if len(agent.buffer) > 0:
        agent.update()

    score = eval_mean_episode_return(
        agent=agent,
        env=env,
        num_episodes=eval_episodes,
        max_steps=cfg.num_steps_per_day,
    )
    return score


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", "-c", type=str, default="configs/config.yaml")
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--train_episodes", type=int, default=20)
    p.add_argument("--eval_episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--study_name", type=str, default="mappo_optuna")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config(args.config)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("results", f"optuna_{args.study_name}_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    storage_path = f"sqlite:///{os.path.join(out_dir, 'study.db')}"
    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        storage=storage_path,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )

    def objective(trial: optuna.Trial) -> float:
        hp = {
            # learning rates
            "actor_lr": trial.suggest_float("actor_lr", 1e-5, 3e-4, log=True),
            "critic_lr": trial.suggest_float("critic_lr", 1e-5, 1e-3, log=True),
            # PPO/MAPPO core
            "gamma": 0.99, # trial.suggest_float("gamma", 0.92, 0.999),
            "eps_clip": trial.suggest_float("eps_clip", 0.1, 0.4),
            "k_epochs": trial.suggest_int("k_epochs", 3, 20),
            "entropy_coeff": trial.suggest_float("entropy_coeff", 0.0, 0.005),
            # training schedule
            "update_interval": trial.suggest_int("update_interval", 15, 20),
            "discomfort_weight": cfg.env_discomfort_weight
        }

        score = train_short_budget(
            cfg=cfg,
            hp=hp,
            train_episodes=args.train_episodes,
            eval_episodes=args.eval_episodes,
            seed=args.seed,
            trial=trial,
        )
        return score

    study.optimize(objective, n_trials=args.trials)

    best = study.best_trial
    best_txt = os.path.join(out_dir, "best_trial.txt")
    with open(best_txt, "w", encoding="utf-8") as f:
        f.write(f"Best value (mean eval episode return): {best.value}\n")
        f.write("Best params:\n")
        for k, v in best.params.items():
            f.write(f"  {k}: {v}\n")

    print("=" * 60)
    print("OPTUNA DONE")
    print(f"Results saved in: {out_dir}")
    print(f"Best value (maximize): {best.value}")
    print("Best params:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")
    print("=" * 60)


if __name__ == "__main__":
    main()
