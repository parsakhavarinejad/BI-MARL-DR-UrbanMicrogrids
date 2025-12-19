import numpy as np
import pandas as pd


def evaluate_agent(agent, env, num_episodes=50):
    """
    Runs the agent for multiple episodes to gather robust statistics.
    Returns a Pandas DataFrame suitable for paper tables.
    """
    print(f"Starting Evaluation over {num_episodes} episodes...")

    results = {
        "par_base": [],
        "par_opt": [],
        "cost_base": [],
        "cost_opt": [],
        "peak_base": [],
        "peak_opt": [],
        "discomfort": [],
    }

    for i in range(num_episodes):
        obs, _ = env.reset()
        done = False

        base_loads = []
        opt_loads = []
        real_prices = []
        actions_sq = []

        while not done:
            actions, _, _ = agent.actions(obs)

            raw = env._get_obs_raw()
            current_base_load = raw[:, 0].copy()
            current_base_price = raw[:, 1].copy()

            clipped_actions = np.clip(actions.reshape(-1), -1.0, 1.0)
            actual_load = current_base_load * (1 + clipped_actions)

            total_grid_load = np.sum(actual_load)
            real_price_vector = env._get_dynamic_price(
                total_grid_load, current_base_price
            )

            obs, _, done, _, _ = env.step(actions)

            base_loads.append(current_base_load)
            opt_loads.append(actual_load)
            real_prices.append(real_price_vector)
            actions_sq.append(clipped_actions**2)

        base_loads = np.array(base_loads)  # (96, 12)
        opt_loads = np.array(opt_loads)  # (96, 12)
        real_prices = np.array(real_prices)  # (96, 12)
        actions_sq = np.array(actions_sq)  # (96, 12)

        comm_base = np.sum(base_loads, axis=1)
        comm_opt = np.sum(opt_loads, axis=1)

        par_b = np.max(comm_base) / (np.mean(comm_base) + 1e-6)
        par_o = np.max(comm_opt) / (np.mean(comm_opt) + 1e-6)

        cost_b = np.sum(base_loads * real_prices)
        cost_o = np.sum(opt_loads * real_prices)

        peak_b = np.max(comm_base)
        peak_o = np.max(comm_opt)

        disc_score = np.mean(actions_sq)

        results["par_base"].append(par_b)
        results["par_opt"].append(par_o)
        results["cost_base"].append(cost_b)
        results["cost_opt"].append(cost_o)
        results["peak_base"].append(peak_b)
        results["peak_opt"].append(peak_o)
        results["discomfort"].append(disc_score)

    df = pd.DataFrame(results)

    summary = pd.DataFrame(
        {
            "Metric": ["PAR", "Total Cost", "Peak Load", "Avg Discomfort"],
            "Baseline (Mean)": [
                df["par_base"].mean(),
                df["cost_base"].mean(),
                df["peak_base"].mean(),
                0.0,
            ],
            "MAPPO (Mean)": [
                df["par_opt"].mean(),
                df["cost_opt"].mean(),
                df["peak_opt"].mean(),
                df["discomfort"].mean(),
            ],
            "Improvement (%)": [
                (df["par_base"].mean() - df["par_opt"].mean())
                / df["par_base"].mean()
                * 100,
                (df["cost_base"].mean() - df["cost_opt"].mean())
                / df["cost_base"].mean()
                * 100,
                (df["peak_base"].mean() - df["peak_opt"].mean())
                / df["peak_base"].mean()
                * 100,
                np.nan,
            ],
        }
    )

    return summary
