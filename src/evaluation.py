import numpy as np
import pandas as pd


def evaluate_agent(agent, env, num_episodes=50):
    """
    Corrected for:
    (2) Baseline vs Opt costs use their own price trajectories (dynamic pricing fairness)
    (3) Logged loads match the actions actually applied to env.step() (clip consistency)

    Assumptions:
    - env.step() accepts the same action shape returned by agent.actions(obs)
    - env._get_obs_raw() returns an array shaped (n_agents, n_features) where:
        raw[:, 0] = baseline load at current step
        raw[:, 1] = base price signal at current step
    - env._get_dynamic_price(total_grid_load, base_price_vector) returns price vector per agent
      for the current step (same shape as base_price_vector)
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

    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False

        base_loads = []
        opt_loads = []
        prices_base = []
        prices_opt = []
        actions_sq = []

        while not done:
            # 1) Agent proposes actions
            actions, _, _ = agent.actions(obs)

            # 2) Read raw baseline signals for THIS step
            raw = env._get_obs_raw()
            current_base_load = raw[:, 0].copy()
            current_base_price = raw[:, 1].copy()

            # 3) Clip actions and ensure we STEP with the same clipped actions
            clipped_flat = np.clip(np.asarray(actions).reshape(-1), -1.0, 1.0)
            env_actions = clipped_flat.reshape(np.asarray(actions).shape)

            # 4) Compute baseline and optimized loads for logging
            actual_load = current_base_load * (1.0 + clipped_flat)

            # 5) Compute TWO price trajectories (baseline vs optimized),
            #    because dynamic price depends on total grid load.
            total_grid_load_base = float(np.sum(current_base_load))
            total_grid_load_opt = float(np.sum(actual_load))

            price_base_vec = env._get_dynamic_price(total_grid_load_base, current_base_price)
            price_opt_vec = env._get_dynamic_price(total_grid_load_opt, current_base_price)

            # 6) Step environment using the SAME actions used in logging
            obs, _, terminated, truncated, _ = env.step(env_actions)
            done = bool(terminated or truncated)

            # 7) Store trajectories
            base_loads.append(current_base_load)
            opt_loads.append(actual_load)
            prices_base.append(price_base_vec)
            prices_opt.append(price_opt_vec)
            actions_sq.append(clipped_flat ** 2)

        # Convert to arrays: (T, n_agents)
        base_loads = np.asarray(base_loads)
        opt_loads = np.asarray(opt_loads)
        prices_base = np.asarray(prices_base)
        prices_opt = np.asarray(prices_opt)
        actions_sq = np.asarray(actions_sq)

        # Community load per step
        comm_base = np.sum(base_loads, axis=1)
        comm_opt = np.sum(opt_loads, axis=1)

        # PAR
        par_b = np.max(comm_base) / (np.mean(comm_base) + 1e-6)
        par_o = np.max(comm_opt) / (np.mean(comm_opt) + 1e-6)

        cost_b = np.sum(base_loads * prices_base)
        cost_o = np.sum(opt_loads * prices_opt)

        # Peak
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
                / (df["par_base"].mean() + 1e-12)
                * 100,
                (df["cost_base"].mean() - df["cost_opt"].mean())
                / (df["cost_base"].mean() + 1e-12)
                * 100,
                (df["peak_base"].mean() - df["peak_opt"].mean())
                / (df["peak_base"].mean() + 1e-12)
                * 100,
                np.nan,
            ],
        }
    )

    return summary
