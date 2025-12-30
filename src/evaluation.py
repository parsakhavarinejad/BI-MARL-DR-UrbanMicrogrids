import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def calculate_jains_index(values):
    """
    Compute Jain's fairness index for a vector of values.
    """
    values_arr = np.asarray(values, dtype=np.float64)
    values_arr = np.maximum(values_arr, 0.0)

    n = values_arr.size
    if n == 0:
        return 0.0

    s1 = float(np.sum(values_arr))
    s2 = float(np.sum(values_arr ** 2))

    if s2 == 0.0:
        return 0.0

    return (s1 * s1) / (n * s2)


def evaluate_agent(agent, env, num_episodes=50):
    """
    Run evaluation episodes and return a summary DataFrame of community-level metrics.
    """
    results = {
        "par_base": [],
        "par_opt": [],
        "cost_base": [],
        "cost_opt": [],
        "peak_base": [],
        "peak_opt": [],
        "discomfort": [],
        "jains_index": [],
        "voltage_violations": [],
    }

    for _ in range(int(num_episodes)):
        obs, _ = env.reset()
        done = False

        base_loads = []
        opt_loads = []
        prices_base = []
        prices_opt = []
        actions_sq = []

        episode_voltage_violations = 0
        total_steps = 0

        while not done:
            # Compatible call: uses the alias defined in agents
            actions, _, _ = agent.actions(obs)

            raw = env._get_obs_raw_norm()
            base_load = raw[:, 0].copy()
            base_price = raw[:, 1].copy()

            clipped = np.clip(np.asarray(actions).reshape(-1), -1.0, 1.0)
            env_actions = clipped.reshape(np.asarray(actions).shape)

            opt_load = base_load * (1.0 + clipped)

            total_base = float(np.sum(base_load))
            total_opt = float(np.sum(opt_load))

            price_base_vec = env._get_dynamic_price_real(total_base, base_price)
            price_opt_vec = env._get_dynamic_price_real(total_opt, base_price)

            obs, _, terminated, truncated, info = env.step(env_actions)
            done = bool(terminated or truncated)

            voltages = None if info is None else info.get("voltages")
            if voltages is not None:
                v = np.asarray(voltages, dtype=np.float64)
                violations = np.sum((v < 0.95) | (v > 1.05))
                episode_voltage_violations += int(violations > 0)

            total_steps += 1

            base_loads.append(base_load)
            opt_loads.append(opt_load)
            prices_base.append(price_base_vec)
            prices_opt.append(price_opt_vec)
            actions_sq.append(clipped ** 2)

        base_loads = np.asarray(base_loads, dtype=np.float64)
        opt_loads = np.asarray(opt_loads, dtype=np.float64)
        prices_base = np.asarray(prices_base, dtype=np.float64)
        prices_opt = np.asarray(prices_opt, dtype=np.float64)
        actions_sq = np.asarray(actions_sq, dtype=np.float64)

        comm_base = np.sum(base_loads, axis=1)
        comm_opt = np.sum(opt_loads, axis=1)

        par_base = float(np.max(comm_base) / (np.mean(comm_base) + 1e-6))
        par_opt = float(np.max(comm_opt) / (np.mean(comm_opt) + 1e-6))

        cost_base = float(np.sum(base_loads * prices_base))
        cost_opt = float(np.sum(opt_loads * prices_opt))

        agent_cost_base = np.sum(base_loads * prices_base, axis=0)
        agent_cost_opt = np.sum(opt_loads * prices_opt, axis=0)
        agent_savings = agent_cost_base - agent_cost_opt
        jain = float(calculate_jains_index(agent_savings))

        peak_base = float(np.max(comm_base))
        peak_opt = float(np.max(comm_opt))
        avg_discomfort = float(np.mean(actions_sq))

        results["par_base"].append(par_base)
        results["par_opt"].append(par_opt)
        results["cost_base"].append(cost_base)
        results["cost_opt"].append(cost_opt)
        results["peak_base"].append(peak_base)
        results["peak_opt"].append(peak_opt)
        results["discomfort"].append(avg_discomfort)
        results["jains_index"].append(jain)
        results["voltage_violations"].append(episode_voltage_violations)

    # Return a DataFrame with generic "Value" column
    summary = {
        "Metric": [
            "PAR (Base)", "PAR (Opt)", 
            "Cost (Base)", "Cost (Opt)", 
            "Peak (Base)", "Peak (Opt)", 
            "Discomfort", "Jain's Index", "Voltage Violations"
        ],
        "Value": [
            np.mean(results["par_base"]), np.mean(results["par_opt"]),
            np.mean(results["cost_base"]), np.mean(results["cost_opt"]),
            np.mean(results["peak_base"]), np.mean(results["peak_opt"]),
            np.mean(results["discomfort"]), np.mean(results["jains_index"]),
            np.mean(results["voltage_violations"])
        ]
    }
    return pd.DataFrame(summary)