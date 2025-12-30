import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_jains_index(values):
    """
    Computes Jain's Fairness Index.
    J = (sum(x)^2) / (n * sum(x^2))
    Range: [1/n, 1] (1 = perfectly fair)
    """
    values = np.array(values)
    # Handle negative savings (if any) by shifting or clipping, 
    # though usually savings should be positive.
    # Here we clip negative savings to 0 for fairness calculation purposes.
    values = np.maximum(values, 0) 
    
    n = len(values)
    if n == 0: return 0.0
    
    numerator = np.sum(values) ** 2
    denominator = n * np.sum(values ** 2)
    
    if denominator == 0:
        return 0.0
        
    return numerator / denominator

def analyze_convergence(log_path, threshold_ratio=0.90, window=10):
    """
    Analyzes training speed.
    Returns the episode number where the Moving Average reward 
    first consistently crossed 90% (or other ratio) of the max achieved MA.
    """
    try:
        df = pd.read_csv(log_path)
        if "moving_avg" not in df.columns:
            return "N/A (Column 'moving_avg' missing)"
            
        max_ma = df["moving_avg"].max()
        target = max_ma * threshold_ratio
        
        # Find first episode where MA >= target
        converged_indices = df.index[df["moving_avg"] >= target].tolist()
        
        if not converged_indices:
            return "Not Converged"
            
        return converged_indices[0] + 1 # Return 1-based episode index
    except Exception as e:
        print(f"Could not calculate convergence: {e}")
        return "Error"

def evaluate_agent(agent, env, num_episodes=50):
    print(f"Starting Evaluation over {num_episodes} episodes...")

    results = {
        "par_base": [],
        "par_opt": [],
        "cost_base": [],
        "cost_opt": [],
        "peak_base": [],
        "peak_opt": [],
        "discomfort": [],
        "jains_index": [],     # NEW: Fairness
        "voltage_violations": [] # NEW: Grid Reliability
    }

    # Store per-agent savings for global fairness analysis
    all_agent_savings = []

    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False

        base_loads = []
        opt_loads = []
        prices_base = []
        prices_opt = []
        actions_sq = []
        
        # Track voltage violations for this episode
        episode_voltage_violations = 0
        total_steps = 0

        while not done:
            # 1) Agent proposes actions
            actions, _, _ = agent.actions(obs)

            # 2) Read raw baseline signals for THIS step
            raw = env._get_obs_raw_norm()
            current_base_load = raw[:, 0].copy()
            current_base_price = raw[:, 1].copy()

            # 3) Clip actions
            clipped_flat = np.clip(np.asarray(actions).reshape(-1), -1.0, 1.0)
            env_actions = clipped_flat.reshape(np.asarray(actions).shape)

            # 4) Compute loads
            actual_load = current_base_load * (1.0 + clipped_flat)

            # 5) Compute Prices
            total_grid_load_base = float(np.sum(current_base_load))
            total_grid_load_opt = float(np.sum(actual_load))

            price_base_vec = env._get_dynamic_price_real(total_grid_load_base, current_base_price)
            price_opt_vec = env._get_dynamic_price_real(total_grid_load_opt, current_base_price)

            # 6) Step environment
            obs, _, terminated, truncated, info = env.step(env_actions)
            done = bool(terminated or truncated)

            # --- NEW: Check Voltage Violations ---
            # NOTE: Your env.step needs to return voltage info. 
            # If info is empty, this defaults to 0.
            if "voltages" in info:
                v = np.array(info["voltages"])
                # IEEE Standard 1547: Violation if V < 0.95 or V > 1.05 p.u.
                violations = np.sum((v < 0.95) | (v > 1.05))
                episode_voltage_violations += (violations > 0) # Binary flag per step
            
            total_steps += 1

            # 7) Store trajectories
            base_loads.append(current_base_load)
            opt_loads.append(actual_load)
            prices_base.append(price_base_vec)
            prices_opt.append(price_opt_vec)
            actions_sq.append(clipped_flat ** 2)

        # Convert to arrays
        base_loads = np.asarray(base_loads)
        opt_loads = np.asarray(opt_loads)
        prices_base = np.asarray(prices_base)
        prices_opt = np.asarray(prices_opt)
        actions_sq = np.asarray(actions_sq)

        # Community load
        comm_base = np.sum(base_loads, axis=1)
        comm_opt = np.sum(opt_loads, axis=1)

        # PAR
        par_b = np.max(comm_base) / (np.mean(comm_base) + 1e-6)
        par_o = np.max(comm_opt) / (np.mean(comm_opt) + 1e-6)

        # Costs (Total Community)
        cost_b = np.sum(base_loads * prices_base)
        cost_o = np.sum(opt_loads * prices_opt)
        
        # --- NEW: Fairness (Jain's Index of Savings) ---
        # Calculate cost per agent
        agent_cost_base = np.sum(base_loads * prices_base, axis=0)
        agent_cost_opt = np.sum(opt_loads * prices_opt, axis=0)
        
        # Savings per agent
        agent_savings = agent_cost_base - agent_cost_opt
        jain = calculate_jains_index(agent_savings)
        all_agent_savings.extend(agent_savings)

        # Peak
        peak_b = np.max(comm_base)
        peak_o = np.max(comm_opt)

        disc_score = np.mean(actions_sq)
        
        # Voltage Violation Rate (Percentage of steps with issues)
        volt_rate = (episode_voltage_violations / total_steps) * 100 if total_steps > 0 else 0.0

        results["par_base"].append(par_b)
        results["par_opt"].append(par_o)
        results["cost_base"].append(cost_b)
        results["cost_opt"].append(cost_o)
        results["peak_base"].append(peak_b)
        results["peak_opt"].append(peak_o)
        results["discomfort"].append(disc_score)
        results["jains_index"].append(jain)
        results["voltage_violations"].append(volt_rate)

    df = pd.DataFrame(results)

    summary = pd.DataFrame(
        {
            "Metric": ["PAR", "Total Cost", "Peak Load", "Discomfort", "Fairness (Jain)", "Voltage Viol. (%)"],
            "Baseline (Mean)": [
                df["par_base"].mean(),
                df["cost_base"].mean(),
                df["peak_base"].mean(),
                0.0,
                0.0, # Baseline fairness is usually N/A or 0 as there is no savings
                0.0  # Placeholder, usually baseline has violations too if grid is stressed
            ],
            "MAPPO (Mean)": [
                df["par_opt"].mean(),
                df["cost_opt"].mean(),
                df["peak_opt"].mean(),
                df["discomfort"].mean(),
                df["jains_index"].mean(),
                df["voltage_violations"].mean()
            ],
            "Improvement (%)": [
                (df["par_base"].mean() - df["par_opt"].mean()) / (df["par_base"].mean() + 1e-12) * 100,
                (df["cost_base"].mean() - df["cost_opt"].mean()) / (df["cost_base"].mean() + 1e-12) * 100,
                (df["peak_base"].mean() - df["peak_opt"].mean()) / (df["peak_base"].mean() + 1e-12) * 100,
                np.nan,
                np.nan,
                np.nan
            ],
        }
    )

    return summary