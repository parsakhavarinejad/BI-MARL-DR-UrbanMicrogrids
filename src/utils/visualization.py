import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os


def save_paper_visualizations(agent, env, save_dir="results"):
    """
    Generates high-quality plots for research papers:
    1. Training Learning Curve
    2. Single Agent: Load vs. Price vs. Action
    3. Community: Aggregate Peak Shaving Analysis
    """
    os.makedirs(save_dir, exist_ok=True)
    sns.set(style="whitegrid", font_scale=1.1)

    csv_path = os.path.join(save_dir, "training_data.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)

        plt.figure(figsize=(10, 6))
        plt.plot(
            df.index, df["episode_reward"], alpha=0.2, color="gray", label="Raw Reward"
        )
        plt.plot(
            df.index,
            df["moving_avg"],
            color="#d62728",
            linewidth=2.5,
            label="Moving Avg (10)",
        )

        plt.xlabel("Episodes", fontsize=14)
        plt.ylabel("Total Community Reward", fontsize=14)
        plt.title("MAPPO Training Convergence", fontsize=16, fontweight="bold")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "paper_fig_1_convergence.png"), dpi=300)
        plt.close()
        print(f"[Plot] Saved Convergence Plot to {save_dir}")

    print("[Plot] Running evaluation episode for detailed profiles...")
    obs, _ = env.reset()

    history = {"base_load": [], "actual_load": [], "price": [], "actions": []}

    # ... inside save_paper_visualizations ...

    steps = env.total_steps
    for _ in range(steps):
        # Get actions
        actions, _, _ = agent.actions(obs)

        # Get Raw State
        raw_state = env._get_obs_raw()
        current_base_load = raw_state[:, 0].copy()
        current_base_price = raw_state[:, 1].copy()  # <--- This is the FLAT Base Price

        # Calculate Actual Load
        clipped_actions = np.clip(actions.reshape(-1), -1.0, 1.0)
        actual_load = current_base_load * (1 + clipped_actions)

        # --- NEW: CALCULATE DYNAMIC PRICE FOR PLOTTING ---
        total_grid_load = np.sum(actual_load)
        # We manually call the env's pricing function to get the REAL price
        dynamic_price = env._get_dynamic_price(total_grid_load, current_base_price)
        # -------------------------------------------------

        obs, rewards, done, truncated, _ = env.step(actions)

        history["base_load"].append(current_base_load)
        history["actual_load"].append(actual_load)

        # CHANGE THIS LINE: Store dynamic_price instead of current_base_price
        history["price"].append(dynamic_price)

        history["actions"].append(clipped_actions)

    base_load = np.array(history["base_load"])
    actual_load = np.array(history["actual_load"])
    price = np.array(history["price"])
    actions = np.array(history["actions"])
    time_steps = np.arange(steps) / 4.0

    for agent_id in range(12):
        fig, ax1 = plt.subplots(figsize=(12, 6))

        ax1.plot(
            time_steps,
            base_load[:, agent_id],
            "--",
            color="gray",
            label="Baseline Load (No AI)",
            linewidth=2,
        )
        ax1.plot(
            time_steps,
            actual_load[:, agent_id],
            "-",
            color="#1f77b4",
            label="Optimized Load (MAPPO)",
            linewidth=2.5,
        )
        ax1.set_xlabel("Time of Day (Hour)", fontsize=14)
        ax1.set_ylabel("Power Consumption (kWh)", fontsize=14, color="#1f77b4")
        ax1.tick_params(axis="y", labelcolor="#1f77b4")
        ax1.grid(False)

        ax2 = ax1.twinx()
        ax2.plot(
            time_steps,
            price[:, agent_id],
            ":",
            color="#ff7f0e",
            label="Electricity Price",
            linewidth=2,
        )
        ax2.fill_between(time_steps, price[:, agent_id], alpha=0.1, color="#ff7f0e")
        ax2.set_ylabel("Price (Pence/kWh)", fontsize=14, color="#ff7f0e")
        ax2.tick_params(axis="y", labelcolor="#ff7f0e")
        ax2.grid(False)

        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

        plt.title(
            f"Single Home Optimization (Agent {agent_id})",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(save_dir, f"paper_fig_2_single_agent_{agent_id}.png"), dpi=300
        )
        plt.close()
        print(f"[Plot] Saved Single Agent Plot to {save_dir}")

    total_base = np.sum(base_load, axis=1)
    total_opt = np.sum(actual_load, axis=1)

    par_base = np.max(total_base) / np.mean(total_base)
    par_opt = np.max(total_opt) / np.mean(total_opt)

    plt.figure(figsize=(12, 6))
    plt.plot(
        time_steps,
        total_base,
        "--",
        color="black",
        alpha=0.6,
        label=f"Community Baseline (PAR: {par_base:.2f})",
    )
    plt.plot(
        time_steps,
        total_opt,
        "-",
        color="#2ca02c",
        linewidth=2.5,
        label=f"Community Optimized (PAR: {par_opt:.2f})",
    )

    plt.fill_between(
        time_steps,
        total_base,
        total_opt,
        where=(total_base > total_opt),
        interpolate=True,
        color="green",
        alpha=0.2,
        label="Peak Shaving (Saved Energy)",
    )

    plt.xlabel("Time of Day (Hour)", fontsize=14)
    plt.ylabel("Total Community Load (kWh)", fontsize=14)
    plt.title("Community Peak Shaving Analysis", fontsize=16, fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "paper_fig_3_community_peak_shaving.png"), dpi=300
    )
    plt.close()
    print(f"[Plot] Saved Community Plot to {save_dir}")
