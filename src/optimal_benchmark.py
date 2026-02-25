import os
import numpy as np
import cvxpy as cp

from utils.data_loader import SmartGridDataLoader
from envs.smart_grid_env import SmartGridEnv
from utils.config_parser import Config


# ==============================
# LOAD CONFIG
# ==============================

cfg = Config("configs/config.yaml")
cfg.data_path = cfg.data_path.replace("\\", "/")

print("Using dataset:", cfg.data_path)

# ==============================
# BUILD ENV
# ==============================

data_loader = SmartGridDataLoader(cfg.data_path, cfg.num_agents, verbose=0)

env = SmartGridEnv(
    data_loader,
    cfg.env_ratio_clip_min_max,
    cfg.env_total_price_clip_min_max,
    cfg.actor_state_dim,
    cfg.env_scaling_factor,
    cfg.env_discomfort_weight,
    cfg.num_agents,
    cfg.num_steps_per_day
)

obs, _ = env.reset()

Pbase = []
price = []

for _ in range(cfg.num_steps_per_day):
    raw = env._get_obs_raw_norm()

    Pbase.append(raw[:, 0].copy())
    price.append(raw[:, 1].copy())

    obs, _, terminated, truncated, _ = env.step(
        np.zeros((cfg.num_agents, 1))
    )
    if terminated or truncated:
        break

Pbase = np.array(Pbase).T
price = np.array(price)[:, 0]

N, T = Pbase.shape

print("Extracted Pbase shape:", Pbase.shape)


# ==============================
# OPTIMAL PARAMETERS
# ==============================

lambda_c = 1.0
lambda_p = 1.5
lambda_d = cfg.env_discomfort_weight

a_min = -1.0
a_max = 1.0

seeds = [0, 42, 123, 999, 2024]

cost_list = []
peak_list = []
par_list = []
discomfort_list = []

# ==============================
# RUN 5 SEEDS
# ==============================

for seed in seeds:

    np.random.seed(seed)

    a = cp.Variable((N, T))
    z = cp.Variable((N, T), nonneg=True)

    Pload = cp.multiply(Pbase, (1 + a))
    Ptot = cp.sum(Pload, axis=0)

    cost_term = cp.sum(cp.multiply(price, Ptot)) * lambda_c
    peak_term = lambda_p * cp.sum_squares(Ptot)
    discomfort_term = lambda_d * cp.sum(z)

    objective = cp.Minimize(cost_term + peak_term + discomfort_term)

    constraints = [
        a >= a_min,
        a <= a_max,
        z >= cp.multiply(Pbase, a),
        z >= -cp.multiply(Pbase, a)
    ]

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.OSQP)

    a_opt = a.value
    Pload_opt = Pbase * (1 + a_opt)

    comm = np.sum(Pload_opt, axis=0)

    total_cost = np.sum(price * comm)
    peak = np.max(comm)
    par = peak / (np.mean(comm) + 1e-6)
    avg_discomfort = np.mean(np.abs(Pbase * a_opt))

    cost_list.append(total_cost)
    peak_list.append(peak)
    par_list.append(par)
    discomfort_list.append(avg_discomfort)

# ==============================
# REPORT
# ==============================

def report(name, arr):
    print(f"{name}: {np.mean(arr):.6f} ± {np.std(arr):.6f}")

print("\n===== OPTIMAL (5 SEEDS) =====")
report("Total Cost", cost_list)
report("Peak Load", peak_list)
report("PAR", par_list)
report("Avg Discomfort", discomfort_list)
print("=============================\n")