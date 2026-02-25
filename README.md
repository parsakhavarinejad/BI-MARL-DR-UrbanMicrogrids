# BI-MARL-DR-UrbanMicrogrids

> **Blockchain-Integrated Multi-Agent Reinforcement Learning for Continuous Dynamic Pricing and Occupancy-Aware Demand Response in Urban Microgrids**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

This repository contains the official implementation of the framework proposed in our paper. We introduce a closed-loop intelligent microgrid control architecture integrating behavioral modeling, Multi-Agent Reinforcement Learning (MARL)-based decision-making, and trustless incentive mechanisms to optimize continuous dynamic electricity pricing and occupancy-aware demand response.

---

## 📑 Table of Contents
1. [Introduction](#introduction)
2. [Key Features](#key-features)
3. [Methodology & Architecture](#methodology--architecture)
4. [Supported Algorithms](#supported-algorithms)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Dataset Preparation](#dataset-preparation)
7. [Installation & Usage](#installation--usage)
8. [Repository Structure](#repository-structure)
9. [Citation](#citation)

---

## 📖 Introduction

Modern urban microgrids require highly adaptive demand response (DR) mechanisms to balance grid stability, user comfort, and energy costs. This project implements a **Blockchain-Integrated MARL (BI-MARL)** approach where individual households act as autonomous agents reacting to dynamic continuous pricing. 

Agents learn to shift their loads to off-peak hours based on state vectors that include behavioral features (income band, urban/rural class) and environmental factors (temperature, time of day, historical load).

---

## ✨ Key Features

* **Continuous Dynamic Pricing & Load Control**: Agents output continuous consumption-adjustment actions $a \in [-1, 1]$ directly mapped to load scaling, driven by dynamically calculated community pricing.
* **Occupancy-Aware Demand Response**: Embedded behavioral features (e.g., from the IDEAL dataset) and external variables into local agent state vectors.
* **CTDE MARL Framework**: Centralized Training with Decentralized Execution architectures ensuring global grid stability while maintaining household privacy.
* **Grid-Aware Constraints**: Tracks voltage drop violations to penalize actions that threaten distribution-level network stability.

---

## 🧠 Methodology & Architecture

![Placeholder: System Architecture](<IMAGE_1_PLACEHOLDER_PATH> "Figure 1: BI-MARL System Architecture Overview")
*Figure 1: Overview of the BI-MARL architecture showing the interaction between household agents, the centralized critic, the smart grid environment, and the dynamic pricing mechanism.*

Our framework simulates a continuous 15-minute interval environment (96 steps/day). The pricing function applies a non-linear penalty for aggregate community peak loads, incentivizing agents to collaboratively "shave" the peak without explicit peer-to-peer communication.

---

## 🤖 Supported Algorithms

We benchmark state-of-the-art multi-agent RL algorithms alongside heuristic baselines. 

| Algorithm | Type | Paradigm | Description |
| :--- | :--- | :--- | :--- |
| **MAPPO** | On-Policy | CTDE | Multi-Agent PPO with centralized critic and global state sharing. |
| **IPPO** | On-Policy | Decentralized | Independent PPO where each agent acts and learns independently. |
| **MADDPG** | Off-Policy | CTDE | Multi-Agent Deep Deterministic Policy Gradient. |
| **MATD3** | Off-Policy | CTDE | Twin Delayed DDPG adapted for multi-agent systems. |
| **ToU (Baseline)** | Heuristic | Fixed Rule | Time-of-Use pricing heuristic (peak shedding / off-peak charging). |
| **Rule (Baseline)** | Heuristic | Threshold | Deterministic threshold-based response to real-time pricing. |

---

## 📊 Evaluation Metrics

Agents and community behaviors are evaluated on a strict set of grid and economic metrics. Below is a placeholder for your experimental results table.

| Metric | Description | Model A (e.g. MAPPO) | Baseline (e.g. ToU) |
| :--- | :--- | :---: | :---: |
| **PAR (Peak-to-Average Ratio)** | Grid load stability indicator. | *[Result]* | *[Result]* |
| **Cost (Opt)** | Total monetary cost post-optimization. | *[Result]* | *[Result]* |
| **Peak Load** | Maximum continuous community draw. | *[Result]* | *[Result]* |
| **Discomfort** | Mean squared action penalties (user effort). | *[Result]* | *[Result]* |
| **Jain's Fairness Index** | Distribution of cost savings among agents. | *[Result]* | *[Result]* |
| **Voltage Violations** | Number of episodes with $V < 0.95$ or $V > 1.05$. | *[Result]* | *[Result]* |

---

## 📈 Visualizing Results

![Placeholder: Training Convergence](<IMAGE_2_PLACEHOLDER_PATH> "Figure 2: Reward Convergence across Algorithms")
*Figure 2: Moving average of the total community reward over training episodes comparing MAPPO, MADDPG, and MATD3.*

![Placeholder: Community Peak Shaving](<IMAGE_3_PLACEHOLDER_PATH> "Figure 3: Community Baseline vs Optimized Load")
*Figure 3: Analysis of community peak shaving and resulting dynamic pricing curves during an evaluation episode.*

---

## 💾 Dataset Preparation

The environment requires real-world household consumption data. We utilize the **IDEAL** dataset.

1. Download the raw IDEAL dataset.
2. Run the provided Jupyter Notebook to clean, interploate missing values, append weather data, and extract the best 50 agents with contiguous 96-step (daily) data:
   ```bash
   jupyter notebook data/Shrinking_IDEAL_data.ipynb
   ```

## 🚀 Installation & Usage

1. Setup Environment

```git clone [https://github.com/your-username/BI-MARL-DR-UrbanMicrogrids.git](https://github.com/your-username/BI-MARL-DR-UrbanMicrogrids.git)
cd BI-MARL-DR-UrbanMicrogrids
pip install -r requirements.txt
```

2. Hyperparameter Tuning (Optuna)
To search for optimal actor/critic learning rates, entropy coefficients, and clipping ratios:

```
python src/optuna_tune.py --config configs/config.yaml --trials 30 --train_episodes 20 --eval_episodes 10
```

3. Run the Full Experiment Suite
To train all algorithms sequentially, evaluate them against rule-based baselines, and automatically generate comparison tables and charts (saved to `results/Experiment_Name/`):

```
python run_full_experiment.py
```

4. Run a Specific Algorithm
To manually train a single algorithm (e.g., MAPPO) using the main configuration file:
```
python src/main.py --config configs/config.yaml --algo mappo --exp_name my_custom_run
```

## 📁 Repository Structure
```
BI-MARL-DR-UrbanMicrogrids/
├── configs/
│   └── config.yaml            # Environment, network, and training parameters
├── data/
│   └── Shrinking_IDEAL_data.ipynb # Data preprocessing & feature engineering pipeline
├── src/
│   ├── envs/
│   │   └── smart_grid_env.py  # Custom Gym environment with dynamic pricing
│   ├── marl/
│   │   ├── agents/            # MAPPO, IPPO, MADDPG, MATD3, ToU, Rule-based
│   │   └── networks/          # Actor and Critic standard/Q-networks
│   ├── utils/                 # Data loading, experiment logging, and visualization
│   ├── main.py                # Main training loop script
│   ├── evaluation.py          # Metric calculations (PAR, Jain's, Voltage, etc.)
│   ├── compare_models.py      # Cross-model evaluation script
│   └── optuna_tune.py         # Automated hyperparameter search
├── run_full_experiment.py     # End-to-end execution pipeline
└── README.md
```

## 📝 Citation
If you use this code in your research, please cite our paper:

```
@article{your_paper_2025,
  title={Blockchain-Integrated Multi-Agent Reinforcement Learning for Continuous Dynamic Pricing and Occupancy-Aware Demand Response in Urban Microgrids},
  author={Khavarinejad, Parsa and ...},
  journal={TBD},
  year={2025}
}
````