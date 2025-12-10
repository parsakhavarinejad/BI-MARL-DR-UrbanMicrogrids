# BI-MARL-DR-UrbanMicrogrids
Blockchain-Integrated Multi-Agent Reinforcement Learning for Continuous Dynamic Pricing and Occupancy-Aware Demand Response in Urban Microgrids

This repository implements a full research and development framework for a Blockchain-Integrated Multi-Agent Reinforcement Learning (BI-MARL) architecture designed to enable:

- Continuous dynamic electricity pricing  
- Occupancy-aware, behavior-driven demand response  
- Blockchain-verified incentive mechanisms  
- Grid-aware optimization using IEEE 33-bus and 69-bus networks  

The system integrates behavioral modeling, MARL-based decision-making, and trustless blockchain reward verification into a unified closed-loop intelligent microgrid control architecture.

## Key Features

### Occupancy-Aware Demand Response
- Uses datasets such as IDEAL and UK-DALE.
- Embeds occupancy, property type, and behavioral features into agent state vectors.
- Supports residential, commercial, and industrial load modeling.

### Continuous Dynamic Pricing (MARL)
- Implements MI-TRPO, TPMIL, MAPPO baselines.
- Agents output continuous consumption-adjustment actions.
- Trust-region updates guarantee monotonic improvement and stable convergence.

### Blockchain-Verified Reward Signals
- Smart contracts:
  - DemandResponseContract  
  - RewardContract  
  - AuditContract  
- Verifies DR participation, reduction, and compliance.
- Provides tamper-proof on-chain learning signals for MARL reward shaping.

### IEEE Bus Grid Simulation
- Supports IEEE 33-bus and 69-bus distribution networks.
- Includes voltage, congestion, and line constraint modeling.
- Realistic power-flow coupled with agent decisions.

### Modular Framework
- Swap MARL algorithms, blockchain backends, datasets, or pricing models.
- Clear separation of environment, agents, blockchain, and utilities.

## Repository Structure

```
BI-MARL-DR-UrbanMicrogrids/
├── README.md
├── configs/
├── data/
│   ├── IDEAL/
│   ├── UK-DALE/
│   ├── IEEE_Bus/
│   └── processed/
├── notebooks/
├── src/
│   ├── envs/
│   ├── marl/
│   ├── blockchain/
│   └── utils/
├── experiments/
├── models/
├── results/
└── docs/
```

## Getting Started

### Clone the repository
```bash
git clone https://github.com/<your-username>/BI-MARL-DR-UrbanMicrogrids.git
cd BI-MARL-DR-UrbanMicrogrids
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Train MARL agents
```bash
python src/main.py --config configs/marl_config.yaml
```

## License
MIT License
