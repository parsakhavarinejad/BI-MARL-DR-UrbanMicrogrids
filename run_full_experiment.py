import os
import subprocess
from datetime import datetime

def run_command(cmd):
    print(f"Executing: {cmd}")
    os.system(cmd)

def main():
    # Create one unique ID for this whole run
    exp_name = f"Experiment_{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    print(f"Starting Full Experiment Suite: {exp_name}")
    print("=" * 60)

    # 1. Run All Agents (Use --exp_name to keep them in the same folder)
    # The script will now save CSVs to: results/Experiment_.../training_results/{algo}/
    
    # Rule Based
    run_command(f"python src/main.py --algo rule --exp_name {exp_name}")
    
    # IPPO
    run_command(f"python src/main.py --algo ippo --exp_name {exp_name}")
    
    # MADDPG
    run_command(f"python src/main.py --algo maddpg --exp_name {exp_name}")
    
    # MAPPO
    run_command(f"python src/main.py --algo mappo --exp_name {exp_name}")

    # 2. Run Comparison
    # Saves to: results/Experiment_.../comparison_results/
    print("\nRunning Comparison & Generating Tables...")
    run_command(f"python src/compare_models.py --exp_name {exp_name}")

    print("=" * 60)
    print(f"Experiment Complete! Results located in: results/{exp_name}")

if __name__ == "__main__":
    main()