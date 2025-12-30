import os
import subprocess
from datetime import datetime

def run_command(cmd):
    print(f"Executing: {cmd}")
    os.system(cmd)

def main():
    exp_name = f"Experiment_{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    print(f"Starting Full Experiment Suite: {exp_name}")
    print("=" * 60)

    run_command(f"python src/main.py --algo rule --exp_name {exp_name}")

    run_command(f"python src/main.py --algo tou --exp_name {exp_name}")
    
    run_command(f"python src/main.py --algo ippo --exp_name {exp_name}")
    run_command(f"python src/main.py --algo maddpg --exp_name {exp_name}")
    run_command(f"python src/main.py --algo matd3 --exp_name {exp_name}")
    run_command(f"python src/main.py --algo mappo --exp_name {exp_name}")

    print("\nRunning Comparison & Generating Tables...")
    run_command(f"python src/compare_models.py --exp_name {exp_name}")

    print("=" * 60)
    print(f"Experiment Complete! Results located in: results/{exp_name}")

if __name__ == "__main__":
    main()