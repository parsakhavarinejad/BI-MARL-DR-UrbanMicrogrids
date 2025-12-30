import os
from datetime import datetime

class ExperimentPaths:
    def __init__(self, experiment_name=None, base_dir="results"):
        """
        Manages the file structure for a specific experiment run.
        Structure:
          results/
            Experiment_Name/
              models/
                mappo/ ...
              training_results/
                mappo/ ... (csv + training images)
              inference_images/
                mappo/ ...
              comparison_results/
                (tables and charts)
        """
        if experiment_name is None:
            experiment_name = f"Run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.exp_root = os.path.join(base_dir, experiment_name)
        
        self.models_dir = os.path.join(self.exp_root, "models")
        self.training_dir = os.path.join(self.exp_root, "training_results")
        self.inference_dir = os.path.join(self.exp_root, "inference_images")
        self.comparison_dir = os.path.join(self.exp_root, "comparison_results")
        
        for d in [self.models_dir, self.training_dir, self.inference_dir, self.comparison_dir]:
            os.makedirs(d, exist_ok=True)
            
    def get_agent_dirs(self, algo_name):
        """
        Creates and returns the specific subfolders for a given algorithm.
        """
        paths = {
            "model": os.path.join(self.models_dir, algo_name),
            "training": os.path.join(self.training_dir, algo_name),
            "inference": os.path.join(self.inference_dir, algo_name),
        }
        
        for p in paths.values():
            os.makedirs(p, exist_ok=True)
            
        return paths

    def get_comparison_dirs(self):
        """
        Returns the path for comparison results.
        """
        return self.comparison_dir