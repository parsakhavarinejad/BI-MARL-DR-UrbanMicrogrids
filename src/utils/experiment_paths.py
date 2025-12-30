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
            # Default to timestamp if not provided
            experiment_name = f"Run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.exp_root = os.path.join(base_dir, experiment_name)
        
        # 1. Models
        self.models_dir = os.path.join(self.exp_root, "models")
        # 2. Training Results (CSV + Curves)
        self.training_dir = os.path.join(self.exp_root, "training_results")
        # 3. Inference Images
        self.inference_dir = os.path.join(self.exp_root, "inference_images")
        # 4. Comparison Results
        self.comparison_dir = os.path.join(self.exp_root, "comparison_results")
        
        # Create the top-level folders immediately
        for d in [self.models_dir, self.training_dir, self.inference_dir, self.comparison_dir]:
            os.makedirs(d, exist_ok=True)
            
    def get_agent_dirs(self, algo_name):
        """
        Creates and returns the specific subfolders for a given algorithm.
        """
        paths = {
            # 1. models/algo_name/
            "model": os.path.join(self.models_dir, algo_name),
            
            # 2. training_results/algo_name/
            "training": os.path.join(self.training_dir, algo_name),
            
            # 3. inference_images/algo_name/
            "inference": os.path.join(self.inference_dir, algo_name),
        }
        
        # Ensure these specific subfolders exist
        for p in paths.values():
            os.makedirs(p, exist_ok=True)
            
        return paths

    def get_comparison_dirs(self):
        """
        Returns the path for comparison results.
        """
        # You can add subfolders like 'tables' or 'charts' if desired, 
        # but your request asked for a "comparison results" folder directly.
        return self.comparison_dir