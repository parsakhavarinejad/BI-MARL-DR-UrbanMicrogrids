import numpy as np

class TOUAgent:
    """
    Time-of-Use (ToU) Heuristic Agent.
    
    Strategy:
    - Peak Hours (16:00 - 20:00): Shed load (-1.0) to save money/grid.
    - Off-Peak (00:00 - 05:00): Increase load (+1.0) to pre-charge.
    - Otherwise: Normal operation (0.0).
    """
    
    def __init__(self):
        # Peak/Off-peak hours can be adjusted here
        self.peak_start = 16.0
        self.peak_end = 20.0
        
        self.off_peak_start = 0.0
        self.off_peak_end = 5.0

    def act(self, state):
        """
        Decodes time from state and applies ToU logic.
        State indices: 4=tod_sin, 5=tod_cos
        """
        # 1. Decode Time from Sin/Cos features
        sin_vals = state[:, 4]
        cos_vals = state[:, 5]
        
        # arctan2 returns [-pi, pi], we map to [0, 2pi]
        angles = np.arctan2(sin_vals, cos_vals)
        angles[angles < 0] += 2 * np.pi
        
        # Map [0, 2pi] -> [0, 24 hours]
        time_fractions = angles / (2 * np.pi)
        hours = time_fractions * 24.0

        # 2. Apply Rules
        actions = np.zeros((state.shape[0], 1), dtype=np.float32)
        
        # Peak: Shed Load
        peak_mask = (hours >= self.peak_start) & (hours < self.peak_end)
        actions[peak_mask] = -1.0
        
        # Off-Peak: Boost Load
        off_peak_mask = (hours >= self.off_peak_start) & (hours < self.off_peak_end)
        actions[off_peak_mask] = 1.0

        return actions, None, None

    # Alias for compatibility
    def actions(self, state):
        return self.act(state)

    # Dummy methods for pipeline compatibility
    def store(self, *args): pass
    def save(self, *args): pass
    def load(self, *args): pass
    def update(self): pass