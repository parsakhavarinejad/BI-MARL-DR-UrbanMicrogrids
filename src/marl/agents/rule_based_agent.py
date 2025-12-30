import numpy as np

class RuleBasedAgent:
    def __init__(self, high_price_threshold=0.7, low_price_threshold=0.3):
        self.high_thresh = high_price_threshold
        self.low_thresh = low_price_threshold

    def actions(self, state):
        """
        State is (num_agents, num_features).
        Assumes feature 1 is 'price_norm' (normalized 0-1).
        """
        # Extract price (assuming 2nd feature at index 1 is price)
        prices = state[:, 1]
        
        actions = []
        for p in prices:
            if p > self.high_thresh:
                actions.append(-1.0) # Maximum reduction
            elif p < self.low_thresh:
                actions.append(1.0)  # Maximum consumption (shift load here)
            else:
                actions.append(0.0)  # Do nothing
        
        return np.array(actions).reshape(-1, 1), None, None

    def save(self, path):
        pass # Nothing to save
    
    def load(self, path):
        pass