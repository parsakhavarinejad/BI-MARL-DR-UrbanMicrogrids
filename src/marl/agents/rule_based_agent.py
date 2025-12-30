import numpy as np


class RuleBasedAgent:
    """
    Deterministic rule-based agent that selects actions based on thresholded price signals.
    """

    def __init__(self, high_price_threshold=0.7, low_price_threshold=0.3):
        self.high_threshold = high_price_threshold
        self.low_threshold = low_price_threshold

    def actions(self, state):
        """
        Compute actions for each agent based on normalized price values.
        """
        prices = state[:, 1]

        actions = np.zeros(len(prices), dtype=np.float32)

        actions[prices > self.high_threshold] = -1.0
        actions[prices < self.low_threshold] = 1.0

        return actions.reshape(-1, 1), None, None
    
    # Alias
    def act(self, state):
        return self.actions(state)

    def save(self, path):
        return None

    def load(self, path):
        return None