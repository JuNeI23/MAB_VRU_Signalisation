"""
Upper Confidence Bound (UCB) Multi-Armed Bandit implementation.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
from pathlib import Path
import logging
from .base_mab import BaseMAB

logger = logging.getLogger(__name__)

class UCBMAB(BaseMAB):
    """Upper Confidence Bound MAB implementation."""
    def __init__(self, n_arms: int):
        super().__init__(n_arms)
        # Remove redundant pulls tracking - use inherited counts
        
    def select_arm(self) -> int:
        # Check if any arm hasn't been pulled yet
        if 0 in self.counts:
            # Select first arm that hasn't been pulled yet
            arm = self.counts.index(0)
            return arm
        
        # Calculate UCB values for all arms
        total_pulls = sum(self.counts)
        exploration = np.sqrt(2 * np.log(total_pulls) / np.array(self.counts))
        ucb_values = np.array(self.values) + exploration
        
        selected_arm = int(np.argmax(ucb_values))
        return selected_arm
    
    def _get_ucb_value(self, arm: int) -> float:
        """Calculate UCB value for a specific arm."""
        if self.counts[arm] == 0:
            return float('inf')  # Unplayed arms have infinite priority
        
        total_pulls = sum(self.counts)
        exploration = np.sqrt(2 * np.log(total_pulls) / self.counts[arm])
        return self.values[arm] + exploration
