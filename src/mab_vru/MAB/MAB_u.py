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
        self.total_pulls = 0
        self.pulls = np.zeros(n_arms)  # Track number of pulls per arm
        
    def select_arm(self) -> int:
        if 0 in self.pulls:
            # Select first arm that hasn't been pulled yet
            arm = int(np.where(self.pulls == 0)[0][0])
            self.pulls[arm] += 1
            self.total_pulls += 1
            return arm
        
        self.total_pulls += 1
        exploration = np.sqrt(2 * np.log(self.total_pulls) / self.pulls)
        ucb = self.values + exploration
        selected_arm = int(np.argmax(ucb))
        self.pulls[selected_arm] += 1
        return selected_arm
