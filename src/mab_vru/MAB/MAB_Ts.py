"""
Thompson Sampling Multi-Armed Bandit implementation.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
from pathlib import Path
import logging
from .base_mab import BaseMAB

logger = logging.getLogger(__name__)

class ThompsonSamplingMAB(BaseMAB):
    """Thompson Sampling MAB implementation."""
    def __init__(self, n_arms: int):
        super().__init__(n_arms)
        self.alpha = np.ones(n_arms)  # Success count + 1
        self.beta = np.ones(n_arms)   # Failure count + 1
        self.means = np.zeros(n_arms)
        self.stds = np.ones(n_arms)
        
    def select_arm(self) -> int:
        samples = np.random.beta(self.alpha, self.beta)
        return int(np.argmax(samples))
    
    def update(self, arm: int, reward: float) -> None:
        """Update arm statistics with new reward."""
        super().update(arm, reward)
        # Treat reward as binary success/failure probability
        success_prob = min(max(reward, 0), 1)  # Clip to [0,1]
        self.alpha[arm] += success_prob
        self.beta[arm] += (1 - success_prob)
        # Update mean and standard deviation for arm
        n = self.alpha[arm] + self.beta[arm] - 2  # Subtract 2 for initial alpha=beta=1
        self.means[arm] = self.alpha[arm] / (self.alpha[arm] + self.beta[arm])
        if n > 0:
            self.stds[arm] = np.sqrt(
                (self.alpha[arm] * self.beta[arm]) /
                ((self.alpha[arm] + self.beta[arm])**2 * (self.alpha[arm] + self.beta[arm] + 1))
            )


