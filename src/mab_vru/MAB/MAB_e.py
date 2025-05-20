"""
Epsilon-Greedy Multi-Armed Bandit implementation.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List, Union, Optional
from pathlib import Path
import logging
from .base_mab import BaseMAB

logger = logging.getLogger(__name__)

class EpsilonGreedyMAB(BaseMAB):
    """
    ε-greedy MAB implementation.
    """
    def __init__(self, n_arms: int, epsilon: float = 0.1):
        super().__init__(n_arms)
        self.epsilon = epsilon
        
    def select_arm(self) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_arms)
        return int(np.argmax(self.values))

def run_evolution(df: pd.DataFrame, epsilon: float) -> Tuple[List[float], np.ndarray]:
    """Run evolution on simulation data."""
    times = sorted(df['Time'].unique())
    n_times = len(times)
    mab = EpsilonGreedyMAB(n_arms=2, epsilon=epsilon)
    history = np.zeros((n_times, 2))
    
    for idx, t in enumerate(times):
        try:
            # Get data for current timestep with error handling
            v2v_data = df[(df['Time'] == t) & (df['Protocol'] == 'V2V')].iloc[0]
            v2i_data = df[(df['Time'] == t) & (df['Protocol'] == 'V2I')].iloc[0]
        except IndexError:
            # If we're missing data for one protocol at this timestamp,
            # use the worst possible values for the missing protocol
            worst_case = pd.Series({
                'Average Delay (s)': df['Average Delay (s)'].max(),
                'Loss Rate (%)': 100.0,
                'Average Load': df['Average Load'].max()
            })
            
            # Check which protocol is missing
            v2v_exists = len(df[(df['Time'] == t) & (df['Protocol'] == 'V2V')]) > 0
            v2i_exists = len(df[(df['Time'] == t) & (df['Protocol'] == 'V2I')]) > 0
            
            if not v2v_exists:
                v2v_data = worst_case
                v2i_data = df[(df['Time'] == t) & (df['Protocol'] == 'V2I')].iloc[0]
            else:
                v2v_data = df[(df['Time'] == t) & (df['Protocol'] == 'V2V')].iloc[0]
                v2i_data = worst_case
        
        # Safely get metric values with inf/nan handling
        def safe_get_metric(data, metric):
            val = data[metric]
            if np.isinf(val) or np.isnan(val):
                return 1.0  # Penalize inf/nan values
            return val
            
        # Get metrics with safety checks
        v2v_delay = safe_get_metric(v2v_data, 'Average Delay (s)')
        v2v_loss = safe_get_metric(v2v_data, 'Loss Rate (%)')
        v2v_load = safe_get_metric(v2v_data, 'Average Load')
        
        v2i_delay = safe_get_metric(v2i_data, 'Average Delay (s)')
        v2i_loss = safe_get_metric(v2i_data, 'Loss Rate (%)')
        v2i_load = safe_get_metric(v2i_data, 'Average Load')
        
        # Normalize metrics to [0,1] range (lower is better)
        max_delay = max(v2v_delay, v2i_delay)
        max_loss = max(v2v_loss, v2i_loss)
        max_load = max(v2v_load, v2i_load)
        
        # All metrics weighted equally (1/3 each)
        v2v_score = (
            (1/3) * (v2v_delay / max_delay if max_delay > 0 else 0.0) +
            (1/3) * (v2v_loss / max_loss if max_loss > 0 else 0.0) +
            (1/3) * (v2v_load / max_load if max_load > 0 else 0.0)
        )
        
        v2i_score = (
            (1/3) * (v2i_delay / max_delay if max_delay > 0 else 0.0) +
            (1/3) * (v2i_loss / max_loss if max_loss > 0 else 0.0) +
            (1/3) * (v2i_load / max_load if max_load > 0 else 0.0)
        )
        
        # Convert to rewards (higher is better)
        v2v_reward = 1.0 - v2v_score
        v2i_reward = 1.0 - v2i_score
        
        # Select and update
        arm = mab.select_arm()
        if arm == 0:
            mab.update(0, v2v_reward)
        else:
            mab.update(1, v2i_reward)
            
        history[idx] = mab.values
        
    return times, history

