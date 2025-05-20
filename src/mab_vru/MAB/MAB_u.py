"""
Upper Confidence Bound (UCB) Multi-Armed Bandit implementation.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List, Union, Optional
from pathlib import Path
import logging
from .base_mab import BaseMAB

logger = logging.getLogger(__name__)

class UCBMAB(BaseMAB):
    """UCB1 MAB implementation with improved confidence bounds."""
    
    def __init__(self, n_arms: int):
        super().__init__(n_arms)
        self.total_counts = 0
        
    def _get_ucb_value(self, arm: int) -> float:
        """
        Calculate the UCB value for a given arm.
        
        Args:
            arm: Index of the arm
            
        Returns:
            UCB value for the arm
        """
        if self.counts[arm] == 0:
            return float('inf')
        if self.total_counts == 0:
            return self.values[arm]  # Return current value if no exploration term
        return self.values[arm] + np.sqrt(2 * np.log(self.total_counts) / self.counts[arm])
        
    def select_arm(self) -> int:
        """Select the arm with highest UCB value."""
        ucb_values = [self._get_ucb_value(arm) for arm in range(self.n_arms)]
        return int(np.argmax(ucb_values))
        
    def update(self, chosen_arm: int, reward: float) -> None:
        """Update statistics for the chosen arm."""
        super().update(chosen_arm, reward)
        self.total_counts += 1

def run_evolution(df: pd.DataFrame, n_arms: int = 2) -> Tuple[List[float], np.ndarray]:
    """Run UCB evolution on simulation data."""
    times = sorted(df['Time'].unique())
    n_times = len(times)
    mab = UCBMAB(n_arms)
    history = np.zeros((n_times, n_arms))
    
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

def plot_evolution(times, history, save_path=None):
    """Plot the evolution of both arms over time."""
    plt.figure(figsize=(10, 6))
    plt.plot(times, history[:, 0], label='V2V', color='blue')
    plt.plot(times, history[:, 1], label='V2I', color='red')
    plt.xlabel('Time')
    plt.ylabel('UCB Value')
    plt.title('UCB: Protocol Performance Evolution')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def compare_protocols(df: pd.DataFrame, save_path: Optional[Path] = None) -> Tuple[str, float]:
    """Compare V2V and V2I protocols using UCB strategy."""
    logger.info("Starting UCB comparison")
    
    times, history = run_evolution(df)
    
    if save_path:
        plot_evolution(times, history, save_path)
    
    mab = UCBMAB(n_arms=2)
    
    # Process data chronologically
    for t in sorted(df['Time'].unique()):
        v2v_data = df[(df['Time'] == t) & (df['Protocol'] == 'V2V')].iloc[0]
        v2i_data = df[(df['Time'] == t) & (df['Protocol'] == 'V2I')].iloc[0]
        
        def safe_get_metric(data, metric):
            val = data[metric]
            if np.isinf(val) or np.isnan(val):
                return 1.0
            return val
            
        # Get metrics with safety checks
        v2v_delay = safe_get_metric(v2v_data, 'Average Delay (s)')
        v2v_loss = safe_get_metric(v2v_data, 'Loss Rate (%)')
        v2v_load = safe_get_metric(v2v_data, 'Average Load')
        
        v2i_delay = safe_get_metric(v2i_data, 'Average Delay (s)')
        v2i_loss = safe_get_metric(v2i_data, 'Loss Rate (%)')
        v2i_load = safe_get_metric(v2i_data, 'Average Load')
        
        # UCB emphasizes minimizing packet loss more
        max_delay = max(v2v_delay, v2i_delay)
        max_loss = max(v2v_loss, v2i_loss)
        max_load = max(v2v_load, v2i_load)
        
        # Different weights: 20% delay, 60% loss, 20% load
        v2v_score = (
            0.2 * (v2v_delay / max_delay if max_delay > 0 else 0.0) +
            0.6 * (v2v_loss / max_loss if max_loss > 0 else 0.0) +
            0.2 * (v2v_load / max_load if max_load > 0 else 0.0)
        )
        
        v2i_score = (
            0.2 * (v2i_delay / max_delay if max_delay > 0 else 0.0) +
            0.6 * (v2i_loss / max_loss if max_loss > 0 else 0.0) +
            0.2 * (v2i_load / max_load if max_load > 0 else 0.0)
        )
        
        # Convert to rewards (higher is better)
        v2v_reward = 1.0 - v2v_score
        v2i_reward = 1.0 - v2i_score
        
        # Update MAB with both rewards
        mab.update(0, v2v_reward)
        mab.update(1, v2i_reward)
    
    # Determine best protocol based on final values
    v2v_value = mab.values[0] 
    v2i_value = mab.values[1]
    best_protocol = 'V2V' if v2v_value > v2i_value else 'V2I'
    best_value = max(v2v_value, v2i_value)
    
    logger.info(f"Final values after {len(df['Time'].unique())} iterations:")
    logger.info(f"V2V: {v2v_value:.3f}")
    logger.info(f"V2I: {v2i_value:.3f}")
    logger.info(f"Best protocol: {best_protocol}")
    
    return best_protocol, best_value
