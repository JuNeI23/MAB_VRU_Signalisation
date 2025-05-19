"""
Thompson Sampling Multi-Armed Bandit implementation.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List, Union, Optional
from pathlib import Path
import logging
from .base_mab import BaseMAB

logger = logging.getLogger(__name__)

class GaussianThompsonSampling(BaseMAB):
    """Thompson Sampling MAB with Gaussian rewards."""
    
    def __init__(self, n_arms: int):
        super().__init__(n_arms)
        self.means = np.zeros(n_arms)
        self.stds = np.ones(n_arms)
        
    def select_arm(self) -> int:
        return int(np.argmax([
            np.random.normal(self.means[i], self.stds[i])
            for i in range(self.n_arms)
        ]))
        
    def update(self, chosen_arm: int, reward: float) -> None:
        super().update(chosen_arm, reward)
        
        n = self.counts[chosen_arm]
        old_mean = self.means[chosen_arm]
        old_std = self.stds[chosen_arm]
        
        # Bayesian update for Gaussian distribution
        self.means[chosen_arm] = (
            (n - 1) * old_mean + reward
        ) / n
        
        if n > 1:
            self.stds[chosen_arm] = np.sqrt(
                ((n - 1) * (old_std ** 2) + 
                 (reward - old_mean) * (reward - self.means[chosen_arm])) / n
            )

def run_evolution(df: pd.DataFrame, n_arms: int = 2) -> Tuple[List[float], np.ndarray]:
    """Run Thompson Sampling evolution on simulation data."""
    times = sorted(df['Time'].unique())
    n_times = len(times)
    mab = GaussianThompsonSampling(n_arms)
    history = np.zeros((n_times, n_arms))
    
    for idx, t in enumerate(times):
        v2v_data = df[(df['Time'] == t) & (df['Protocol'] == 'V2V')].iloc[0]
        v2i_data = df[(df['Time'] == t) & (df['Protocol'] == 'V2I')].iloc[0]
        
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
        
        # Prevent divide by zero
        v2v_score = (
            0.5 * (v2v_delay / max_delay if max_delay > 0 else 0.0) +
            0.3 * (v2v_loss / max_loss if max_loss > 0 else 0.0) +
            0.2 * (v2v_load / max_load if max_load > 0 else 0.0)
        )
        
        v2i_score = (
            0.5 * (v2i_delay / max_delay if max_delay > 0 else 0.0) +
            0.3 * (v2i_loss / max_loss if max_loss > 0 else 0.0) +
            0.2 * (v2i_load / max_load if max_load > 0 else 0.0)
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
            
        history[idx] = mab.means
        
    return times, history

def plot_evolution(times, history, save_path=None):
    """Plot the evolution of both arms over time."""
    plt.figure(figsize=(10, 6))
    plt.plot(times, history[:, 0], label='V2V', color='blue')
    plt.plot(times, history[:, 1], label='V2I', color='red')
    plt.xlabel('Time')
    plt.ylabel('Mean Value')
    plt.title('Thompson Sampling: Protocol Performance Evolution')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def compare_protocols(df: pd.DataFrame, save_path: Optional[Path] = None) -> Tuple[str, float]:
    """Compare V2V and V2I protocols using Thompson Sampling strategy."""
    logger.info("Starting Thompson Sampling comparison")
    
    times, history = run_evolution(df)
    
    if save_path:
        plot_evolution(times, history, save_path)
    
    mab = GaussianThompsonSampling(n_arms=2)
    
    # Process data chronologically
    for t in sorted(df['Time'].unique()):
        v2v_data = df[(df['Time'] == t) & (df['Protocol'] == 'V2V')].iloc[0]
        v2i_data = df[(df['Time'] == t) & (df['Protocol'] == 'V2I')].iloc[0]
        
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
        
        # Prevent divide by zero
        v2v_score = (
            0.5 * (v2v_delay / max_delay if max_delay > 0 else 0.0) +
            0.3 * (v2v_loss / max_loss if max_loss > 0 else 0.0) +
            0.2 * (v2v_load / max_load if max_load > 0 else 0.0)
        )
        
        v2i_score = (
            0.5 * (v2i_delay / max_delay if max_delay > 0 else 0.0) +
            0.3 * (v2i_loss / max_loss if max_loss > 0 else 0.0) +
            0.2 * (v2i_load / max_load if max_load > 0 else 0.0)
        )
        
        # Convert to rewards (higher is better)
        v2v_reward = 1.0 - v2v_score
        v2i_reward = 1.0 - v2i_score
        
        # Update MAB with both rewards
        mab.update(0, v2v_reward)
        mab.update(1, v2i_reward)
    
    # Determine best protocol based on final means
    v2v_value = mab.means[0]
    v2i_value = mab.means[1]
    best_protocol = 'V2V' if v2v_value > v2i_value else 'V2I'
    best_value = max(v2v_value, v2i_value)
    
    logger.info(f"Final values after {len(df['Time'].unique())} iterations:")
    logger.info(f"V2V: {v2v_value:.3f} ± {mab.stds[0]:.3f}")
    logger.info(f"V2I: {v2i_value:.3f} ± {mab.stds[1]:.3f}")
    logger.info(f"Best protocol: {best_protocol}")
    
    return best_protocol, best_value
