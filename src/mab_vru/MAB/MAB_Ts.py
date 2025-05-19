"""
Thompson Sampling Multi-Armed Bandit implementation.

This module provides an implementation of Thompson Sampling for the
Multi-Armed Bandit problem, using Gaussian priors for value estimation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List, Union, Optional
import logging

from .base_mab import BaseMAB

logger = logging.getLogger(__name__)

class GaussianThompsonSampling(BaseMAB):
    """
    Thompson Sampling MAB implementation with Gaussian priors.
    
    This implementation uses Gaussian (Normal) distributions to model
    uncertainty in the value estimates, with conjugate Normal priors.
    """
    
    def __init__(
        self,
        n_arms: int,
        prior_mean: float = 0.0,
        prior_std: float = 1.0
    ):
        """
        Initialize Thompson Sampling MAB.
        
        Args:
            n_arms: Number of arms (actions)
            prior_mean: Mean of the prior distribution
            prior_std: Standard deviation of the prior
            
        The Gaussian implementation assumes rewards are normally
        distributed around true means.
        """
        super().__init__(n_arms)
        
        # Parameters for Gaussian priors
        self.means = [prior_mean] * n_arms
        self.stds = [prior_std] * n_arms
        
        logger.info(
            f"Initialized Thompson Sampling MAB with {n_arms} arms, "
            f"prior_mean={prior_mean}, prior_std={prior_std}"
        )
        
    def select_arm(self) -> int:
        """
        Select an arm using Thompson Sampling.
        
        Samples a value from each arm's posterior distribution
        and selects the arm with highest sampled value.
        
        Returns:
            Index of the selected arm
        """
        # Sample from each arm's posterior
        samples = [
            self._rng.normal(self.means[i], self.stds[i])
            for i in range(self.n_arms)
        ]
        
        selected_arm = int(np.argmax(samples))
        logger.debug(
            f"Selected arm {selected_arm} with "
            f"mean={self.means[selected_arm]:.3f}, "
            f"std={self.stds[selected_arm]:.3f}, "
            f"sample={samples[selected_arm]:.3f}"
        )
        return selected_arm
        
    def update(self, chosen_arm: int, reward: float) -> None:
        """
        Update posterior distribution for the chosen arm.
        
        Args:
            chosen_arm: Index of the arm that was played
            reward: Reward received from playing the arm
            
        Updates both mean and standard deviation of the
        posterior distribution using Bayesian update rules.
        """
        super().update(chosen_arm, reward)
        
        # Update posterior parameters
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
        
        logger.debug(
            f"Updated arm {chosen_arm}: "
            f"mean={self.means[chosen_arm]:.3f}, "
            f"std={self.stds[chosen_arm]:.3f}"
        )

def run_evolution(
    df: pd.DataFrame,
    n_arms: int = 2
) -> Tuple[List[Union[int, float]], np.ndarray]:
    """
    Run Thompson Sampling evolution on simulation data.
    
    Args:
        df: DataFrame with simulation results
        n_arms: Number of arms (default: 2 for V2V/V2I)
        
    Returns:
        Tuple of:
        - List of simulation times
        - History of estimated values (shape: n_times x n_arms)
    """
    logger.info("Starting Thompson Sampling evolution")
    
    times = sorted(df['Temps'].unique())
    n_times = len(times)
    mab = GaussianThompsonSampling(n_arms)
    history = np.zeros((n_times, n_arms))
    
    for idx, t in enumerate(times):
        # Get data for current timestep
        v2v_data = df[(df['Temps'] == t) & (df['Protocole'] == 'V2V')].iloc[0]
        v2i_data = df[(df['Temps'] == t) & (df['Protocole'] == 'V2I')].iloc[0]
        
        # Calculate rewards with weighted metrics
        reward_v2v = -(
            0.5 * v2v_data['Délai moyen (s)'] +
            0.3 * v2v_data['Taux de perte (%)'] +
            0.2 * v2v_data['Charge moyenne']
        )
        reward_v2i = -(
            0.5 * v2i_data['Délai moyen (s)'] +
            0.3 * v2i_data['Taux de perte (%)'] +
            0.2 * v2i_data['Charge moyenne']
        )
        
        # Select and update
        arm = mab.select_arm()
        if arm == 0:
            mab.update(0, reward_v2v)
        else:
            mab.update(1, reward_v2i)
            
        # Record history
        history[idx] = mab.means
        
        if (idx + 1) % 10 == 0:
            logger.debug(
                f"Step {idx+1}/{n_times}: "
                f"V2V={mab.means[0]:.3f}±{mab.stds[0]:.3f}, "
                f"V2I={mab.means[1]:.3f}±{mab.stds[1]:.3f}"
            )
            
    logger.info("Thompson Sampling evolution completed")
    return times, history

def plot_evolution(df: pd.DataFrame) -> None:
    """Plot evolution of estimated values with confidence intervals."""
    times, history = run_evolution(df)
    
    plt.figure(figsize=(10, 6))
    plt.plot(times, history[:, 0], label='V2V', color='blue')
    plt.plot(times, history[:, 1], label='V2I', color='red')
    plt.xlabel('Time')
    plt.ylabel('Estimated Value')
    plt.title('Thompson Sampling Evolution')
    plt.legend()
    plt.grid(True)
    plt.savefig('Figure_3.png')
    plt.close()
    
def compare_protocols(df: pd.DataFrame) -> Tuple[str, float]:
    """
    Compare V2V and V2I protocols using Thompson Sampling.
    
    Args:
        df: DataFrame with simulation results
        
    Returns:
        Tuple of (best_protocol, best_value)
    """
    logger.info("Starting Thompson Sampling protocol comparison")
    
    # Initialize MABs for each protocol
    protocol_mabs = {
        'V2V': GaussianThompsonSampling(n_arms=3),
        'V2I': GaussianThompsonSampling(n_arms=3)
    }
    
    # Update MABs with simulation data
    metrics = ['Average Delay (s)', 'Loss Rate (%)', 'Average Load']
    
    for protocol in protocol_mabs:
        protocol_data = df[df['Protocol'] == protocol].sort_values('Time')
        for arm, metric in enumerate(metrics):
            values = protocol_data[metric].values
            for value in values:
                # Normalize reward to [0, 1] - lower is better
                reward = 1 - (value / df[metric].max())
                protocol_mabs[protocol].update(arm, reward)
    
    # Get final values (using mean estimates)
    v2v_value = max(protocol_mabs['V2V'].means)
    v2i_value = max(protocol_mabs['V2I'].means)
    
    # Determine best protocol
    best_protocol = 'V2V' if v2v_value > v2i_value else 'V2I'
    best_value = max(v2v_value, v2i_value)
    
    logger.info(f"V2V final mean value: {v2v_value:.3f}")
    logger.info(f"V2I final mean value: {v2i_value:.3f}")
    logger.info(f"Best protocol: {best_protocol}")
    
    return best_protocol, best_value
