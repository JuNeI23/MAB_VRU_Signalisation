"""
Upper Confidence Bound (UCB) Multi-Armed Bandit implementation.

This module provides an implementation of the UCB1 algorithm for the
Multi-Armed Bandit problem, using confidence bounds for exploration.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List, Union, Optional
import logging

from .base_mab import BaseMAB

logger = logging.getLogger(__name__)

class UCBMAB(BaseMAB):
    """
    UCB1 MAB implementation with improved confidence bounds.
    
    This implementation uses the UCB1 algorithm to balance exploration
    and exploitation based on uncertainty in value estimates.
    """
    
    def __init__(self, n_arms: int):
        """
        Initialize UCB MAB.
        
        Args:
            n_arms: Number of arms (actions)
            
        The UCB1 algorithm requires no additional parameters beyond
        the number of arms, as it automatically balances exploration
        and exploitation based on uncertainty.
        """
        super().__init__(n_arms)
        self.total_counts = 0
        logger.info(f"Initialized UCB MAB with {n_arms} arms")
        
    def select_arm(self) -> int:
        """
        Select an arm using UCB1 strategy.
        
        First plays each arm once, then selects the arm maximizing
        UCB = empirical_mean + sqrt(2 * ln(total_plays) / arm_plays)
        
        Returns:
            Index of the selected arm
        """
        # Play each arm once initially
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                logger.debug(f"Initial play of arm {arm}")
                return arm
                
        # Calculate UCB values for each arm
        ucb_values = [
            self.values[arm] + np.sqrt(
                2 * np.log(self.total_counts) / self.counts[arm]
            )
            for arm in range(self.n_arms)
        ]
        
        selected_arm = int(np.argmax(ucb_values))
        logger.debug(
            f"Selected arm {selected_arm} with "
            f"value={self.values[selected_arm]:.3f}, "
            f"UCB={ucb_values[selected_arm]:.3f}"
        )
        return selected_arm
        
    def update(self, chosen_arm: int, reward: float) -> None:
        """
        Update value estimate and counts for the chosen arm.
        
        Args:
            chosen_arm: Index of the arm that was played
            reward: Reward received from playing the arm
            
        Raises:
            ValueError: If chosen_arm is invalid
        """
        super().update(chosen_arm, reward)
        self.total_counts += 1

def run_evolution(
    df: pd.DataFrame,
    n_arms: int = 2
) -> Tuple[List[Union[int, float]], np.ndarray]:
    """
    Run UCB evolution on simulation data.
    
    Args:
        df: DataFrame with simulation results
        n_arms: Number of arms (default: 2 for V2V/V2I)
        
    Returns:
        Tuple of:
        - List of simulation times
        - History of estimated values (shape: n_times x n_arms)
    """
    logger.info("Starting UCB evolution")
    
    times = sorted(df['Temps'].unique())
    n_times = len(times)
    mab = UCBMAB(n_arms)
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
        history[idx] = mab.values
        
        if (idx + 1) % 10 == 0:
            logger.debug(
                f"Step {idx+1}/{n_times}: "
                f"V2V={mab.values[0]:.3f}, V2I={mab.values[1]:.3f}"
            )
            
    logger.info("UCB evolution completed")
    return times, history

def plot_evolution(df: pd.DataFrame) -> None:
    """Plot evolution of estimated values."""
    times, history = run_evolution(df)
    
    plt.figure(figsize=(10, 6))
    plt.plot(times, history[:, 0], label='V2V', color='blue')
    plt.plot(times, history[:, 1], label='V2I', color='red')
    plt.xlabel('Time')
    plt.ylabel('Estimated Value')
    plt.title('UCB Evolution')
    plt.legend()
    plt.grid(True)
    plt.savefig('Figure_2.png')
    plt.close()
    
def compare_protocols(df: pd.DataFrame) -> None:
    """Compare final protocol performance."""
    _, history = run_evolution(df)
    
    final_v2v = history[-1, 0]
    final_v2i = history[-1, 1]
    
    logger.info("\nFinal Protocol Comparison:")
    logger.info(f"V2V: {final_v2v:.3f}")
    logger.info(f"V2I: {final_v2i:.3f}")
    
    if final_v2v > final_v2i:
        logger.info("V2V performs better")
    elif final_v2i > final_v2v:
        logger.info("V2I performs better")
    else:
        logger.info("Both protocols perform equally")
