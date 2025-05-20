"""
Epsilon-Greedy Multi-Armed Bandit implementation.

This module provides an implementation of the ε-greedy strategy for the
Multi-Armed Bandit problem, balancing exploration and exploitation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List, Union, Optional
from tqdm import tqdm
import logging

from .base_mab import BaseMAB

logger = logging.getLogger(__name__)

class EpsilonGreedyMAB(BaseMAB):
    """
    Epsilon-Greedy MAB implementation with improved exploration control.
    
    This implementation uses a constant epsilon value to balance
    exploration vs exploitation, with options for annealing.
    """
    
    def __init__(self, n_arms: int, epsilon: float) -> None:
        """
        Initialize ε-greedy MAB.
        
        Args:
            n_arms: Number of arms (actions)
            epsilon: Exploration probability (0 ≤ ε ≤ 1)
            
        Raises:
            ValueError: If parameters are invalid
        """
        super().__init__(n_arms)
        
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be between 0 and 1")
            
        self.epsilon = epsilon
        logger.info(f"Initialized ε-greedy MAB with {n_arms} arms, ε={epsilon}")
        
    def select_arm(self) -> int:
        """
        Select an arm using ε-greedy strategy.
        
        With probability ε, selects a random arm (exploration).
        With probability 1-ε, selects the arm with highest estimated value (exploitation).
        
        Returns:
            Index of the selected arm
        """
        if self._rng.random() < self.epsilon:
            # Exploration: random arm
            selected_arm = self._rng.randint(self.n_arms)
            logger.debug(f"Exploring: selected arm {selected_arm}")
            return selected_arm
            
        # Exploitation: best arm
        selected_arm = int(np.argmax(self.values))
        logger.debug(
            f"Exploiting: selected arm {selected_arm} "
            f"with value {self.values[selected_arm]:.3f}"
        )
        return selected_arm

def run_evolution(
    df: pd.DataFrame,
    epsilon: float,
    n_arms: int = 2
) -> Tuple[List[Union[int, float]], np.ndarray]:
    """
    Run ε-greedy evolution on simulation data.
    
    Args:
        df: DataFrame with simulation results
        epsilon: Exploration parameter
        n_arms: Number of arms (default: 2 for V2V/V2I)
        
    Returns:
        Tuple of:
        - List of simulation times
        - History of estimated values (shape: n_times x n_arms)
    """
    logger.info(f"Starting ε-greedy evolution (ε={epsilon:.2f})")
    
    times = sorted(df['Temps'].unique())
    n_times = len(times)
    eg = EpsilonGreedyMAB(n_arms, epsilon)
    history = np.zeros((n_times, n_arms))
    
    with tqdm(total=n_times, desc="ε-greedy Evolution") as pbar:
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
            arm = eg.select_arm()
            if arm == 0:
                eg.update(0, reward_v2v)
            else:
                eg.update(1, reward_v2i)
                
            # Record history
            history[idx] = eg.values
            
            # Update progress
            if (idx + 1) % 10 == 0:
                pbar.set_postfix({
                    'V2V': f"{eg.values[0]:.3f}",
                    'V2I': f"{eg.values[1]:.3f}"
                })
            pbar.update(1)
            
    logger.info("ε-greedy evolution completed")
    return times, history

def plot_evolution(df: pd.DataFrame, epsilon: float = 0.1) -> None:
    """Plot evolution of estimated values."""
    times, history = run_evolution(df, epsilon)
    
    plt.figure(figsize=(10, 6))
    plt.plot(times, history[:, 0], label='V2V', color='blue')
    plt.plot(times, history[:, 1], label='V2I', color='red')
    plt.xlabel('Time')
    plt.ylabel('Estimated Value')
    plt.title(f'ε-greedy Evolution (ε={epsilon})')
    plt.legend()
    plt.grid(True)
    plt.savefig('Figure_1.png')
    plt.close()
    
def compare_protocols(df: pd.DataFrame, epsilon: float = 0.1) -> None:
    """Compare final protocol performance."""
    _, history = run_evolution(df, epsilon)
    
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
