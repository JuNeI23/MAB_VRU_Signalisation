"""
Base Multi-Armed Bandit implementation.
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict
import numpy as np
import logging

logger = logging.getLogger(__name__)

class BaseMAB(ABC):
    """
    Abstract base class for Multi-Armed Bandit algorithms.
    
    This class provides common functionality for all MAB implementations,
    including value tracking, arm selection validation, and reward updates.
    """
    
    def __init__(self, n_arms: int):
        """
        Initialize the base MAB.
        
        Args:
            n_arms: Number of arms (actions) available
            
        Raises:
            ValueError: If n_arms is not positive
        """
        if n_arms <= 0:
            raise ValueError("Number of arms must be positive")
            
        self.n_arms = n_arms
        self.counts = [0] * n_arms
        self.values = [0.0] * n_arms
        self._rng = np.random.RandomState()
        self.selections = [0] * n_arms  # Track number of times each arm was selected
        self.total_selections = 0  # Track total number of selections
        
    def validate_arm(self, arm: int) -> None:
        """
        Validate arm index.
        
        Args:
            arm: Index of the arm to validate
            
        Raises:
            ValueError: If arm index is invalid
        """
        if not 0 <= arm < self.n_arms:
            raise ValueError(f"Arm index must be between 0 and {self.n_arms-1}")
    
    @abstractmethod
    def select_arm(self) -> int:
        """
        Select an arm to play.
        
        Returns:
            Index of the selected arm
        """
        pass
    
    def update(self, chosen_arm: int, reward: float) -> None:
        """
        Update value estimate for the chosen arm.
        
        Args:
            chosen_arm: Index of the arm that was played
            reward: Reward received from playing the arm
            
        Raises:
            ValueError: If chosen_arm is invalid
        """
        self.validate_arm(chosen_arm)
        
        # Update counts and values
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        
        # Incremental update formula
        self.values[chosen_arm] = ((n - 1) * value + reward) / n
        
        # Track selections
        self.selections[chosen_arm] += 1
        self.total_selections += 1
        
        logger.debug(
            f"Updated arm {chosen_arm}: count={n}, "
            f"old_value={value:.3f}, new_value={self.values[chosen_arm]:.3f}"
        )
    
    def get_selection_rate(self, arm_index: Optional[int] = None) -> float:
        """
        Get the selection rate for a specific arm or all arms.
        
        Args:
            arm_index: Index of the arm to get rate for, or None for all arms
            
        Returns:
            Selection rate as a float between 0 and 1
        """
        if self.total_selections == 0:
            return 0.0
            
        if arm_index is not None:
            self.validate_arm(arm_index)
            return self.selections[arm_index] / self.total_selections
            
        return [s / self.total_selections for s in self.selections]
    
    def reset(self) -> None:
        """Reset the bandit to initial state."""
        self.counts = [0] * self.n_arms
        self.values = [0.0] * self.n_arms
        self.selections = [0] * self.n_arms
        self.total_selections = 0
