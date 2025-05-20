"""
Unit tests for MAB algorithms implementations.
"""
import pytest
import numpy as np
from MAB.base_mab import BaseMAB
from MAB.MAB_e import EpsilonGreedyMAB
from MAB.MAB_u import UCBMAB
from MAB.MAB_Ts import ThompsonSampling

def test_epsilon_greedy_initialization():
    """Test epsilon-greedy algorithm initialization."""
    epsilon = 0.1
    n_arms = 2
    mab = EpsilonGreedyMAB(n_arms=n_arms, epsilon=epsilon)
    
    assert mab.n_arms == n_arms
    assert mab.epsilon == epsilon
    assert len(mab.counts) == n_arms
    assert len(mab.values) == n_arms
    assert all(count == 0 for count in mab.counts)
    assert all(value == 0.0 for value in mab.values)

def test_ucb_initialization():
    """Test UCB algorithm initialization."""
    n_arms = 2
    mab = UCBMAB(n_arms=n_arms)
    
    assert mab.n_arms == n_arms
    assert len(mab.counts) == n_arms
    assert len(mab.values) == n_arms
    assert all(count == 0 for count in mab.counts)
    assert all(value == 0.0 for value in mab.values)

def test_thompson_sampling_initialization():
    """Test Thompson Sampling algorithm initialization."""
    n_arms = 2
    mab = ThompsonSampling(n_arms=n_arms)
    
    assert mab.n_arms == n_arms
    assert len(mab.counts) == n_arms
    assert len(mab.values) == n_arms
    assert all(count == 0 for count in mab.counts)
    assert all(value == 0.0 for value in mab.values)

def test_invalid_arm_selection():
    """Test error handling for invalid arm selection."""
    mab = EpsilonGreedyMAB(n_arms=2, epsilon=0.1)
    
    with pytest.raises(ValueError):
        mab.validate_arm(-1)
    with pytest.raises(ValueError):
        mab.validate_arm(2)

def test_reward_update():
    """Test reward update mechanism."""
    mab = EpsilonGreedyMAB(n_arms=2, epsilon=0.1)
    arm = 0
    reward = 1.0
    
    # Initial state
    assert mab.counts[arm] == 0
    assert mab.values[arm] == 0.0
    
    # After update
    mab.update(arm, reward)
    assert mab.counts[arm] == 1
    assert mab.values[arm] == reward
    
    # Second update
    mab.update(arm, 0.0)
    assert mab.counts[arm] == 2
    assert mab.values[arm] == reward / 2  # Average of 1.0 and 0.0

def test_exploration_exploitation_balance():
    """Test that epsilon-greedy balances exploration and exploitation."""
    np.random.seed(42)  # For reproducibility
    mab = EpsilonGreedyMAB(n_arms=2, epsilon=0.5)
    
    # Make one arm clearly better
    mab.update(0, 1.0)  # Good arm
    mab.update(1, 0.0)  # Bad arm
    
    # Run multiple selections
    selections = [mab.select_arm() for _ in range(1000)]
    
    # Count selections of each arm
    arm0_count = selections.count(0)
    arm1_count = selections.count(1)
    
    # With epsilon=0.5, we expect roughly:
    # 75% selections for arm 0 (50% exploitation + 25% exploration)
    # 25% selections for arm 1 (25% exploration)
    assert 650 < arm0_count < 850  # Roughly 75%
    assert 150 < arm1_count < 350  # Roughly 25%
