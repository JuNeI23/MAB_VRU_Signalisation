"""
Multi-Armed Bandit algorithms package.
"""
from .MAB_e import compare_protocols as epsilon_greedy_compare
from .MAB_u import compare_protocols as ucb_compare
from .MAB_Ts import compare_protocols as thompson_compare
from .base_mab import BaseMAB

__all__ = [
    'epsilon_greedy_compare',
    'ucb_compare',
    'thompson_compare',
    'BaseMAB'
]
