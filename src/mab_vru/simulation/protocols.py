"""
Protocol implementations for V2V and V2I communication.
Simplified version that maintains core functionality.
"""
from dataclasses import dataclass
import random
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .models import Message

@dataclass
class Protocol:
    """Protocol configuration and behavior."""
    name: str
    network_load: float
    packet_loss_rate: float
    transmission_time: float
    
    def __post_init__(self):
        """Validate protocol parameters."""
        if not 0 <= self.network_load <= 1:
            raise ValueError("Network load must be between 0 and 1")
        if not 0 <= self.packet_loss_rate <= 1:
            raise ValueError("Packet loss rate must be between 0 and 1")
        if self.transmission_time < 0:
            raise ValueError("Transmission time must be non-negative")
    
    def calculate_reward(self, success: bool, distance: float, range_: float, delay: float) -> float:
        """
        Calculate reward based on multiple factors:
        - Success/failure of transmission
        - Distance relative to range (normalized by protocol type)
        - Network delay
        - Reachability (based on whether distance is within range)
        
        Returns:
            float: Reward value between 0 and 1
        """
        if not success:
            return 0.0
            
        # Distance factor: Normalized differently for V2V and V2I due to their different scales
        if self.name == "V2V":
            # For V2V, prefer shorter distances (0-3m is optimal)
            distance_factor = max(0, 1 - (distance / 3.0))
        else:  # V2I
            # For V2I, distances up to 100m are acceptable
            distance_factor = max(0, 1 - (distance / 100.0))
        
        # Delay factor: 1 when delay is minimal, decreasing as delay increases
        delay_factor = 1 / (1 + delay)  # Uses a decay function
        
        # Network load factor: 1 when load is low, decreasing as load increases
        load_factor = 1 - self.network_load
        
        # Reachability factor: Binary reward for being within range
        reachability_factor = 1.0 if distance <= range_ else 0.0
        
        # Combine factors with weights
        # - Increased weight for reachability as it's critical
        # - Reduced weight for raw distance as it's now normalized by protocol
        # - Maintained weights for delay and load as they remain important
        reward = (
            0.35 * reachability_factor +  # Reachability is critical
            0.25 * distance_factor +      # Distance affects reliability
            0.25 * delay_factor +         # Delay is important for real-time communication
            0.15 * load_factor            # Network load affects scalability
        )
        
        return reward
    
    def transmit_message(self, message: Any) -> bool:
        """
        Simulate message transmission with the configured parameters.
        Returns True if transmission was successful.
        """
        # Apply network load effect
        if random.random() < self.network_load:
            message.delay += self.transmission_time * 1.5
        else:
            message.delay += self.transmission_time
            
        # Check for packet loss
        return random.random() > self.packet_loss_rate
