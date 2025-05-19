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
