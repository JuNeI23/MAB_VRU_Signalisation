"""
Protocol implementation for V2V and V2I communication.
"""
from dataclasses import dataclass, field
from typing import Optional, ClassVar
import time
import logging
import numpy as np
from functools import wraps

logger = logging.getLogger(__name__)

def validate_probability(func):
    """Decorator to validate probability values."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        for arg in args:
            if isinstance(arg, float) and not 0 <= arg <= 1:
                raise ValueError(f"Probability must be between 0 and 1, got {arg}")
        return func(self, *args, **kwargs)
    return wrapper

@dataclass
class Protocole:
    """
    Communication protocol implementation with improved validation and performance.
    
    Attributes:
        name: Protocol identifier
        network_load: Current network load (0-1)
        packet_loss_rate: Probability of packet loss (0-1)
        transmission_time: Base transmission time in seconds
        transmission_success_rate: Probability of successful transmission (0-1)
    """
    name: str
    network_load: float
    packet_loss_rate: float
    transmission_time: float
    transmission_success_rate: float = field(default=0.9)
    
    # Class-level constants
    MAX_TRANSMISSION_TIME: ClassVar[float] = 5.0
    MIN_TRANSMISSION_TIME: ClassVar[float] = 0.001
    
    def __post_init__(self):
        """Validate parameters and initialize RNG."""
        self._validate_parameters()
        self._rng_seed = np.random.randint(0, 2**32)
        self._initialize_rng()
    
    def _initialize_rng(self):
        """Initialize random number generator with saved seed."""
        self._rng = np.random.RandomState(self._rng_seed)
    
    def __getstate__(self):
        """Get object state for pickling."""
        state = self.__dict__.copy()
        # Remove the RNG instance as it can't be pickled
        del state['_rng']
        return state
    
    def __setstate__(self, state):
        """Restore object state after unpickling."""
        self.__dict__.update(state)
        self._initialize_rng()
    
    @validate_probability
    def _validate_parameters(self) -> None:
        """Validate protocol parameters."""
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Protocol name must be a non-empty string")
            
        if not 0 < self.transmission_time <= self.MAX_TRANSMISSION_TIME:
            raise ValueError(
                f"Transmission time must be between 0 and {self.MAX_TRANSMISSION_TIME}"
            )
    
    def transmit_message(self, sender, message, receiver) -> Optional[float]:
        """
        Optimized message transmission with improved error handling.
        
        Args:
            sender: Sending node
            message: Message to transmit
            receiver: Receiving node
            
        Returns:
            Optional[float]: Transmission delay if successful, None if failed
        """
        try:
            # Check transmission success
            if self._rng.random() >= self.transmission_success_rate:
                self.update_network_load()
                return None

            # Validate distance
            distance = sender.distance_to(receiver)
            if distance > sender.range:
                logger.debug(f"Distance {distance} exceeds range {sender.range}")
                return None

            # Calculate and apply delay
            delay = max(
                self.MIN_TRANSMISSION_TIME,
                self.transmission_time + 0.01 * distance
            )
            
            if delay >= self.MIN_TRANSMISSION_TIME:
                time.sleep(delay)
                
            self.update_network_load()
            return delay
            
        except Exception as e:
            logger.error(f"Transmission error: {str(e)}")
            return None

    def update_network_load(self) -> None:
        """Update network load with randomization."""
        try:
            # Add some randomness to network load
            variation = self._rng.uniform(-0.1, 0.1)
            self.network_load = max(0, min(1, self.network_load + variation))
        except Exception as e:
            logger.error(f"Error updating network load: {str(e)}")
            self.network_load = 0.5  # Fallback to middle value
