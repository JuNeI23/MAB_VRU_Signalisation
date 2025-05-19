"""
Metrics collection and calculation for the simulation.
Simplified version that maintains core functionality.
"""
from dataclasses import dataclass, field
from typing import List
import numpy as np
from .models import Message

@dataclass
class Metric:
    """Collect and calculate communication metrics."""
    total_delay: float = 0.0
    message_count: int = 0
    failed_count: int = 0
    total_load: float = 0.0
    
    def update(self, message: Message, success: bool) -> None:
        """Update metrics with a new message transmission."""
        self.message_count += 1
        if success:
            self.total_delay += message.delay
        else:
            self.failed_count += 1
        self.total_load += 1.0
    
    @property
    def average_delay(self) -> float:
        """Calculate average delay for successful transmissions."""
        if self.message_count - self.failed_count <= 0:
            return float('inf')
        return self.total_delay / (self.message_count - self.failed_count)
    
    @property
    def loss_rate(self) -> float:
        """Calculate packet loss rate."""
        if self.message_count <= 0:
            return 0.0
        return self.failed_count / self.message_count
    
    @property
    def average_load(self) -> float:
        """Calculate average network load."""
        if self.message_count <= 0:
            return 0.0
        return self.total_load / self.message_count