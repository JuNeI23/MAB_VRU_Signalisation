"""
Metrics collection and calculation for the simulation.
Simplified version that maintains core functionality.
"""
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
from .models import Message

@dataclass
class Metric:
    """Collect and calculate communication metrics."""
    total_delay: float = 0.0
    message_count: int = 0
    failed_count: int = 0
    total_load: float = 0.0
    total_distance: float = 0.0  # Added for range tracking
    reachable_count: int = 0  # Track nodes in range
    unreachable_count: int = 0  # Track nodes out of range
    
    def update(self, message: Message, success: bool, distance: Optional[float] = None, range_: Optional[float] = None) -> None:
        """Update metrics with a new message transmission."""
        self.message_count += 1
        if success:
            self.total_delay += message.delay
        else:
            self.failed_count += 1
        self.total_load += 1.0
        
        if distance is not None and range_ is not None:
            self.total_distance += distance
            if distance <= range_:
                self.reachable_count += 1
            else:
                self.unreachable_count += 1
    
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
        
    @property
    def average_distance(self) -> float:
        """Calculate average distance between communicating nodes."""
        if self.message_count <= 0:
            return 0.0
        return self.total_distance / self.message_count
    
    @property
    def reachability_rate(self) -> float:
        """Calculate percentage of nodes that were within communication range."""
        total = self.reachable_count + self.unreachable_count
        if total <= 0:
            return 0.0
        return self.reachable_count / total