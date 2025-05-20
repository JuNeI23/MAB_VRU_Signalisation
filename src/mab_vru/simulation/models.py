"""
Core models for VRU simulation.
Simplified version that maintains essential functionality.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING, Any
import math

if TYPE_CHECKING:
    from .protocols import Protocol

@dataclass
class Message:
    """Communication message between nodes."""
    sender_id: str
    receiver_id: str
    priority: int
    creation_time: float
    delay: float = 0.0
    size: float = 1.0  # Default message size
    
    def __post_init__(self):
        if self.priority < 0:
            raise ValueError("Priority must be non-negative")
        if self.delay < 0:
            raise ValueError("Delay must be non-negative")
        if self.size <= 0:
            raise ValueError("Message size must be positive")

class Node(ABC):
    """Base class for network nodes."""
    def __init__(
        self,
        x: float,
        y: float,
        range_: float = 100.0,
        priority: int = 0,
        processing_capacity: int = 100,
        user_id: Optional[str] = None,
    ):
        self.x = float(x)
        self.y = float(y)
        self.range = float(range_)
        self.priority = int(priority)
        self.processing_capacity = int(processing_capacity)
        self.user_id = str(user_id) if user_id else str(id(self))
    
    def distance_to(self, other: 'Node') -> float:
        """Calculate Euclidean distance to another node."""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def in_range(self, other: 'Node') -> bool:
        """Check if another node is within communication range."""
        return self.distance_to(other) <= self.range
    
    def create_message(self, receiver_id: Optional[str] = None) -> Message:
        """Create a new message from this node."""
        return Message(
            sender_id=self.user_id,
            receiver_id=receiver_id or "broadcast",
            priority=self.priority,
            creation_time=0.0
        )

class User(Node):
    """Vehicle or VRU in the simulation."""
    def __init__(
        self,
        usager_id: str,
        x: float,
        y: float,
        angle: float,
        speed: float,
        position: float,
        lane: str,
        time: float,
        usager_type: str = "car",
        categorie: str = "vehicule",
        protocol: Optional['Protocol'] = None
    ):
        super().__init__(x=x, y=y, user_id=usager_id)
        self.angle = float(angle)
        self.speed = float(speed)
        self.position = float(position)
        self.lane = str(lane)
        self.time = float(time)
        self.usager_type = str(usager_type)
        self.categorie = str(categorie)
        self.protocol = protocol

class Infrastructure(Node):
    """Fixed infrastructure node."""
    def __init__(
        self,
        id: str,
        protocol: Any,  # Changed from 'Protocole' to Any to break circular import
        x: float,
        y: float,
        processing_capacity: int,
        time: float,
    ):
        super().__init__(
            x=x,
            y=y,
            processing_capacity=processing_capacity,
            user_id=id
        )
        self.protocol = protocol
        self.time = float(time)