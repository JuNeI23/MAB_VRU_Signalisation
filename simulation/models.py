"""
Models for VRU simulation including Node, User, and Infrastructure implementations.
"""
from typing import Optional, Dict, List, TypeVar, Generic, TYPE_CHECKING
import time
import numpy as np
from dataclasses import dataclass, field
import logging
from abc import ABC, abstractmethod
from .pickleable_queue import PickleableQueue

if TYPE_CHECKING:
    from .protocols import Protocole
    from .metric import Metric

logger = logging.getLogger(__name__)

@dataclass(order=True, slots=True)
class Message:
    """Message implementation with pickleable attributes."""
    priority: int
    sender_id: str = field(compare=False)
    receiver_id: str = field(compare=False)
    size: float = field(compare=False)
    timestamp: float = field(default_factory=time.time, compare=False)
    
    def __post_init__(self):
        """Validate message parameters."""
        if self.size <= 0:
            raise ValueError("Message size must be positive")
        if not self.sender_id or not self.receiver_id:
            raise ValueError("Sender and receiver IDs must not be empty")

class Node(ABC):
    """Abstract base class for network nodes with pickleable attributes."""
    __slots__ = (
        'x', 'y', 'range', 'priority', 'processing_capacity',
        'user_id', '_queue', 'protocol', '_last_positions'
    )
    
    def __init__(
        self,
        x: float,
        y: float,
        range_: float,
        priority: int,
        processing_capacity: int,
        user_id: Optional[str] = None,
    ):
        # First initialize all slots with default values
        for slot in self.__slots__:
            object.__setattr__(self, slot, None)
        
        try:
            # Now set the actual values with validation
            self._validate_parameters(x, y, range_, priority, processing_capacity)
            
            # Store coordinates and parameters with type conversion and safe defaults
            self.x = float(x if x is not None else 0.0)
            self.y = float(y if y is not None else 0.0)
            self.range = float(range_ if range_ is not None else 100.0)
            self.priority = int(priority if priority is not None else 0)
            self.processing_capacity = int(processing_capacity if processing_capacity is not None else 100)
            self.user_id = str(user_id) if user_id is not None else str(id(self))
            self._queue = PickleableQueue()
            self.protocol = None
            self._last_positions = {}
            
        except (ValueError, TypeError) as e:
            logger.warning(f"Error in Node initialization: {str(e)}. Using default values.")
            # Set safe defaults if conversion fails
            self.x = 0.0
            self.y = 0.0
            self.range = 100.0
            self.priority = 0
            self.processing_capacity = 100
            self.user_id = str(id(self))
            self._queue = PickleableQueue()
            self.protocol = None
            self._last_positions = {}

    @staticmethod
    def _validate_parameters(
        x: float,
        y: float,
        range_: float,
        priority: int,
        processing_capacity: int
    ) -> None:
        """Validate input parameters with more lenient rules."""
        try:
            if x is not None:
                float(x)
            if y is not None:
                float(y)
                
            # Allow zero or small positive values for range
            if range_ is not None:
                range_val = float(range_)
                if range_val < 0:
                    raise ValueError("Range must be non-negative")
                    
            # Allow zero or positive values for processing capacity
            if processing_capacity is not None:
                proc_cap = int(processing_capacity)
                if proc_cap < 0:
                    raise ValueError("Processing capacity must be non-negative")
                    
            # Allow any non-negative priority
            if priority is not None:
                prio = int(priority)
                if prio < 0:
                    raise ValueError("Priority must be non-negative")
                    
        except (ValueError, TypeError) as e:
            logger.warning(f"Parameter validation error: {str(e)}. Will use defaults.")

    def __getstate__(self):
        """Get state for pickling with all required attributes."""
        state = {}
        for slot in self.__slots__:
            try:
                value = getattr(self, slot)
                # Ensure coordinates are properly stored
                if slot in ['x', 'y', 'range']:
                    value = float(value) if value is not None else 0.0
                elif slot in ['priority', 'processing_capacity']:
                    value = int(value) if value is not None else 0
                elif slot == 'user_id':
                    value = str(value) if value is not None else str(id(self))
                elif slot == '_queue':
                    value = value if value is not None else PickleableQueue()
                elif slot == '_last_positions':
                    value = value if value is not None else {}
                state[slot] = value
            except AttributeError:
                logger.warning(f"Missing attribute {slot} in {self.__class__.__name__}")
                state[slot] = None
        return state

    def __setstate__(self, state):
        """Restore state with default values for missing attributes."""
        # First set all slots to ensure they exist
        for slot in self.__slots__:
            object.__setattr__(self, slot, None)
        
        # Now restore from state with validation
        for slot, value in state.items():
            try:
                if slot in ['x', 'y', 'range']:
                    value = float(value) if value is not None else 0.0
                elif slot in ['priority', 'processing_capacity']:
                    value = int(value) if value is not None else 0
                elif slot == 'user_id':
                    value = str(value) if value is not None else str(id(self))
                elif slot == '_queue':
                    value = value if value is not None else PickleableQueue()
                elif slot == '_last_positions':
                    value = value if value is not None else {}
                object.__setattr__(self, slot, value)
            except (ValueError, TypeError) as e:
                logger.warning(f"Error setting {slot} in {self.__class__.__name__}: {e}")
                # Set a safe default
                if slot in ['x', 'y', 'range']:
                    object.__setattr__(self, slot, 0.0)
                elif slot in ['priority', 'processing_capacity']:
                    object.__setattr__(self, slot, 0)
                elif slot == 'user_id':
                    object.__setattr__(self, slot, str(id(self)))
                elif slot == '_queue':
                    object.__setattr__(self, slot, PickleableQueue())
                elif slot == '_last_positions':
                    object.__setattr__(self, slot, {})

    def distance_to(self, other: 'Node') -> float:
        """Calculate distance to another node."""
        try:
            return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)
        except Exception as e:
            logger.error(f"Error calculating distance: {str(e)}")
            return float('inf')
    
    def within_range(self, other: 'Node') -> bool:
        """Check if another node is within communication range."""
        return self.distance_to(other) <= self.range
    
    def send_message(self, receiver: 'Node', size: float) -> None:
        """Send a message to another node."""
        try:
            if not self.within_range(receiver):
                return
            if size > self.processing_capacity:
                return
            message = Message(self.priority, self.user_id, receiver.user_id, size)
            self._queue.put(message)
        except Exception as e:
            logger.error(f"Error sending message: {str(e)}")
    
    @abstractmethod
    def process_messages(self, users: Dict[str, 'Node'], metric: 'Metric') -> None:
        """Process queued messages."""
        pass

class User(Node):
    """User node implementation with pickleable attributes."""
    __slots__ = (
        'angle', 'speed', 'position', 'lane', 'time',
        'usager_type', 'categorie', 'queue_size'
    )
    
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
        usager_type: str = "DEFAULT",
        categorie: str = "vru",
    ):
        # Validate and set category first as it affects other parameters
        categorie = str(categorie).lower()
        if categorie not in ["vru", "vehicule"]:
            raise ValueError("Invalid category. Must be 'vru' or 'vehicule'")
            
        # Set parameters based on category
        priority = 1 if categorie == "vru" else 2
        range_ = 90.0 if categorie == "vru" else 120.0
        processing_capacity = 1 if categorie == "vru" else 2
            
        # Initialize parent class
        super().__init__(
            x=float(x),
            y=float(y),
            range_=float(range_),
            priority=int(priority),
            processing_capacity=int(processing_capacity),
            user_id=str(usager_id)
        )
        
        # Initialize own attributes with explicit type conversion
        self.angle = float(angle)
        self.speed = float(speed)
        self.position = float(position)
        self.lane = str(lane)
        self.time = float(time)
        self.usager_type = str(usager_type)
        self.categorie = str(categorie)
        self.queue_size = int(10 if categorie == "vru" else 50)
    
    def __getstate__(self):
        """Get complete state including parent and own attributes."""
        state = super().__getstate__()
        for slot in self.__slots__:
            try:
                state[slot] = getattr(self, slot)
            except AttributeError as e:
                logger.error(f"Missing attribute in User.__getstate__: {str(e)}")
                raise
        return state
    
    def __setstate__(self, state):
        """Restore complete state including parent and own attributes."""
        super().__setstate__(state)
        for slot in self.__slots__:
            if slot in state:
                setattr(self, slot, state[slot])
            else:
                logger.error(f"Missing attribute in User state: {slot}")
                raise AttributeError(f"Missing required attribute: {slot}")

    def process_messages(self, users: Dict[str, 'Node'], metric: 'Metric') -> None:
        """Process queued messages with batching."""
        try:
            messages: List[Message] = []
            while not self._queue.empty() and len(messages) < self.queue_size:
                messages.append(self._queue.get())
                
            if not messages:
                return
                
            current_time = time.time()
            for message in messages:
                receiver = users.get(message.receiver_id)
                if not receiver:
                    continue
                    
                if receiver.protocol is None:
                    receiver.protocol = self.protocol
                    
                if self.protocol:
                    transmission_delay = self.protocol.transmit_message(
                        self, message, receiver
                    )
                    queue_delay = current_time - message.timestamp
                    metric.update_metrics(
                        transmission_delay,
                        queue_delay,
                        self.protocol.network_load
                    )
                    
        except Exception as e:
            logger.error(f"Error processing messages: {str(e)}")

class Infrastructure(Node):
    """Infrastructure node implementation with pickleable attributes."""
    __slots__ = ('time', 'user_type', 'categorie')
    
    def __init__(
        self,
        id: str,
        protocol: 'Protocole',
        x: float,
        y: float,
        processing_capacity: int,
        time: float,
    ):
        # Initialize parent class with explicit type conversions
        super().__init__(
            x=float(x),
            y=float(y),
            range_=300.0,  # Fixed range for infrastructure
            priority=0,     # Highest priority
            processing_capacity=int(processing_capacity),
            user_id=str(id)
        )
        
        # Initialize own attributes with explicit type conversion
        self.protocol = protocol
        self.time = float(time)
        self.user_type = "Infrastructure"
        self.categorie = "infrastructure"
    
    def __getstate__(self):
        """Get complete state including parent and own attributes."""
        state = super().__getstate__()
        for slot in self.__slots__:
            try:
                state[slot] = getattr(self, slot)
            except AttributeError as e:
                logger.error(f"Missing attribute in Infrastructure.__getstate__: {str(e)}")
                raise
        return state
    
    def __setstate__(self, state):
        """Restore complete state including parent and own attributes."""
        super().__setstate__(state)
        for slot in self.__slots__:
            if slot in state:
                setattr(self, slot, state[slot])
            else:
                logger.error(f"Missing attribute in Infrastructure state: {slot}")
                raise AttributeError(f"Missing required attribute: {slot}")

    def process_messages(self, users: Dict[str, 'Node'], metric: 'Metric') -> None:
        """Process messages with priority handling."""
        try:
            while not self._queue.empty():
                message = self._queue.get()
                receiver = users.get(message.receiver_id)
                
                if not receiver:
                    continue
                    
                if self.protocol:
                    transmission_delay = self.protocol.transmit_message(
                        self, message, receiver
                    )
                    queue_delay = time.time() - message.timestamp
                    metric.update_metrics(
                        transmission_delay,
                        queue_delay,
                        self.protocol.network_load
                    )
                    
        except Exception as e:
            logger.error(f"Error processing messages: {str(e)}")