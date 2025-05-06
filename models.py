import queue
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict

# Avoid circular imports when type‐checking
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from protocols import Protocole, Metric

logger = logging.getLogger(__name__)

@dataclass(order=True)
class Message:
    priority: int
    sender_id: str = field(compare=False)
    receiver_id: str = field(compare=False)
    size: float = field(compare=False)
    timestamp: float = field(default_factory=time.time, compare=False)

class Node:
    def __init__(
        self,
        x: float,
        y: float,
        range_: float,
        priority: int,
        processing_capacity: int,
        user_id: Optional[str] = None,
    ):
        self.x = x
        self.y = y
        self.range = range_
        self.priority = priority
        self.processing_capacity = processing_capacity
        self.user_id = user_id
        self.queue = queue.PriorityQueue()
        self.protocol: Optional['Protocole'] = None

    def distance_to(self, other: 'Node') -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

    def within_range(self, other: 'Node') -> bool:
        return self.distance_to(other) <= self.range

    def send_message(self, receiver: 'Node', size: float) -> None:
        distance = self.distance_to(receiver)
        if size <= self.processing_capacity and self.within_range(receiver):
            message = Message(self.priority, self.user_id, receiver.user_id, size)
            self.queue.put(message)
            logger.info(f"{self.user_id} → {receiver.user_id} : message ready (distance: {distance:.2f})")
        else:
            logger.error(
                f"{self.user_id} → {getattr(receiver, 'user_id', None)} : failed to send message "
                f"(distance: {distance:.2f}, range: {self.range}, size: {size})"
            )

    def process_queue(self, users: Dict[str, 'Node'], metric: 'Metric') -> None:
        messages_by_receiver: Dict['Node', list[Message]] = {}
        while not self.queue.empty():
            message: Message = self.queue.get()
            receiver = users.get(message.receiver_id)
            if receiver:
                messages_by_receiver.setdefault(receiver, []).append(message)
        for receiver, messages in messages_by_receiver.items():
            for message in messages:
                if receiver.protocol is None:
                    receiver.protocol = self.protocol
                transmission_delay = self.protocol.transmit_message(self, message, receiver)
                queue_delay = time.time() - message.timestamp
                metric.update_metrics(transmission_delay, queue_delay, self.protocol.network_load)

class User(Node):
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
        if categorie == "vru":
            priority = 1
            range_ = 90
            processing_capacity = 1
        elif categorie == "vehicule":
            priority = 2
            range_ = 120
            processing_capacity = 2
        else:
            raise ValueError("Invalid category. Must be 'vru' or 'vehicule'.")
        super().__init__(x, y, range_, priority, processing_capacity, user_id=usager_id)
        self.angle = angle
        self.speed = speed
        self.position = position
        self.lane = lane
        self.usager_type = usager_type
        self.time = time
        self.categorie = categorie
        self.queue_size = 10 if categorie == "vru" else 50

class Infrastructure(Node):
    def __init__(
        self,
        id: str,
        protocol: 'Protocole',
        x: float,
        y: float,
        processing_capacity: int,
        time: float,
    ):
        super().__init__(x, y, range_=300, priority=1, processing_capacity=processing_capacity, user_id=id)
        self.protocol = protocol
        self.time = time
        self.user_type = "Infrastructure"