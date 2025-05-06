import random
import time
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class Protocole:
    name: str
    network_load: float
    packet_loss_rate: float
    transmission_time: float
    transmission_success_rate: float = 0.9

    def transmit_message(self, sender, message, receiver):
        if random.random() < self.transmission_success_rate:
            distance = sender.distance_to(receiver)
            if distance > sender.range:
                logger.warning(f"{self.name}: message distance {distance:.2f} > range {sender.range}")
                return None
            delay = self.transmission_time + 0.01 * distance
            time.sleep(delay)
            self.update_network_load()
            return delay
        else:
            self.update_network_load()
            return None

    def update_network_load(self) -> None:
        self.network_load = random.random()
        logger.debug(f"{self.name}: network_load updated to {self.network_load:.2f}")

class Metric:
    def __init__(self):
        self.total_transmission_delay = 0.0
        self.total_queue_delay = 0.0
        self.total_messages = 0
        self.lost_messages = 0
        self.total_network_load = 0.0

    def update_metrics(self, transmission_delay, queue_delay, network_load):
        if transmission_delay is not None:
            self.total_transmission_delay += transmission_delay
            self.total_queue_delay += queue_delay
            self.total_messages += 1
            self.total_network_load += network_load
        else:
            self.lost_messages += 1

    def get_metrics(self):
        average_delay = None
        if self.total_messages > 0:
            average_delay = (self.total_transmission_delay + self.total_queue_delay) / self.total_messages
        packet_loss_rate = None
        total = self.total_messages + self.lost_messages
        if total > 0:
            packet_loss_rate = self.lost_messages / total
        average_network_load = None
        if self.total_messages > 0:
            average_network_load = self.total_network_load / self.total_messages
        return average_delay, packet_loss_rate, average_network_load