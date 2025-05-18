import random
import time
import logging
from dataclasses import dataclass

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
