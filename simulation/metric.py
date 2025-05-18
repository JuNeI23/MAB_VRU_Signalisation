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