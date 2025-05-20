"""
Metrics collection and analysis for VRU communication simulation.
"""
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import numpy as np
import logging

logger = logging.getLogger(__name__)

@dataclass
class Metric:
    """
    Enhanced metrics collection with statistical analysis.
    
    Attributes:
        total_transmission_delay: Sum of all transmission delays
        total_queue_delay: Sum of all queueing delays
        total_messages: Count of successful transmissions
        lost_messages: Count of failed transmissions
        total_network_load: Sum of network loads during transmissions
        transmission_delays: List of all transmission delays
        queue_delays: List of all queueing delays
        network_loads: List of all network loads
    """
    total_transmission_delay: float = 0.0
    total_queue_delay: float = 0.0
    total_messages: int = 0
    lost_messages: int = 0
    total_network_load: float = 0.0
    transmission_delays: List[float] = field(default_factory=list)
    queue_delays: List[float] = field(default_factory=list)
    network_loads: List[float] = field(default_factory=list)
    
    def update_metrics(
        self,
        transmission_delay: Optional[float],
        queue_delay: float,
        network_load: float
    ) -> None:
        """
        Update metrics with new measurement.
        
        Args:
            transmission_delay: Delay in transmission, None if failed
            queue_delay: Time spent in queue
            network_load: Current network load
        """
        try:
            if transmission_delay is not None:
                self.total_transmission_delay += transmission_delay
                self.total_queue_delay += queue_delay
                self.total_messages += 1
                self.total_network_load += network_load
                
                # Store individual measurements
                self.transmission_delays.append(transmission_delay)
                self.queue_delays.append(queue_delay)
                self.network_loads.append(network_load)
            else:
                self.lost_messages += 1
                
        except Exception as e:
            logger.error(f"Error updating metrics: {str(e)}")
    
    def get_metrics(self) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Calculate final metrics with statistical analysis.
        
        Returns:
            Tuple containing:
            - Average delay (transmission + queue)
            - Packet loss rate
            - Average network load
        """
        try:
            # Calculate average delay
            average_delay = None
            if self.total_messages > 0:
                average_delay = (
                    self.total_transmission_delay + self.total_queue_delay
                ) / self.total_messages
            
            # Calculate packet loss rate
            packet_loss_rate = None
            total = self.total_messages + self.lost_messages
            if total > 0:
                packet_loss_rate = self.lost_messages / total
            
            # Calculate average network load
            average_network_load = None
            if self.total_messages > 0:
                average_network_load = self.total_network_load / self.total_messages
                
            return average_delay, packet_loss_rate, average_network_load
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return None, None, None
    
    def get_statistics(self) -> dict:
        """
        Get detailed statistics of the measurements.
        
        Returns:
            Dictionary containing various statistical measures
        """
        try:
            stats = {}
            
            if self.transmission_delays:
                stats['transmission_delay'] = {
                    'mean': np.mean(self.transmission_delays),
                    'std': np.std(self.transmission_delays),
                    'min': np.min(self.transmission_delays),
                    'max': np.max(self.transmission_delays),
                    'median': np.median(self.transmission_delays)
                }
                
            if self.queue_delays:
                stats['queue_delay'] = {
                    'mean': np.mean(self.queue_delays),
                    'std': np.std(self.queue_delays),
                    'min': np.min(self.queue_delays),
                    'max': np.max(self.queue_delays),
                    'median': np.median(self.queue_delays)
                }
                
            if self.network_loads:
                stats['network_load'] = {
                    'mean': np.mean(self.network_loads),
                    'std': np.std(self.network_loads),
                    'min': np.min(self.network_loads),
                    'max': np.max(self.network_loads),
                    'median': np.median(self.network_loads)
                }
                
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating statistics: {str(e)}")
            return {}