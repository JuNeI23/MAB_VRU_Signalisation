"""
Module for simulating V2V and V2I communications in a VRU signalization system.
Simplified version that maintains core functionality.
"""
from dataclasses import dataclass
from typing import Optional, List, Dict
import pandas as pd
import numpy as np
import logging
from pathlib import Path

from mab_vru.simulation.models import User, Infrastructure, Node
from mab_vru.simulation.protocols import Protocol
from mab_vru.simulation.metric import Metric

logger = logging.getLogger(__name__)

@dataclass
class SimulationConfig:
    """Configuration for simulation parameters."""
    # Network parameters
    v2v_network_load: float = 0.1
    v2v_packet_loss: float = 0.1
    v2v_transmission_time: float = 0.1
    v2i_network_load: float = 0.1
    v2i_packet_loss: float = 0.05
    v2i_transmission_time: float = 0.5
    
    # Infrastructure parameters
    infra_positions: List[tuple[float, float]] = None
    infra_processing_capacity: float = 100.0
    
    # File paths
    csv_output: str = 'resultats.csv'
    trace_file: str = 'sumoTrace_edge.csv'

    def __post_init__(self):
        """Initialize default values after dataclass initialization."""
        if self.infra_positions is None:
            self.infra_positions = [(0.0, 0.0)]  # Default single infrastructure

def load_users(trace_file: str, v2v_protocol: Protocol) -> List[User]:
    """Load users from SUMO trace file.
    
    Args:
        trace_file: Path to the SUMO trace CSV file
        v2v_protocol: Protocol to use for V2V communication
        
    Returns:
        List of User objects
        
    Raises:
        ValueError: If required columns are missing
        FileNotFoundError: If trace file doesn't exist
    """
    required_columns = {'id', 'x', 'y', 'angle', 'speed', 'pos', 'lane', 'time', 'type'}
    users = []
    
    try:
        df = pd.read_csv(trace_file)
        missing_cols = required_columns - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        for _, row in df.iterrows():
            try:
                users.append(User(
                    usager_id=str(row['id']),
                    x=float(row['x']),
                    y=float(row['y']),
                    angle=float(row['angle']),
                    speed=float(row['speed']),
                    position=float(row['pos']),
                    lane=str(row['lane']),
                    time=float(row['time']),
                    usager_type=str(row['type']),
                    categorie='vehicule',
                    protocol=v2v_protocol
                ))
            except (ValueError, KeyError) as e:
                logger.warning(f"Skipping invalid user data: {e}")
                continue
                
        if not users:
            logger.warning("No valid users loaded from trace file")
            
        return users
        
    except FileNotFoundError:
        logger.error(f"Trace file not found: {trace_file}")
        raise
    except Exception as e:
        logger.error(f"Error loading trace file: {e}")
        raise

def run_timestep(users: List[User], infras: List[Infrastructure], time: float) -> Dict:
    """Run simulation for one timestep."""
    metrics = {
        'V2V': Metric(),
        'V2I': Metric()
    }
    
    for user in users:
        if user.time != time:
            continue
            
        # Try V2V communication
        for other in users:
            if user != other and user.in_range(other):
                msg = user.create_message()
                success = user.protocol.transmit_message(msg)
                metrics['V2V'].update(msg, success)
                
        # Try V2I communication
        for infra in infras:
            if user.in_range(infra):
                msg = user.create_message()
                success = infra.protocol.transmit_message(msg)
                metrics['V2I'].update(msg, success)
    
    return {
        protocol: {
            'Time': time,
            'Protocol': protocol,
            'Average Delay (s)': metric.average_delay,
            'Loss Rate (%)': metric.loss_rate * 100,
            'Average Load': metric.average_load
        }
        for protocol, metric in metrics.items()
    }

def main(config: Optional[SimulationConfig] = None) -> bool:
    """Run the main simulation."""
    if config is None:
        config = SimulationConfig()
        
    try:
        # Setup protocols
        v2v_protocol = Protocol("V2V", config.v2v_network_load, config.v2v_packet_loss, config.v2v_transmission_time)
        v2i_protocol = Protocol("V2I", config.v2i_network_load, config.v2i_packet_loss, config.v2i_transmission_time)
        
        # Load users with V2V protocol
        users = load_users(config.trace_file, v2v_protocol)
        if not users:
            logger.error("No users loaded")
            return False
        
        infras = [
            Infrastructure("infra_1", v2i_protocol, x=0, y=0, processing_capacity=100, time=0)
        ]
        
        # Run simulation
        results = []
        times = sorted(set(user.time for user in users))
        
        for time in times:
            timestep_results = run_timestep(users, infras, time)
            results.extend(timestep_results.values())
        
        # Save results
        pd.DataFrame(results).to_csv(config.csv_output, index=False)
        return True
        
    except Exception as e:
        logger.error(f"Simulation failed: {str(e)}")
        return False
