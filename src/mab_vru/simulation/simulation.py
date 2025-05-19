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
    users = []
    
    try:
        df = pd.read_csv(trace_file)
        
        # Process vehicles
        vehicle_prefix = "vehicle/0/_"  # Start with first vehicle
        vehicle_cols = {
            'id': f'{vehicle_prefix}id',
            'x': f'{vehicle_prefix}x',
            'y': f'{vehicle_prefix}y',
            'angle': f'{vehicle_prefix}angle',
            'speed': f'{vehicle_prefix}speed',
            'pos': f'{vehicle_prefix}pos',
            'lane': f'{vehicle_prefix}lane',
            'type': f'{vehicle_prefix}type'
        }
        
        # Process pedestrians
        person_prefix = "person/_"
        person_cols = {
            'id': f'{person_prefix}id',
            'x': f'{person_prefix}x',
            'y': f'{person_prefix}y',
            'angle': f'{person_prefix}angle',
            'speed': f'{person_prefix}speed',
            'pos': f'{person_prefix}pos',
            'type': f'{person_prefix}type'
        }
        
        # Add time column
        df['time'] = df['_time']
        
        # Process vehicles first
        if all(col in df.columns for col in vehicle_cols.values()):
            for _, row in df.iterrows():
                try:
                    users.append(User(
                        usager_id=str(row[vehicle_cols['id']]),
                        x=float(row[vehicle_cols['x']]),
                        y=float(row[vehicle_cols['y']]),
                        angle=float(row[vehicle_cols['angle']]),
                        speed=float(row[vehicle_cols['speed']]),
                        position=float(row[vehicle_cols['pos']]),
                        lane=str(row[vehicle_cols['lane']]),
                        time=float(row['time']),
                        usager_type=str(row[vehicle_cols['type']]),
                        categorie='vehicule',
                        protocol=v2v_protocol
                    ))
                except (ValueError, KeyError) as e:
                    logger.warning(f"Skipping invalid vehicle data: {e}")
                    continue
        
        # Then process pedestrians
        if all(col in df.columns for col in person_cols.values()):
            for _, row in df.iterrows():
                try:
                    users.append(User(
                        usager_id=str(row[person_cols['id']]),
                        x=float(row[person_cols['x']]),
                        y=float(row[person_cols['y']]),
                        angle=float(row[person_cols['angle']]),
                        speed=float(row[person_cols['speed']]),
                        position=float(row[person_cols['pos']]),
                        lane='pedestrian',  # Default lane for pedestrians
                        time=float(row['time']),
                        usager_type=str(row[person_cols['type']]),
                        categorie='pieton',
                        protocol=v2v_protocol
                    ))
                except (ValueError, KeyError) as e:
                    logger.warning(f"Skipping invalid pedestrian data: {e}")
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

def load_infrastructure(trace_file: str, v2i_protocol: Protocol) -> List[Infrastructure]:
    """Load infrastructure nodes from SUMO trace file.
    
    Args:
        trace_file: Path to the SUMO trace CSV file
        v2i_protocol: Protocol to use for V2I communication
        
    Returns:
        List of Infrastructure objects
    """
    infras = []
    
    try:
        df = pd.read_csv(trace_file)
        
        # Process infrastructure (containers with type 'infra')
        container_prefix = "container/_"
        container_cols = {
            'id': f'{container_prefix}id',
            'x': f'{container_prefix}x',
            'y': f'{container_prefix}y',
            'type': f'{container_prefix}type',
            'edge': f'{container_prefix}edge'
        }
        
        # Filter for infrastructure nodes
        infra_data = df[df[container_cols['type']] == 'infra'].drop_duplicates(subset=[container_cols['id']])
        
        for _, row in infra_data.iterrows():
            try:
                infras.append(Infrastructure(
                    id=str(row[container_cols['id']]),
                    protocol=v2i_protocol,
                    x=float(row[container_cols['x']]),
                    y=float(row[container_cols['y']]),
                    processing_capacity=100,  # Default capacity
                    time=0.0  # Infrastructure is always available
                ))
            except (ValueError, KeyError) as e:
                logger.warning(f"Skipping invalid infrastructure data: {e}")
                continue
        
        if not infras:
            logger.warning("No infrastructure nodes found in trace file")
            # Add default infrastructure at origin
            infras.append(Infrastructure(
                "infra_default",
                v2i_protocol,
                x=0.0,
                y=0.0,
                processing_capacity=100.0,
                time=0.0
            ))
            
        return infras
        
    except Exception as e:
        logger.error(f"Error loading infrastructure data: {e}")
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
        
        # Load infrastructure
        infras = load_infrastructure(config.trace_file, v2i_protocol)
        
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
