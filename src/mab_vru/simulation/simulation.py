"""
Module for simulating V2V and V2I communications in a VRU signalization system.
Simplified version that maintains core functionality.
"""
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Union
import pandas as pd
import numpy as np
import logging
from pathlib import Path

from mab_vru.simulation.models import User, Infrastructure, Node
from mab_vru.simulation.protocols import Protocol
from mab_vru.simulation.metric import Metric
from mab_vru.MAB.MAB_u import UCBMAB  # Import MAB algorithm

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
    trace_file: str = 'sumoTraceCroisement.csv'

    def __post_init__(self):
        """Initialize default values after dataclass initialization."""
        if self.infra_positions is None:
            self.infra_positions = [(0.0, 0.0)]  # Default single infrastructure

def load_users(trace_file: str, v2v_protocol: Protocol) -> List[User]:
    """Load users (vehicles and pedestrians) from SUMO trace file."""
    try:
        df = pd.read_csv(trace_file)
        users = []
        
        # Load pedestrians
        if 'person/_id' in df.columns:
            for _, row in df.iterrows():
                try:
                    if pd.isna(row['person/_id']):
                        continue
                    users.append(User(
                        usager_id=str(row['person/_id']),
                        x=float(row['person/_x']),
                        y=float(row['person/_y']),
                        angle=float(row['person/_angle']),
                        speed=float(row['person/_speed']),
                        position=float(row['person/_pos']),
                        lane=str(row['person/_edge']),
                        time=float(row['_time']),
                        usager_type=str(row['person/_type']),
                        categorie='pieton',
                        protocol=v2v_protocol
                    ))
                except (ValueError, KeyError) as e:
                    logger.debug(f"Skipping invalid pedestrian data: {e}")
                    continue
        
        # Load vehicles from vehicle/0 and vehicle/1
        for vehicle_num in [0, 1]:
            prefix = f'vehicle/{vehicle_num}/'
            id_col = f'{prefix}_id'
            
            if id_col in df.columns:
                for _, row in df.iterrows():
                    try:
                        if pd.isna(row[id_col]):
                            continue
                        users.append(User(
                            usager_id=str(row[id_col]),
                            x=float(row[f'{prefix}_x']),
                            y=float(row[f'{prefix}_y']),
                            angle=float(row[f'{prefix}_angle']),
                            speed=float(row[f'{prefix}_speed']),
                            position=float(row[f'{prefix}_pos']),
                            lane=str(row[f'{prefix}_lane']),
                            time=float(row['_time']),
                            usager_type=str(row[f'{prefix}_type']),
                            categorie='vehicule',
                            protocol=v2v_protocol
                        ))
                    except (ValueError, KeyError) as e:
                        logger.debug(f"Skipping invalid vehicle data: {e}")
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
    """Load infrastructure nodes from trace file."""
    try:
        df = pd.read_csv(trace_file)
        infras = []
        
        for _, row in df.iterrows():
            # Check for infrastructure data
            if 'infra/_id' in df.columns and not pd.isna(row['infra/_id']):
                try:
                    infras.append(Infrastructure(
                        id=str(row['infra/_id']),
                        protocol=v2i_protocol,
                        x=float(row['infra/_x']),
                        y=float(row['infra/_y']),
                        processing_capacity=100,  # Default value
                        time=float(row['_time'])
                    ))
                except (ValueError, KeyError) as e:
                    logger.debug(f"Skipping invalid infrastructure data: {e}")
                    continue
        
        # If no infrastructure found in trace, create default ones from config
        if not infras:
            logger.info("No infrastructure in trace file, using default positions")
            config = SimulationConfig()
            for i, (x, y) in enumerate(config.infra_positions):
                infras.append(Infrastructure(
                    id=f"infra_{i}",
                    protocol=v2i_protocol,
                    x=float(x),
                    y=float(y),
                    processing_capacity=int(config.infra_processing_capacity),
                    time=0.0
                ))
        
        return infras
        
    except FileNotFoundError:
        logger.error(f"Trace file not found: {trace_file}")
        return []
    except Exception as e:
        logger.error(f"Error loading infrastructure: {e}")
        return []

def run_timestep(users: List[User], infras: List[Infrastructure], time: float, mab_algorithm: Any) -> Dict:
    """Run simulation for one timestep with MAB-based protocol selection."""
    metrics = {
        'V2V': Metric(),
        'V2I': Metric()
    }
    
    # Get active users for this timestep
    active_users = [user for user in users if user.time == time]
    
    for user in active_users:
        # Use MAB to select protocol (0 = V2V, 1 = V2I)
        selected_arm = mab_algorithm.select_arm()
        selected_protocol = 'V2I' if selected_arm == 1 else 'V2V'
        
        if selected_protocol == 'V2I' and infras:
            # Try V2I communication
            best_infra = None
            min_distance = float('inf')
            
            # Find closest infrastructure
            for infra in infras:
                distance = user.distance_to(infra)
                if distance < min_distance:
                    min_distance = distance
                    best_infra = infra
            
            if best_infra and user.in_range(best_infra):
                msg = user.create_message()
                success = best_infra.protocol.transmit_message(msg)
                
                # Calculate reward using the new reward function
                reward = best_infra.protocol.calculate_reward(
                    success=success,
                    distance=min_distance,
                    range_=best_infra.range,
                    delay=msg.delay
                )
                
                # Update MAB with reward (1 for V2I)
                mab_algorithm.update(1, reward)
                
                # Update metrics
                metrics['V2I'].update(msg, success, min_distance, best_infra.range)
                continue
        
        # V2V communication (either selected by MAB or fallback)
        best_peer = None
        min_distance = float('inf')
        
        # Find closest peer
        for other in users:
            if user != other:
                distance = user.distance_to(other)
                if distance < min_distance:
                    min_distance = distance
                    best_peer = other
        
        if best_peer and user.in_range(best_peer):
            msg = user.create_message()
            success = user.protocol.transmit_message(msg)
            
            # Calculate reward
            reward = user.protocol.calculate_reward(
                success=success,
                distance=min_distance,
                range_=user.range,
                delay=msg.delay
            )
            
            # Update MAB with reward (0 for V2V)
            if selected_protocol == 'V2V':
                mab_algorithm.update(0, reward)
            
            # Update metrics
            metrics['V2V'].update(msg, success, min_distance, user.range)
    
    # Prepare results with range information
    results = {}
    for protocol, metric in metrics.items():
        if metric.message_count > 0:
            # Map protocol name to arm index (V2V = 0, V2I = 1)
            arm_index = 1 if protocol == 'V2I' else 0
            selection_rate = mab_algorithm.get_selection_rate(arm_index)
            
            results[protocol] = {
                'Time': time,
                'Protocol': protocol,
                'Average Delay (s)': metric.average_delay,
                'Loss Rate (%)': metric.loss_rate * 100,
                'Average Load': metric.average_load,
                'Average Distance': metric.average_distance,
                'Reachability Rate (%)': metric.reachability_rate * 100,
                'MAB Selection Rate (%)': selection_rate * 100
            }
    
    return results

def main(config: Optional[SimulationConfig] = None) -> bool:
    """Run the main simulation with MAB-based protocol selection."""
    if config is None:
        config = SimulationConfig()
        
    try:
        # Setup protocols
        v2v_protocol = Protocol("V2V", config.v2v_network_load, config.v2v_packet_loss, config.v2v_transmission_time)
        v2i_protocol = Protocol("V2I", config.v2i_network_load, config.v2i_packet_loss, config.v2i_transmission_time)
        
        # Initialize MAB algorithm
        mab = UCBMAB(n_arms=2)  # 2 arms: V2V and V2I
        
        # Load users with V2V protocol as default
        users = load_users(config.trace_file, v2v_protocol)
        if not users:
            logger.error("No users loaded")
            return False
            
        # Load infrastructure for V2I
        infras = load_infrastructure(config.trace_file, v2i_protocol)
        if not infras:
            logger.warning("No infrastructure nodes found, V2V-only mode")
        
        # Run simulation with MAB protocol selection
        results = []
        simulation_times = sorted(set(user.time for user in users))
        
        for time in simulation_times:
            timestep_results = run_timestep(users, infras, time, mab)
            for protocol_results in timestep_results.values():
                results.append(protocol_results)
        
        # Save results to CSV
        df = pd.DataFrame(results)
        df.to_csv(config.csv_output, index=False)
        logger.info(f"Results saved to {config.csv_output}")
        
        # Also save protocol-specific results
        for protocol in ['V2V', 'V2I']:
            protocol_results = [r for r in results if r['Protocol'] == protocol]
            if protocol_results:
                output_file = config.csv_output.replace('.csv', f'_{protocol.lower()}.csv')
                pd.DataFrame(protocol_results).to_csv(output_file, index=False)
                logger.info(f"{protocol} results saved to {output_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        return False
