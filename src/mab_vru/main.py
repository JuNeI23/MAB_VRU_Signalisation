"""
Module for simulating V2V and V2I communications in a VRU signalization system.
Simplified version that maintains core functionality.
"""
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Union, Tuple, TypeVar
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
import sys
import math

from simulation.models import User, Infrastructure, Node
from simulation.protocols import Protocol
from simulation.metric import Metric
from simulation.analytics import analyze_simulation_results
from simulation.spatial_index import SpatialIndex
from MAB.MAB_u import UCBMAB  # Import MAB algorithm
from MAB.MAB_Ts import ThompsonSamplingMAB # Import MAB algorithm
from MAB.MAB_e import EpsilonGreedyMAB  # Import MAB algorithm

# Protocol arm indices
V2V_ARM_INDEX = 0
V2I_ARM_INDEX = 1

# Default values for infrastructure
DEFAULT_INFRA_PROCESSING_CAPACITY = 100.0

# Type alias for MAB algorithms
MABAlgorithm = Union[UCBMAB, EpsilonGreedyMAB, ThompsonSamplingMAB]

def setup_logging():
    """Configure logging with both file and console handlers."""
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"simulation_{timestamp}.log"
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # File handler (DEBUG level)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    
    # Console handler (INFO level)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging initialized. Log file: {log_file}")

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
    
    # MAB algorithm configuration
    mab_algorithms: List[str] = None  # List of MAB algorithms to use: 'ucb', 'epsilon-greedy', 'thompson'
    epsilon_value: float = 0.1  # For epsilon-greedy algorithm
    
    # File paths
    csv_output: str = 'results/resultats.csv'
    trace_file: str = 'sumoTraceCroisement.csv'

    def __post_init__(self):
        """Initialize default values after dataclass initialization and validate parameters."""
        if self.infra_positions is None:
            self.infra_positions = [(0.0, 0.0)]  # Default single infrastructure
        if self.mab_algorithms is None:
            # Use all available MAB algorithms by default
            self.mab_algorithms = ['ucb', 'epsilon-greedy', 'thompson']
        
        # Validate network parameters
        if not 0 <= self.v2v_packet_loss <= 1:
            raise ValueError("v2v_packet_loss must be between 0 and 1")
        if not 0 <= self.v2i_packet_loss <= 1:
            raise ValueError("v2i_packet_loss must be between 0 and 1")
        if not 0 <= self.v2v_network_load <= 1:
            raise ValueError("v2v_network_load must be between 0 and 1")
        if not 0 <= self.v2i_network_load <= 1:
            raise ValueError("v2i_network_load must be between 0 and 1")
        if self.v2v_transmission_time < 0:
            raise ValueError("v2v_transmission_time must be positive")
        if self.v2i_transmission_time < 0:
            raise ValueError("v2i_transmission_time must be positive")
        if self.infra_processing_capacity <= 0:
            raise ValueError("infra_processing_capacity must be positive")
        
        # Validate MAB parameters
        if not 0 <= self.epsilon_value <= 1:
            raise ValueError("epsilon_value must be between 0 and 1")
        
        # Validate algorithm names
        valid_algorithms = ['ucb', 'epsilon-greedy', 'thompson']
        for algo in self.mab_algorithms:
            if algo not in valid_algorithms:
                raise ValueError(f"Unknown MAB algorithm: {algo}. Valid options: {valid_algorithms}")

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
            # Check for infrastructure data from both infra and container
            if 'infra/_id' in df.columns and not pd.isna(row['infra/_id']):
                try:
                    infras.append(Infrastructure(
                        id=str(row['infra/_id']),
                        protocol=v2i_protocol,
                        x=float(row['infra/_x']),
                        y=float(row['infra/_y']),
                        processing_capacity=DEFAULT_INFRA_PROCESSING_CAPACITY,
                        time=float(row['_time'])
                    ))
                except (ValueError, KeyError) as e:
                    logger.debug(f"Skipping invalid infrastructure data: {e}")
                    continue
            
            # Also check for container data and treat as infrastructure
            if 'container/_id' in df.columns and not pd.isna(row['container/_id']):
                try:
                    infras.append(Infrastructure(
                        id=str(row['container/_id']),
                        protocol=v2i_protocol,
                        x=float(row['container/_x']),
                        y=float(row['container/_y']),
                        processing_capacity=DEFAULT_INFRA_PROCESSING_CAPACITY,
                        time=float(row['_time'])
                    ))
                except (ValueError, KeyError) as e:
                    logger.debug(f"Skipping invalid container data: {e}")
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

def find_best_infrastructure(user: User, infras: List[Infrastructure]) -> Tuple[Optional[Infrastructure], float]:
    """Find the closest infrastructure to a user."""
    best_infra = None
    min_distance = float('inf')
    
    for infra in infras:
        distance = user.distance_to(infra)
        if distance < min_distance:
            min_distance = distance
            best_infra = infra
    
    return best_infra, min_distance

def find_best_peer_optimized(user: User, users: List[User]) -> Tuple[Optional[User], float]:
    """Find the closest peer to a user using a spatial index for better performance."""
    spatial_index = SpatialIndex()
    
    # Add all users to the spatial index
    for other_user in users:
        if other_user != user:  # Don't add the current user
            spatial_index.add_node(other_user, other_user.x, other_user.y)
    
    # Find the nearest neighbor
    nearest, distance = spatial_index.nearest_neighbor(user.x, user.y)
    return nearest, distance

def process_v2i_communication(user: User, infra: Infrastructure, distance: float, 
                             mab_algorithm: MABAlgorithm) -> Tuple[Dict, bool]:
    """Process a V2I communication and return metrics and success."""
    logger.debug(f"User {user.usager_id} attempting V2I communication with {infra.id} at distance {distance:.2f}")
    msg = user.create_message()
    success = infra.protocol.transmit_message(msg)
    
    # Calculate reward
    reward = infra.protocol.calculate_reward(
        success=success,
        distance=distance,
        range_=infra.range,
        delay=msg.delay
    )
    
    # Update MAB with reward for V2I
    mab_algorithm.update(V2I_ARM_INDEX, reward)
    logger.debug(f"V2I communication {'successful' if success else 'failed'}, reward: {reward:.3f}")
    
    # Return metrics
    metrics = {
        'message': msg,
        'success': success,
        'distance': distance,
        'range': infra.range
    }
    
    return metrics, success

def process_v2v_communication(user: User, peer: User, distance: float, 
                             mab_algorithm: MABAlgorithm, selected_protocol: str) -> Tuple[Dict, bool]:
    """Process a V2V communication and return metrics and success."""
    logger.debug(f"User {user.usager_id} attempting V2V communication with {peer.usager_id} at distance {distance:.2f}")
    msg = user.create_message()
    success = user.protocol.transmit_message(msg)
    
    # Calculate reward
    reward = user.protocol.calculate_reward(
        success=success,
        distance=distance,
        range_=user.range,
        delay=msg.delay
    )
    
    # Update MAB with reward for V2V
    if selected_protocol == 'V2V':
        mab_algorithm.update(V2V_ARM_INDEX, reward)
        logger.debug(f"V2V communication {'successful' if success else 'failed'}, reward: {reward:.3f}")
    
    # Return metrics
    metrics = {
        'message': msg,
        'success': success,
        'distance': distance,
        'range': user.range
    }
    
    return metrics, success

def run_timestep(users: List[User], infras: List[Infrastructure], time: float, 
                mab_algorithm: MABAlgorithm) -> Dict[str, Dict[str, float]]:
    """Run simulation for one timestep with MAB-based protocol selection."""
    logger.debug(f"Starting timestep simulation at t={time}")
    
    metrics = {
        'V2V': Metric(),
        'V2I': Metric()
    }
    
    # Get active users for this timestep
    active_users = [user for user in users if user.time == time]
    logger.debug(f"Found {len(active_users)} active users at t={time}")
    
    # Track successful communications
    v2v_successes = 0
    v2i_successes = 0
    total_communications = 0
    
    for user in active_users:
        logger.debug(f"Processing user {user.usager_id} (type: {user.usager_type})")
        total_communications += 1
        
        # Use MAB to select protocol (0 = V2V, 1 = V2I)
        selected_arm = mab_algorithm.select_arm()
        selected_protocol = 'V2I' if selected_arm == V2I_ARM_INDEX else 'V2V'
        logger.debug(f"MAB selected protocol: {selected_protocol} for user {user.usager_id}")
        
        if selected_protocol == 'V2I' and infras:
            # Try V2I communication
            best_infra, min_distance = find_best_infrastructure(user, infras)
            
            if best_infra and user.in_range(best_infra):
                v2i_metrics, success = process_v2i_communication(
                    user, best_infra, min_distance, mab_algorithm
                )
                metrics['V2I'].update(
                    v2i_metrics['message'], 
                    v2i_metrics['success'], 
                    v2i_metrics['distance'], 
                    v2i_metrics['range']
                )
                if success:
                    v2i_successes += 1
                continue
            else:
                logger.debug(f"No infrastructure in range for user {user.usager_id}, falling back to V2V")
        
        # V2V communication (either selected by MAB or fallback)
        best_peer, min_distance = find_best_peer_optimized(user, users)
        
        if best_peer and user.in_range(best_peer):
            v2v_metrics, success = process_v2v_communication(
                user, best_peer, min_distance, mab_algorithm, selected_protocol
            )
            metrics['V2V'].update(
                v2v_metrics['message'], 
                v2v_metrics['success'], 
                v2v_metrics['distance'], 
                v2v_metrics['range']
            )
            if success:
                v2v_successes += 1
        else:
            logger.debug(f"No peers in range for user {user.usager_id}")
    
    # Prepare results with range information
    results = {}
    for protocol, metric in metrics.items():
        if metric.message_count > 0:
            # Map protocol name to arm index (V2V = 0, V2I = 1)
            arm_index = V2I_ARM_INDEX if protocol == 'V2I' else V2V_ARM_INDEX
            selection_rate = mab_algorithm.get_selection_rate(arm_index)
            
            results[protocol] = {
                'Time': time,
                'Protocol': protocol,
                'Average Delay (s)': metric.average_delay,
                'Loss Rate (%)': metric.loss_rate * 100,
                'Average Load': metric.average_load,
                'Average Distance': metric.average_distance,
                'Reachability Rate (%)': metric.reachability_rate * 100,
                'MAB Selection Rate (%)': selection_rate * 100,
                'Success Count': v2v_successes if protocol == 'V2V' else v2i_successes,
                'Total Communications': total_communications
            }
    
    return results

def main(config: Optional[SimulationConfig] = None) -> bool:
    """Run the main simulation with MAB-based protocol selection."""
    if config is None:
        config = SimulationConfig()
    
    # Create results directory if it doesn't exist
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
        
    try:
        # Setup protocols
        v2v_protocol = Protocol("V2V", config.v2v_network_load, config.v2v_packet_loss, config.v2v_transmission_time)
        v2i_protocol = Protocol("V2I", config.v2i_network_load, config.v2i_packet_loss, config.v2i_transmission_time)
        logger.info("Protocols initialized with parameters:")
        logger.info(f"V2V: load={config.v2v_network_load}, loss={config.v2v_packet_loss}, delay={config.v2v_transmission_time}")
        logger.info(f"V2I: load={config.v2i_network_load}, loss={config.v2i_packet_loss}, delay={config.v2i_transmission_time}")
        
        # Map algorithm names to their classes
        mab_map = {
            'ucb': lambda: UCBMAB(n_arms=2),
            'epsilon-greedy': lambda: EpsilonGreedyMAB(n_arms=2, epsilon=config.epsilon_value),
            'thompson': lambda: ThompsonSamplingMAB(n_arms=2)
        }
        
        success = True
        # Run simulation for each specified MAB algorithm
        total_algorithms = len(config.mab_algorithms)
        logger.info(f"Starting simulation with {total_algorithms} MAB algorithms")
        
        for algorithm_idx, algorithm in enumerate(config.mab_algorithms, 1):
            if algorithm not in mab_map:
                logger.warning(f"Unknown MAB algorithm: {algorithm}, skipping...")
                continue
                
            logger.info(f"\n[{algorithm_idx}/{total_algorithms}] Running simulation with {algorithm} algorithm...")
            start_time = datetime.now()
            
            # Initialize MAB algorithm
            mab = mab_map[algorithm]()
            logger.debug(f"Initialized {algorithm} algorithm: {mab.__class__.__name__}")
            
            # Load users with V2V protocol as default
            users = load_users(config.trace_file, v2v_protocol)
            if not users:
                logger.error("No users loaded")
                continue
            logger.info(f"Loaded {len(users)} users from trace file")
                
            # Load infrastructure for V2I
            infras = load_infrastructure(config.trace_file, v2i_protocol)
            if not infras:
                logger.warning("No infrastructure nodes found, V2V-only mode")
            else:
                logger.info(f"Loaded {len(infras)} infrastructure nodes")
            
            # Run simulation with MAB protocol selection
            results = []
            simulation_times = sorted(set(user.time for user in users))
            logger.info(f"Running simulation for {len(simulation_times)} timesteps")
            
            # Track protocol selections and performance metrics
            v2v_selections = 0
            v2i_selections = 0
            v2v_successes = 0
            v2i_successes = 0
            total_communications = 0
            
            for time in simulation_times:
                timestep_results = run_timestep(users, infras, time, mab)
                for protocol_results in timestep_results.values():
                    results.append(protocol_results)
                    # Count selections
                    if protocol_results['Protocol'] == 'V2V':
                        v2v_selections += 1
                    else:
                        v2i_selections += 1
            
            # Log detailed performance statistics
            total_selections = v2v_selections + v2i_selections
            if total_selections > 0:
                v2v_percentage = (v2v_selections / total_selections) * 100
                v2i_percentage = (v2i_selections / total_selections) * 100
                most_picked = "V2V" if v2v_selections >= v2i_selections else "V2I"
                
                # Calculate execution time
                execution_time = datetime.now() - start_time
                
                logger.info(f"\n{algorithm.upper()} Performance Statistics:")
                logger.info("Protocol Selection:")
                logger.info(f"  Most picked protocol: {most_picked}")
                logger.info(f"  V2V selections: {v2v_selections} ({v2v_percentage:.1f}%)")
                logger.info(f"  V2I selections: {v2i_selections} ({v2i_percentage:.1f}%)")
                
                logger.info("\nExecution Statistics:")
                logger.info(f"  Total timesteps: {len(simulation_times)}")
                logger.info(f"  Total communications: {total_communications}")
                logger.info(f"  Execution time: {execution_time}")
                logger.info(f"  Average time per step: {execution_time/len(simulation_times)}")
            
            # Save results to CSV with algorithm name in filename
            base_name = Path(config.csv_output).stem
            ext = Path(config.csv_output).suffix
            output_path = str(Path(config.csv_output).with_name(f"{base_name}_{algorithm}{ext}"))
            
            df = pd.DataFrame(results)
            df.to_csv(output_path, index=False)
            logger.info(f"Results saved to {output_path}")
            
            # Also save protocol-specific results
            for protocol in ['V2V', 'V2I']:
                protocol_results = [r for r in results if r['Protocol'] == protocol]
                if protocol_results:
                    protocol_output = output_path.replace(ext, f"_{protocol.lower()}{ext}")
                    pd.DataFrame(protocol_results).to_csv(protocol_output, index=False)
                    logger.info(f"{protocol} results saved to {protocol_output}")
            
            # Analyze and log the results
            logger.info("Analyzing results...")
            try:
                analysis_results = analyze_simulation_results(Path(output_path))
                for key, value in analysis_results.items():
                    logger.info(f"{key}: {value}")
            except Exception as e:
                logger.error(f"Error during results analysis: {e}")
        
        return success
        
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        return False

if __name__ == "__main__":
    setup_logging()
    config = SimulationConfig()
    success = main(config)
    if success:
        logger.info("Simulation completed successfully")
    else:
        logger.error("Simulation failed")
