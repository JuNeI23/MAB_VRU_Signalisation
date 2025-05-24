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
import csv
import psutil
import time
from contextlib import contextmanager
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

from mab_vru.simulation.models import User, Infrastructure, Node
from mab_vru.simulation.protocols import Protocol
from mab_vru.simulation.metric import Metric
from mab_vru.simulation.analytics import analyze_simulation_results
from mab_vru.simulation.spatial_index import SpatialIndex
from mab_vru.MAB.MAB_u import UCBMAB  # Import MAB algorithm
from mab_vru.MAB.MAB_Ts import ThompsonSamplingMAB # Import MAB algorithm
from mab_vru.MAB.MAB_e import EpsilonGreedyMAB  # Import MAB algorithm

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
    """Configuration for simulation parameters with environment variable support."""
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
    
    @classmethod
    def from_environment(cls) -> 'SimulationConfig':
        """Create configuration from environment variables."""
        config = cls()
        
        # Load parameters from environment variables if they exist
        env_mappings = {
            'MAB_V2V_NETWORK_LOAD': ('v2v_network_load', float),
            'MAB_V2V_PACKET_LOSS': ('v2v_packet_loss', float),
            'MAB_V2V_TRANSMISSION_TIME': ('v2v_transmission_time', float),
            'MAB_V2I_NETWORK_LOAD': ('v2i_network_load', float),
            'MAB_V2I_PACKET_LOSS': ('v2i_packet_loss', float),
            'MAB_V2I_TRANSMISSION_TIME': ('v2i_transmission_time', float),
            'MAB_INFRA_PROCESSING_CAPACITY': ('infra_processing_capacity', float),
            'MAB_EPSILON_VALUE': ('epsilon_value', float),
            'MAB_CSV_OUTPUT': ('csv_output', str),
            'MAB_TRACE_FILE': ('trace_file', str),
        }
        
        for env_var, (attr_name, attr_type) in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                try:
                    if attr_type == float:
                        setattr(config, attr_name, float(env_value))
                    elif attr_type == str:
                        setattr(config, attr_name, env_value)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid environment variable {env_var}={env_value}: {e}")
        
        # Handle special cases
        mab_algorithms_env = os.getenv('MAB_ALGORITHMS')
        if mab_algorithms_env:
            try:
                config.mab_algorithms = [algo.strip() for algo in mab_algorithms_env.split(',')]
            except Exception as e:
                logger.warning(f"Invalid MAB_ALGORITHMS environment variable: {e}")
        
        infra_positions_env = os.getenv('MAB_INFRA_POSITIONS')
        if infra_positions_env:
            try:
                # Parse format: "x1,y1;x2,y2;x3,y3"
                positions = []
                for pos_str in infra_positions_env.split(';'):
                    x, y = map(float, pos_str.split(','))
                    positions.append((x, y))
                config.infra_positions = positions
            except Exception as e:
                logger.warning(f"Invalid MAB_INFRA_POSITIONS environment variable: {e}")
        
        return config

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
        
        # Validate algorithm names and filter out invalid ones
        valid_algorithms = ['ucb', 'epsilon-greedy', 'thompson']
        invalid_algorithms = [algo for algo in self.mab_algorithms if algo not in valid_algorithms]
        if invalid_algorithms:
            logger.warning(f"Invalid MAB algorithms removed: {invalid_algorithms}. Valid options: {valid_algorithms}")
            self.mab_algorithms = [algo for algo in self.mab_algorithms if algo in valid_algorithms]
        
        # Ensure at least one valid algorithm remains
        if not self.mab_algorithms:
            logger.warning("No valid MAB algorithms specified. Using default 'ucb'.")
            self.mab_algorithms = ['ucb']
        
        # Cross-parameter validation
        self._validate_cross_parameters()
    
    def _validate_cross_parameters(self):
        """Validate relationships between parameters."""
        # Check if V2V and V2I parameters are reasonable relative to each other
        if self.v2v_transmission_time > self.v2i_transmission_time * 5:
            logger.warning("V2V transmission time is significantly higher than V2I. This may not be realistic.")
        
        if self.v2v_packet_loss > self.v2i_packet_loss * 2:
            logger.warning("V2V packet loss is significantly higher than V2I. Consider adjusting parameters.")
        
        # Validate infrastructure positions
        for i, (x, y) in enumerate(self.infra_positions):
            if not (-10000 <= x <= 10000) or not (-10000 <= y <= 10000):
                raise ValueError(f"Infrastructure position {i} ({x}, {y}) is outside reasonable bounds")
        
        # Check for duplicate infrastructure positions
        if len(self.infra_positions) != len(set(self.infra_positions)):
            logger.warning("Duplicate infrastructure positions detected")
        
        # Validate file paths
        if not self.csv_output.endswith('.csv'):
            logger.warning("CSV output file should have .csv extension")
        
        if not self.trace_file.endswith('.csv'):
            raise ValueError("Trace file must be a CSV file")

def load_users(trace_file: str, v2v_protocol: Protocol) -> List[User]:
    """Load users (vehicles and pedestrians) from SUMO trace file with enhanced security."""
    try:
        # Validate file path
        file_path = validate_file_path(trace_file, must_exist=True)
        
        # Validate CSV content
        if not validate_csv_content(file_path):
            raise DataLoadingError(f"Invalid CSV content in {trace_file}")
        
        logger.info(f"Loading users from validated file: {file_path}")
        df = pd.read_csv(file_path)
        users = []
        
        # Load pedestrians - handle different numbering patterns
        person_prefixes = []
        for col in df.columns:
            if col.startswith('person/') and col.endswith('/_id'):
                prefix = col.replace('/_id', '')
                if prefix not in person_prefixes:
                    person_prefixes.append(prefix)
        
        for prefix in person_prefixes:
            id_col = f'{prefix}/_id'
            
            if id_col in df.columns:
                for _, row in df.iterrows():
                    try:
                        if pd.isna(row[id_col]):
                            continue
                        
                        # Validate data ranges
                        x_val = float(row[f'{prefix}/_x'])
                        y_val = float(row[f'{prefix}/_y'])
                        speed_val = float(row[f'{prefix}/_speed'])
                        
                        if not (-10000 <= x_val <= 10000) or not (-10000 <= y_val <= 10000):
                            logger.debug(f"Skipping pedestrian with out-of-range coordinates: x={x_val}, y={y_val}")
                            continue
                        
                        if not (0 <= speed_val <= 200):  # Max 200 m/s
                            logger.debug(f"Skipping pedestrian with unrealistic speed: {speed_val}")
                            continue
                        
                        users.append(User(
                            usager_id=str(row[id_col])[:50],  # Limit ID length
                            x=x_val,
                            y=y_val,
                            angle=float(row[f'{prefix}/_angle']),
                            speed=speed_val,
                            position=float(row[f'{prefix}/_pos']),
                            lane=str(row[f'{prefix}/_edge'])[:20],  # Limit lane name length
                            time=float(row['_time']),
                            usager_type=str(row[f'{prefix}/_type'])[:20],
                            categorie='pieton',
                            protocol=v2v_protocol
                        ))
                    except (ValueError, KeyError) as e:
                        logger.debug(f"Skipping invalid pedestrian data: {e}")
                        continue
        
        # Load vehicles - handle different numbering patterns  
        vehicle_prefixes = []
        for col in df.columns:
            if col.startswith('vehicle/') and col.endswith('/_id'):
                prefix = col.replace('/_id', '')
                if prefix not in vehicle_prefixes:
                    vehicle_prefixes.append(prefix)
        
        for prefix in vehicle_prefixes:
            id_col = f'{prefix}/_id'
            
            if id_col in df.columns:
                for _, row in df.iterrows():
                    try:
                        if pd.isna(row[id_col]):
                            continue
                        
                        # Validate vehicle data ranges
                        x_val = float(row[f'{prefix}/_x'])
                        y_val = float(row[f'{prefix}/_y'])
                        speed_val = float(row[f'{prefix}/_speed'])
                        
                        if not (-10000 <= x_val <= 10000) or not (-10000 <= y_val <= 10000):
                            logger.debug(f"Skipping vehicle with out-of-range coordinates: x={x_val}, y={y_val}")
                            continue
                        
                        if not (0 <= speed_val <= 200):  # Max 200 m/s
                            logger.debug(f"Skipping vehicle with unrealistic speed: {speed_val}")
                            continue
                        
                        users.append(User(
                            usager_id=str(row[id_col])[:50],  # Limit ID length
                            x=x_val,
                            y=y_val,
                            angle=float(row[f'{prefix}/_angle']),
                            speed=speed_val,
                            position=float(row[f'{prefix}/_pos']),
                            lane=str(row[f'{prefix}/_lane'])[:20],  # Limit lane name length
                            time=float(row['_time']),
                            usager_type=str(row[f'{prefix}/_type'])[:20],
                            categorie='vehicule',
                            protocol=v2v_protocol
                        ))
                    except (ValueError, KeyError) as e:
                        logger.debug(f"Skipping invalid vehicle data: {e}")
                        continue
        
        if not users:
            logger.warning("No valid users loaded from trace file")
            
        logger.info(f"Successfully loaded {len(users)} users")
        return users
        
    except FileNotFoundError:
        logger.error(f"Trace file not found: {trace_file}")
        raise DataLoadingError(f"Trace file not found: {trace_file}")
    except Exception as e:
        logger.error(f"Error loading trace file: {e}")
        raise DataLoadingError(f"Error loading trace file: {e}")

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
    logger.debug(f"User {user.user_id} attempting V2I communication with {infra.user_id} at distance {distance:.2f}")
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
    logger.debug(f"User {user.user_id} attempting V2V communication with {peer.user_id} at distance {distance:.2f}")
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
        logger.debug(f"Processing user {user.user_id} (type: {user.usager_type})")
        total_communications += 1
        
        # Use MAB to select protocol (0 = V2V, 1 = V2I)
        selected_arm = mab_algorithm.select_arm()
        selected_protocol = 'V2I' if selected_arm == V2I_ARM_INDEX else 'V2V'
        logger.debug(f"MAB selected protocol: {selected_protocol} for user {user.user_id}")
        
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
                logger.debug(f"No infrastructure in range for user {user.user_id}, falling back to V2V")
        
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
            logger.debug(f"No peers in range for user {user.user_id}")
    
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
    
    # Use context manager for resource management
    with SimulationContext(config) as sim_ctx:
        with performance_monitor("Full Simulation"):
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
                        
                    with performance_monitor(f"{algorithm.upper()} Algorithm"):
                        logger.info(f"\n[{algorithm_idx}/{total_algorithms}] Running simulation with {algorithm} algorithm...")
                        start_time = datetime.now()
                        
                        # Initialize MAB algorithm
                        mab = mab_map[algorithm]()
                        logger.debug(f"Initialized {algorithm} algorithm: {mab.__class__.__name__}")
                        
                        # Initialize results list for this algorithm
                        results = []
                        
                        # Load users with V2V protocol as default
                        with performance_monitor("User Loading"):
                            users = load_users(config.trace_file, v2v_protocol)
                            if not users:
                                logger.error("No users loaded")
                            else:
                                logger.info(f"Loaded {len(users)} users from trace file")
                            
                        # Load infrastructure for V2I
                        with performance_monitor("Infrastructure Loading"):
                            infras = load_infrastructure(config.trace_file, v2i_protocol)
                            if not infras:
                                logger.warning("No infrastructure nodes found, V2V-only mode")
                            else:
                                logger.info(f"Loaded {len(infras)} infrastructure nodes")
            
                        
                        # Run simulation with MAB protocol selection
                        if users:
                            simulation_times = sorted(set(user.time for user in users))
                            logger.info(f"Running simulation for {len(simulation_times)} timesteps")
                            
                            # Track protocol selections and performance metrics
                            v2v_selections = 0
                            v2i_selections = 0
                            v2v_successes = 0
                            v2i_successes = 0
                            total_communications = 0
                            
                            with performance_monitor("Timestep Execution"):
                                for time in simulation_times:
                                    timestep_results = run_timestep(users, infras, time, mab)
                                    for protocol_results in timestep_results.values():
                                        results.append(protocol_results)
                                        # Count selections
                                        if protocol_results['Protocol'] == 'V2V':
                                            v2v_selections += 1
                                        else:
                                            v2i_selections += 1
                        else:
                            # No users, create minimal results structure
                            simulation_times = []
                            v2v_selections = 0
                            v2i_selections = 0
                            v2v_successes = 0
                            v2i_successes = 0
                            total_communications = 0
                            logger.info("Skipping timestep execution due to no users")
                        
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
                            if len(simulation_times) > 0:
                                logger.info(f"  Average time per step: {execution_time/len(simulation_times)}")
                        else:
                            logger.info(f"\n{algorithm.upper()} Performance Statistics:")
                            logger.info("No protocol selections made (no valid users loaded)")
                            execution_time = datetime.now() - start_time
                            logger.info(f"Execution time: {execution_time}")
                        
                        # Save results to CSV with algorithm name in filename
                        base_name = Path(config.csv_output).stem
                        ext = Path(config.csv_output).suffix
                        output_path = str(Path(config.csv_output).with_name(f"{base_name}_{algorithm}{ext}"))
                        
                        # Validate output path
                        output_path_validated = validate_file_path(output_path, must_exist=False)
                        
                        # Create DataFrame and save to CSV (even if empty)
                        if results:
                            df = pd.DataFrame(results)
                        else:
                            # Create empty DataFrame with expected columns
                            df = pd.DataFrame(columns=[
                                'Protocol', 'User_ID', 'Time', 'Success', 'Delay', 
                                'Network_Load', 'Packet_Loss', 'Transmission_Time'
                            ])
                        
                        df.to_csv(output_path_validated, index=False)
                        logger.info(f"Results saved to {output_path_validated}")
                        
                        # Also save protocol-specific results (only if there are results)
                        if results:
                            for protocol in ['V2V', 'V2I']:
                                protocol_results = [r for r in results if r['Protocol'] == protocol]
                                if protocol_results:
                                    protocol_output = str(output_path_validated).replace(ext, f"_{protocol.lower()}{ext}")
                                    protocol_path_validated = validate_file_path(protocol_output, must_exist=False)
                                    pd.DataFrame(protocol_results).to_csv(protocol_path_validated, index=False)
                                    logger.info(f"{protocol} results saved to {protocol_path_validated}")
                        
                        # Analyze and log the results (only if there are results)
                        if results:
                            logger.info("Analyzing results...")
                            try:
                                analysis_results = analyze_simulation_results(output_path_validated)
                                for key, value in analysis_results.items():
                                    logger.info(f"{key}: {value}")
                            except Exception as e:
                                logger.error(f"Error during results analysis: {e}")
                        else:
                            logger.info("No results to analyze")
                
                return success
                
            except ConfigurationError as e:
                logger.error(f"Configuration error: {e}")
                success = False
                return False
            except DataLoadingError as e:
                logger.error(f"Data loading error: {e}")
                # Return False for critical errors like missing files
                if "Trace file not found" in str(e):
                    success = False
                    return False
                # Allow graceful degradation for other data parsing errors
                success = True
                return success
            except Exception as e:
                logger.error(f"Simulation failed: {e}")
                success = False
                return False
            
            finally:
                # Only create consolidated output if simulation was successful
                if success and config.csv_output:
                    try:
                        # Collect all algorithm-specific CSV files
                        base_name = Path(config.csv_output).stem
                        ext = Path(config.csv_output).suffix
                        
                        all_results = []
                        for algorithm in config.mab_algorithms:
                            algorithm_file = str(Path(config.csv_output).with_name(f"{base_name}_{algorithm}{ext}"))
                            if Path(algorithm_file).exists():
                                try:
                                    algorithm_df = pd.read_csv(algorithm_file)
                                    if not algorithm_df.empty:
                                        algorithm_df['Algorithm'] = algorithm.upper()
                                        all_results.append(algorithm_df)
                                except Exception as e:
                                    logger.debug(f"Could not read {algorithm_file}: {e}")
                        
                        # Create main consolidated file
                        if all_results:
                            main_df = pd.concat(all_results, ignore_index=True)
                        else:
                            # Create empty DataFrame with expected columns
                            main_df = pd.DataFrame(columns=[
                                'Protocol', 'User_ID', 'Time', 'Success', 'Delay', 
                                'Network_Load', 'Packet_Loss', 'Transmission_Time', 'Algorithm'
                            ])
                        
                        main_output_validated = validate_file_path(config.csv_output, must_exist=False)
                        main_df.to_csv(main_output_validated, index=False)
                        logger.info(f"Consolidated results saved to {main_output_validated}")
                        
                    except Exception as e:
                        logger.error(f"Error creating consolidated output file: {e}")
                        success = False

# Security and validation utilities
def validate_file_path(file_path: str, must_exist: bool = True) -> Path:
    """
    Validate and sanitize file paths to prevent path traversal attacks.
    
    Args:
        file_path: The file path to validate
        must_exist: Whether the file must exist
        
    Returns:
        Validated Path object
        
    Raises:
        ValueError: If path is invalid or dangerous
        FileNotFoundError: If file doesn't exist and must_exist=True
    """
    if not file_path or not isinstance(file_path, str):
        raise ValueError("File path must be a non-empty string")
    
    # Convert to Path object and resolve
    path = Path(file_path).resolve()
    
    # Check for path traversal attempts - only flag obvious malicious patterns
    if '..' in file_path:  # Check original path, not resolved
        raise ValueError(f"Path traversal detected in: {file_path}")
    
    # Allow absolute paths but check for suspicious patterns
    if any(suspicious in str(path).lower() for suspicious in ['/etc/', '/bin/', '/usr/bin/', '/root/']):
        raise ValueError(f"Suspicious system path detected: {file_path}")
    
    # Check file extension for CSV files
    if path.suffix.lower() not in ['.csv', '.log', '.png', '.txt', '']:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    
    if must_exist and not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    return path

def validate_csv_content(file_path: Path, required_columns: List[str] = None) -> bool:
    """
    Validate CSV file content for basic safety checks.
    
    Args:
        file_path: Path to CSV file
        required_columns: List of required column names
        
    Returns:
        True if file is valid
        
    Raises:
        ValueError: If file content is invalid
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Check file size (max 100MB)
            file_size = f.seek(0, 2)
            if file_size > 100 * 1024 * 1024:
                raise ValueError("CSV file too large (max 100MB)")
            
            f.seek(0)
            reader = csv.reader(f)
            
            # Read header
            try:
                header = next(reader)
            except StopIteration:
                raise ValueError("Empty CSV file")
            
            # Validate header
            if required_columns:
                missing_cols = set(required_columns) - set(header)
                if missing_cols:
                    raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Check for suspicious content in header
            for col in header:
                if len(col) > 100 or any(char in col for char in ['<', '>', '&', '"', "'"]):
                    raise ValueError(f"Suspicious column name: {col}")
            
            return True
            
    except Exception as e:
        logger.warning(f"CSV validation failed for {file_path}: {e}")
        return False

# Custom exceptions for better error handling
class SimulationError(Exception):
    """Base exception for simulation-related errors."""
    pass

class ConfigurationError(SimulationError):
    """Raised when configuration is invalid."""
    pass

class DataLoadingError(SimulationError):
    """Raised when data loading fails."""
    pass

class ProtocolError(SimulationError):
    """Raised when protocol communication fails."""
    pass

# Context manager for resource management
class SimulationContext:
    """Context manager for simulation resources."""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.temp_files = []
        self.log_handlers = []
        self.start_time = None
        self.cleanup_performed = False
    
    def __enter__(self):
        self.start_time = time.time()
        logger.debug("Entering simulation context")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cleanup()
        
        if exc_type is not None:
            logger.error(f"Simulation context exited with exception: {exc_type.__name__}: {exc_val}")
        else:
            logger.debug("Simulation context exited normally")
    
    def _cleanup(self):
        """Perform cleanup operations."""
        if self.cleanup_performed:
            return
            
        # Cleanup temporary files
        for temp_file in self.temp_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
                    logger.debug(f"Cleaned up temporary file: {temp_file}")
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file {temp_file}: {e}")
        
        # Cleanup log handlers
        for handler in self.log_handlers:
            try:
                handler.close()
                logging.getLogger().removeHandler(handler)
                logger.debug(f"Cleaned up log handler: {handler}")
            except Exception as e:
                logger.warning(f"Failed to cleanup log handler: {e}")
        
        self.cleanup_performed = True
        
        if self.start_time:
            duration = time.time() - self.start_time
            logger.info(f"Simulation context cleanup completed in {duration:.2f}s")
    
    def add_temp_file(self, file_path: Path):
        """Register a temporary file for cleanup."""
        self.temp_files.append(file_path)
        logger.debug(f"Registered temporary file: {file_path}")
    
    def add_log_handler(self, handler):
        """Register a log handler for cleanup."""
        self.log_handlers.append(handler)
        logger.debug(f"Registered log handler: {handler}")

@contextmanager
def performance_monitor(operation_name: str):
    """Context manager to monitor performance metrics."""
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    logger.debug(f"Starting {operation_name} - Memory: {start_memory:.2f} MB")
    
    try:
        yield
    finally:
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        duration = end_time - start_time
        memory_delta = end_memory - start_memory
        
        logger.info(f"Completed {operation_name}")
        logger.info(f"  Duration: {duration:.2f}s")
        logger.info(f"  Memory: {end_memory:.2f} MB (Δ{memory_delta:+.2f} MB)")
        
        # Warn about high memory usage
        if memory_delta > 100:  # More than 100MB increase
            logger.warning(f"High memory usage detected for {operation_name}: +{memory_delta:.2f} MB")


def run_single_algorithm(algorithm_name: str, config: SimulationConfig) -> Dict[str, Any]:
    """Run simulation for a single MAB algorithm."""
    try:
        # Create a modified config for this algorithm
        algo_config = SimulationConfig(
            v2v_network_load=config.v2v_network_load,
            v2v_packet_loss=config.v2v_packet_loss,
            v2v_transmission_time=config.v2v_transmission_time,
            v2i_network_load=config.v2i_network_load,
            v2i_packet_loss=config.v2i_packet_loss,
            v2i_transmission_time=config.v2i_transmission_time,
            infra_positions=config.infra_positions,
            infra_processing_capacity=config.infra_processing_capacity,
            mab_algorithms=[algorithm_name],  # Only this algorithm
            epsilon_value=config.epsilon_value,
            csv_output=f"results/results_{algorithm_name}.csv",
            trace_file=config.trace_file
        )
        
        with performance_monitor(f"Algorithm {algorithm_name}"):
            success = main(algo_config)
            
        return {
            'algorithm': algorithm_name,
            'success': success,
            'output_file': algo_config.csv_output
        }
        
    except Exception as e:
        logger.error(f"Algorithm {algorithm_name} failed: {e}")
        return {
            'algorithm': algorithm_name,
            'success': False,
            'error': str(e)
        }


def run_parallel_simulation(config: SimulationConfig, max_workers: Optional[int] = None) -> Dict[str, Any]:
    """Run simulation with multiple MAB algorithms in parallel."""
    if len(config.mab_algorithms) <= 1:
        # No need for parallel execution
        return main(config)
    
    if max_workers is None:
        max_workers = min(len(config.mab_algorithms), mp.cpu_count())
    
    logger.info(f"Starting parallel simulation with {len(config.mab_algorithms)} algorithms using {max_workers} workers")
    
    results = {}
    
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all algorithm tasks
            future_to_algorithm = {
                executor.submit(run_single_algorithm, algo, config): algo
                for algo in config.mab_algorithms
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_algorithm):
                algorithm = future_to_algorithm[future]
                try:
                    result = future.result()
                    results[algorithm] = result
                    logger.info(f"Completed algorithm {algorithm}: {'Success' if result['success'] else 'Failed'}")
                except Exception as e:
                    logger.error(f"Algorithm {algorithm} generated an exception: {e}")
                    results[algorithm] = {
                        'algorithm': algorithm,
                        'success': False,
                        'error': str(e)
                    }
    
    except Exception as e:
        logger.error(f"Parallel execution failed: {e}")
        return False
    
    # Aggregate results
    successful_algorithms = [algo for algo, result in results.items() if result['success']]
    failed_algorithms = [algo for algo, result in results.items() if not result['success']]
    
    logger.info(f"Parallel simulation completed:")
    logger.info(f"  Successful algorithms: {successful_algorithms}")
    if failed_algorithms:
        logger.warning(f"  Failed algorithms: {failed_algorithms}")
    
    return {
        'overall_success': len(successful_algorithms) > 0,
        'results': results,
        'successful_algorithms': successful_algorithms,
        'failed_algorithms': failed_algorithms
    }

if __name__ == "__main__":
    setup_logging()
    
    # Check for environment variable configuration
    use_env_config = os.getenv('MAB_USE_ENV_CONFIG', 'false').lower() == 'true'
    enable_parallel = os.getenv('MAB_ENABLE_PARALLEL', 'false').lower() == 'true'
    
    if use_env_config:
        config = SimulationConfig.from_environment()
        logger.info("Using configuration from environment variables")
    else:
        config = SimulationConfig()
        logger.info("Using default configuration")
    
    try:
        if enable_parallel and len(config.mab_algorithms) > 1:
            logger.info("Running parallel simulation")
            result = run_parallel_simulation(config)
            if isinstance(result, dict) and result.get('overall_success', False):
                logger.info("Parallel simulation completed successfully")
            else:
                logger.error("Parallel simulation failed")
        else:
            logger.info("Running single-threaded simulation")
            with performance_monitor("Full Simulation"):
                success = main(config)
            if success:
                logger.info("Simulation completed successfully")
            else:
                logger.error("Simulation failed")
                
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
    except Exception as e:
        logger.error(f"Simulation failed with exception: {e}")
        raise
