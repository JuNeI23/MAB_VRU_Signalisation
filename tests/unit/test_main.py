"""Unit tests for main simulation module."""
import pytest
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import os

from mab_vru.simulation.models import User, Infrastructure, Node
from mab_vru.simulation.protocols import Protocol
from mab_vru.simulation.metric import Metric
from mab_vru.MAB.MAB_u import UCBMAB
from mab_vru.main import (
    SimulationConfig,
    run_timestep,
    load_users,
    load_infrastructure,
    setup_logging,
    main
)

@pytest.fixture
def sample_config():
    """Create a sample simulation configuration."""
    return SimulationConfig(
        v2v_network_load=0.1,
        v2v_packet_loss=0.1,
        v2v_transmission_time=0.1,
        v2i_network_load=0.1,
        v2i_packet_loss=0.05,
        v2i_transmission_time=0.5,
        infra_positions=[(0.0, 0.0)],
        infra_processing_capacity=100.0,
        mab_algorithms=['ucb'],
        epsilon_value=0.1
    )

@pytest.fixture
def sample_users():
    """Create a sample list of users."""
    v2v_protocol = Protocol("V2V", 0.1, 0.1, 0.1)
    users = []
    for i in range(3):
        users.append(User(
            usager_id=f"user_{i}",
            x=float(i * 10),
            y=0.0,
            angle=0.0,
            speed=30.0,
            position=float(i * 10),
            lane="lane_0",
            time=0.0,
            usager_type="car",
            categorie="vehicule",
            protocol=v2v_protocol
        ))
    return users

@pytest.fixture
def sample_infrastructure():
    """Create a sample list of infrastructure nodes."""
    v2i_protocol = Protocol("V2I", 0.1, 0.05, 0.5)
    return [
        Infrastructure(
            id="infra_0",
            protocol=v2i_protocol,
            x=0.0,
            y=0.0,
            processing_capacity=100,
            time=0.0
        )
    ]

def test_simulation_config_defaults():
    """Test SimulationConfig default values."""
    config = SimulationConfig()
    assert config.v2v_network_load == 0.1
    assert config.v2i_network_load == 0.1
    assert config.infra_positions == [(0.0, 0.0)]
    assert config.mab_algorithms == ['ucb', 'epsilon-greedy', 'thompson']

def test_simulation_config_custom_values():
    """Test SimulationConfig with custom values."""
    config = SimulationConfig(
        v2v_network_load=0.2,
        v2i_network_load=0.3,
        infra_positions=[(1.0, 1.0), (2.0, 2.0)],
        mab_algorithms=['ucb']
    )
    assert config.v2v_network_load == 0.2
    assert config.v2i_network_load == 0.3
    assert config.infra_positions == [(1.0, 1.0), (2.0, 2.0)]
    assert config.mab_algorithms == ['ucb']

def test_run_timestep_v2v_only(sample_users):
    """Test run_timestep with V2V communication only."""
    mab = UCBMAB(n_arms=2)
    results = run_timestep(sample_users, [], 0.0, mab)
    
    assert 'V2V' in results
    assert 'V2I' not in results
    assert results['V2V']['Protocol'] == 'V2V'
    assert isinstance(results['V2V']['Average Delay (s)'], float)
    assert isinstance(results['V2V']['Loss Rate (%)'], float)

def test_run_timestep_with_infrastructure(sample_users, sample_infrastructure):
    """Test run_timestep with both V2V and V2I communication."""
    mab = UCBMAB(n_arms=2)
    results = run_timestep(sample_users, sample_infrastructure, 0.0, mab)
    
    # Should have results for both protocols since MAB will try both
    assert 'V2V' in results or 'V2I' in results
    if 'V2V' in results:
        assert isinstance(results['V2V']['Average Delay (s)'], float)
    if 'V2I' in results:
        assert isinstance(results['V2I']['Average Delay (s)'], float)

def test_load_users_empty_file(tmp_path):
    """Test load_users with empty CSV file."""
    csv_path = tmp_path / "empty.csv"
    df = pd.DataFrame()
    df.to_csv(csv_path)
    
    v2v_protocol = Protocol("V2V", 0.1, 0.1, 0.1)
    users = load_users(str(csv_path), v2v_protocol)
    assert len(users) == 0  # Should return empty list

def test_load_infrastructure_empty_file(tmp_path):
    """Test load_infrastructure with empty CSV file."""
    csv_path = tmp_path / "empty.csv"
    df = pd.DataFrame()
    df.to_csv(csv_path)
    
    v2i_protocol = Protocol("V2I", 0.1, 0.05, 0.5)
    infras = load_infrastructure(str(csv_path), v2i_protocol)
    
    # Should create default infrastructure
    assert len(infras) == 1
    assert infras[0].x == 0.0
    assert infras[0].y == 0.0

def test_load_users_with_valid_data(tmp_path):
    """Test load_users with valid CSV data."""
    csv_path = tmp_path / "valid.csv"
    df = pd.DataFrame({
        '_time': [0.0],
        'person/_id': ['person_1'],
        'person/_x': [10.0],
        'person/_y': [20.0],
        'person/_angle': [45.0],
        'person/_speed': [5.0],
        'person/_pos': [10.0],
        'person/_edge': ['edge_1'],
        'person/_type': ['pedestrian']
    })
    df.to_csv(csv_path)
    
    v2v_protocol = Protocol("V2V", 0.1, 0.1, 0.1)
    users = load_users(str(csv_path), v2v_protocol)
    
    assert len(users) == 1
    assert users[0].user_id == 'person_1'
    assert users[0].x == 10.0
    assert users[0].y == 20.0

def test_setup_logging(tmp_path):
    """Test logging setup."""
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        setup_logging()
        log_dir = tmp_path / "logs"
        assert log_dir.exists()
        assert list(log_dir.glob("simulation_*.log"))
    finally:
        os.chdir(old_cwd)

def test_main_with_default_config():
    """Test main function with default configuration."""
    config = SimulationConfig(
        mab_algorithms=['ucb'],  # Only test one algorithm to keep test fast
        csv_output='test_results.csv'
    )
    try:
        success = main(config)
        assert success is True
    finally:
        # Cleanup
        try:
            os.remove('test_results.csv')
            os.remove('test_results_v2v.csv')
            os.remove('test_results_v2i.csv')
        except FileNotFoundError:
            pass

def test_main_with_invalid_algorithm():
    """Test main function with invalid algorithm."""
    config = SimulationConfig(
        mab_algorithms=['invalid_algorithm']
    )
    success = main(config)
    assert success is True  # Should still succeed but skip invalid algorithm
