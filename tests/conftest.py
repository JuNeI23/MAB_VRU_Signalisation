"""
Pytest configuration and fixtures for VRU simulation tests.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import os
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))
from main import SimulationConfig

@pytest.fixture
def sample_simulation_data():
    """Create a sample DataFrame mimicking simulation results."""
    data = {
        'Time': np.repeat(range(10), 2),
        'Protocol': ['V2V', 'V2I'] * 10,
        'Average Delay (s)': np.random.uniform(0.1, 2.0, 20),
        'Loss Rate (%)': np.random.uniform(0, 100, 20),
        'Average Load': np.random.uniform(0.1, 1.0, 20)
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_trace_data():
    """Load sample SUMO trace data from fixture file."""
    fixture_path = Path(__file__).parent / "fixtures" / "sample_trace.csv"
    return pd.read_csv(fixture_path)

@pytest.fixture
def simulation_config(temp_output_dir):
    """Create a simulation configuration for testing."""
    return SimulationConfig(
        v2v_network_load=0.1,
        v2v_packet_loss=0.1,
        v2v_transmission_time=0.1,
        v2i_network_load=0.1,
        v2i_packet_loss=0.05,
        v2i_transmission_time=0.5,
        csv_output=str(temp_output_dir / "test_results.csv"),
        trace_file=str(temp_output_dir / "test_trace.csv")
    )

@pytest.fixture
def temp_output_dir(tmpdir):
    """Create a temporary directory for test outputs."""
    output_dir = Path(tmpdir) / "test_outputs"
    output_dir.mkdir(exist_ok=True)
    return output_dir
