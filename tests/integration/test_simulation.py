"""
Integration tests for VRU simulation.
"""
import pytest
import pandas as pd
from pathlib import Path
from mab_vru.simulation.simulation import main, SimulationConfig
import mab_vru.simulation.simulation as simulation

def test_full_simulation_run(temp_output_dir, sample_trace_data):
    """Test a complete simulation run with sample data."""
    # Create a temporary trace file
    trace_file = temp_output_dir / "test_trace.csv"
    sample_trace_data.to_csv(trace_file, index=False)
    
    # Create simulation config
    config = SimulationConfig(
        v2v_network_load=0.1,
        v2v_packet_loss=0.1,
        v2v_transmission_time=0.1,
        v2i_network_load=0.1,
        v2i_packet_loss=0.05,
        v2i_transmission_time=0.5,
        csv_output=str(temp_output_dir / "test_results.csv"),
        trace_file=str(trace_file)
    )
    
    # Run simulation
    success = simulation.main(config)
    assert success
    
    # Check output file exists
    results_file = Path(config.csv_output)
    assert results_file.exists()
    
    # Load and validate results
    results = pd.read_csv(results_file)
    assert not results.empty
    assert all(col in results.columns for col in [
        'Time', 'Protocol', 'Average Delay (s)', 'Loss Rate (%)', 'Average Load'
    ])
    assert set(results['Protocol'].unique()) == {'V2V', 'V2I'}

def test_simulation_error_handling(temp_output_dir):
    """Test simulation error handling with invalid input."""
    config = SimulationConfig(
        csv_output=str(temp_output_dir / "test_results.csv"),
        trace_file="nonexistent_file.csv"
    )
    
    # Simulation should fail gracefully with nonexistent input
    success = simulation.main(config)
    assert not success
    
    # Results file should not be created
    assert not Path(config.csv_output).exists()

def test_simulation_metrics(temp_output_dir, sample_trace_data):
    """Test that simulation metrics are within expected ranges."""
    # Create a temporary trace file
    trace_file = temp_output_dir / "test_trace.csv"
    sample_trace_data.to_csv(trace_file, index=False)
    
    config = SimulationConfig(
        csv_output=str(temp_output_dir / "test_results.csv"),
        trace_file=str(trace_file)
    )
    
    # Run simulation
    success = simulation.main(config)
    assert success
    
    # Load results
    results = pd.read_csv(config.csv_output)
    
    # Check metric ranges
    assert all(0 <= delay <= 10 for delay in results['Average Delay (s)'])
    assert all(0 <= loss <= 100 for loss in results['Loss Rate (%)'])
    assert all(0 <= load <= 1 for load in results['Average Load'])

def test_protocol_comparison(temp_output_dir, sample_trace_data):
    """Test that protocol comparison produces valid results."""
    # Create a temporary trace file
    trace_file = temp_output_dir / "test_trace.csv"
    sample_trace_data.to_csv(trace_file, index=False)
    
    config = SimulationConfig(
        csv_output=str(temp_output_dir / "test_results.csv"),
        trace_file=str(trace_file)
    )
    
    # Run simulation
    success = simulation.main(config)
    assert success
    
    # Load results
    results = pd.read_csv(config.csv_output)
    
    # Group by protocol and calculate averages
    protocol_stats = results.groupby('Protocol').agg({
        'Average Delay (s)': 'mean',
        'Loss Rate (%)': 'mean',
        'Average Load': 'mean'
    })
    
    # Check that both protocols have results
    assert len(protocol_stats) == 2
    assert set(protocol_stats.index) == {'V2V', 'V2I'}

def test_simulation_with_high_load(temp_output_dir, sample_trace_data):
    """Test simulation behavior under high network load."""
    trace_file = temp_output_dir / "test_trace.csv"
    sample_trace_data.to_csv(trace_file, index=False)
    
    config = SimulationConfig(
        v2v_network_load=0.9,  # High load
        v2v_packet_loss=0.2,
        v2v_transmission_time=0.2,
        v2i_network_load=0.9,  # High load
        v2i_packet_loss=0.1,
        v2i_transmission_time=0.5,
        csv_output=str(temp_output_dir / "high_load_results.csv"),
        trace_file=str(trace_file)
    )
    
    success = simulation.main(config)
    assert success
    
    results = pd.read_csv(config.csv_output)
    assert not results.empty
    
    # Under high load, delays should be higher
    v2v_delays = results[results['Protocol'] == 'V2V']['Average Delay (s)']
    v2i_delays = results[results['Protocol'] == 'V2I']['Average Delay (s)']
    assert v2v_delays.mean() > 0.2
    assert v2i_delays.mean() > 0.5

def test_simulation_protocol_switching(temp_output_dir, sample_trace_data):
    """Test dynamic protocol switching based on network conditions."""
    trace_file = temp_output_dir / "test_trace.csv"
    sample_trace_data.to_csv(trace_file, index=False)
    
    config = SimulationConfig(
        v2v_network_load=0.5,
        v2v_packet_loss=0.1,
        v2v_transmission_time=0.1,
        v2i_network_load=0.5,
        v2i_packet_loss=0.05,
        v2i_transmission_time=0.5,
        csv_output=str(temp_output_dir / "switching_results.csv"),
        trace_file=str(trace_file)
    )
    
    success = simulation.main(config)
    assert success
    
    results = pd.read_csv(config.csv_output)
    protocols = results['Protocol'].values
    
    # Check if protocol switching occurs (both protocols are used)
    assert 'V2V' in protocols
    assert 'V2I' in protocols
    
    # Protocol switches should be based on network conditions
    delays = results['Average Delay (s)']
    loss_rates = results['Loss Rate (%)']
    loads = results['Average Load']
    
    # Verify that protocol selection correlates with network conditions
    high_delay_mask = delays > delays.mean()
    assert not all(results.loc[high_delay_mask, 'Protocol'] == 'V2V')
    assert not all(results.loc[high_delay_mask, 'Protocol'] == 'V2I')
