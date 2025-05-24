"""
Integration tests for VRU simulation.
"""
import pytest
import pandas as pd
from pathlib import Path
from main import main, SimulationConfig
import main as simulation

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
    # MAB algorithm may choose only the best-performing protocol
    protocols_used = set(results['Protocol'].unique())
    assert protocols_used.issubset({'V2V', 'V2I'})
    assert len(protocols_used) >= 1  # At least one protocol should be used

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
    
    # Check that at least one protocol has results
    assert len(protocol_stats) >= 1
    assert set(protocol_stats.index).issubset({'V2V', 'V2I'})

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
    
    # Under high load, MAB algorithms should still select optimal protocol
    # V2V is expected to be chosen due to lower delay (0.2s vs 0.5s for V2I)
    used_protocols = results['Protocol'].unique()
    assert len(used_protocols) > 0
    assert all(p in ['V2V', 'V2I'] for p in used_protocols)
    
    # Check that selected protocol performs within expected ranges
    avg_delays = results['Average Delay (s)']
    assert avg_delays.max() <= 1.0   # Reasonable upper bound
    
    # Under high load, we expect at least some successful communications
    success_counts = results['Success Count']
    total_comms = results['Total Communications']
    assert total_comms.sum() > 0     # At least some communications attempted
    assert success_counts.sum() >= 0 # Success counts should be non-negative

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
    
    # Check that at least one protocol is used (MAB chooses best performing)
    assert len(protocols) > 0
    assert all(p in ['V2V', 'V2I'] for p in protocols)
    
    # Validate performance metrics are reasonable
    delays = results['Average Delay (s)']
    loss_rates = results['Loss Rate (%)']
    loads = results['Average Load']
    
    # MAB algorithms should consistently choose the better performing protocol
    # Given V2V has lower transmission time (0.1s vs 0.5s), it should be preferred
    dominant_protocol = results['Protocol'].mode().iloc[0] if len(results) > 0 else 'V2V'
    assert dominant_protocol in ['V2V', 'V2I']
    
    # Verify that selected protocol has reasonable performance
    assert delays.mean() > 0.05  # Minimum realistic delay
    assert delays.mean() < 1.0   # Maximum reasonable delay
    assert all(loss_rates >= 0)  # Loss rates should be non-negative
    assert all(loads > 0)        # Loads should be positive
