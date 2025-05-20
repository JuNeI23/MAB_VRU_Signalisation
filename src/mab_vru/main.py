"""
MAB VRU Signalization - Main Module
Simplified version that maintains all MAB algorithms.
"""
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from mab_vru.MAB import epsilon_greedy_compare, ucb_compare, thompson_compare
from mab_vru.simulation.simulation import main as run_simulation

def setup_logging() -> None:
    """Configure basic logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )

def load_data() -> pd.DataFrame:
    """Load and merge V2V and V2I simulation data."""
    # Load both CSV files
    try:
        df_v2v = pd.read_csv('resultats_v2v.csv')
        df_v2i = pd.read_csv('resultats_v2i.csv')
    except FileNotFoundError as e:
        logging.error(f"Could not load simulation results: {e}")
        return pd.DataFrame()

    # Add protocol column if not present
    if 'Protocol' not in df_v2v.columns:
        df_v2v['Protocol'] = 'V2V'
    if 'Protocol' not in df_v2i.columns:
        df_v2i['Protocol'] = 'V2I'

    # Basic validation
    required_columns = ['Time', 'Average Delay (s)', 'Loss Rate (%)', 'Average Load']
    for df, name in [(df_v2v, 'V2V'), (df_v2i, 'V2I')]:
        if not all(col in df.columns for col in required_columns):
            logging.error(f"Missing required columns in {name} data")
            return pd.DataFrame()

    # Combine the dataframes
    df = pd.concat([df_v2v, df_v2i], ignore_index=True)

    # Simple data cleaning and conversion
    df = df.replace({'N/A': np.nan, 'inf': np.inf}).dropna(how='all', subset=['Average Delay (s)', 'Loss Rate (%)', 'Average Load'])

    # Convert columns to numeric, keeping inf values
    for col in ['Average Delay (s)', 'Loss Rate (%)', 'Average Load']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Sort by time to ensure chronological order
    df = df.sort_values('Time')

    if df.empty:
        logging.warning("No valid data after cleaning")
        return pd.DataFrame()

    # Validate that we have both protocols for each timestamp
    time_counts = df.groupby('Time')['Protocol'].nunique()
    missing_times = time_counts[time_counts < 2].index
    if len(missing_times) > 0:
        logging.warning(f"Some timestamps are missing data for one protocol: {missing_times.tolist()}")
        # Fill missing protocol data with worst-case values
        all_times = df['Time'].unique()
        all_protocols = ['V2V', 'V2I']
        
        # Create a complete index
        multi_idx = pd.MultiIndex.from_product([all_times, all_protocols], names=['Time', 'Protocol'])
        df = df.set_index(['Time', 'Protocol']).reindex(multi_idx)
        
        # Fill missing values with worst case
        worst_case = {
            'Average Delay (s)': df['Average Delay (s)'].max(),
            'Loss Rate (%)': 100.0,  # Worst case is 100% loss
            'Average Load': df['Average Load'].max()
        }
        df = df.fillna(worst_case).reset_index()

    return df

def ensure_plot_dir():
    """Ensure plots directory exists."""
    plot_dir = Path('plots')
    plot_dir.mkdir(exist_ok=True)
    return plot_dir
   
def simulate_and_analyze() -> bool:
    """Run simulation and analyze protocols."""
    try:
        # Create plots directory
        plot_dir = ensure_plot_dir()
        
        # Run simulation
        logging.info("Starting simulation...")
        if not run_simulation():
            return False
        
        # Load and analyze data
        df = load_data()
        if df.empty:
            return False
        
        # Run all MAB analyses with plot paths
        plot_dir = ensure_plot_dir()
        
        logging.info("\nAnalyzing with Epsilon-Greedy:")
        epsilon_greedy_compare(df, save_path=plot_dir / "epsilon_greedy_evolution.png")
        
        logging.info("\nAnalyzing with UCB:")
        ucb_compare(df, save_path=plot_dir / "ucb_evolution.png")
        
        logging.info("\nAnalyzing with Thompson Sampling:")
        thompson_compare(df, save_path=plot_dir / "thompson_evolution.png")
        
        logging.info(f"\nPlots saved in {plot_dir}/")
        return True
        
    except Exception as e:
        logging.error(f"Error during simulation: {e}")
        return False

if __name__ == "__main__":
    setup_logging()
    success = simulate_and_analyze()
    if not success:
        logging.error("Simulation failed")
