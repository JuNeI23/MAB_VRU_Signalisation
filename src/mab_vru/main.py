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

def load_data(result_path: str = 'resultats.csv') -> pd.DataFrame:
    """Load and validate simulation data."""
    df = pd.read_csv(result_path)
    
    # Basic validation
    required_columns = ['Time', 'Protocol', 'Average Delay (s)', 'Loss Rate (%)', 'Average Load']
    if not all(col in df.columns for col in required_columns):
        raise ValueError("Missing required columns")
    
    # Simple data cleaning
    df = df.replace({'N/A': np.nan}).dropna()
    
    return df
   
def simulate_and_analyze() -> bool:
    """Run simulation and analyze protocols."""
    try:
        # Run simulation
        logging.info("Starting simulation...")
        if not run_simulation():
            return False
        
        # Load and analyze data
        df = load_data()
        if df.empty:
            return False
        
        # Run all MAB analyses
        logging.info("\nAnalyzing with Epsilon-Greedy:")
        epsilon_greedy_compare(df)
        
        logging.info("\nAnalyzing with UCB:")
        ucb_compare(df)
        
        logging.info("\nAnalyzing with Thompson Sampling:")
        thompson_compare(df)
        
        return True
        
    except Exception as e:
        logging.error(f"Error during simulation: {e}")
        return False

if __name__ == "__main__":
    setup_logging()
    success = simulate_and_analyze()
    if not success:
        logging.error("Simulation failed")
