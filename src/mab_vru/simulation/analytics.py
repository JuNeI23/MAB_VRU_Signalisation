"""
Analytics module for analyzing MAB simulation results.
"""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

def analyze_protocol_performance(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Analyze the performance metrics for each protocol."""
    protocol_stats = df.groupby('Protocol').agg({
        'Average Delay (s)': ['mean', 'std'],
        'Loss Rate (%)': ['mean', 'std'],
        'Average Load': ['mean', 'std'],
        'MAB Selection Rate (%)': 'mean',
        'Reachability Rate (%)': 'mean'
    }).round(3)
    
    return {
        protocol: {
            'avg_delay': stats[('Average Delay (s)', 'mean')],
            'delay_std': stats[('Average Delay (s)', 'std')],
            'loss_rate': stats[('Loss Rate (%)', 'mean')],
            'loss_std': stats[('Loss Rate (%)', 'std')],
            'avg_load': stats[('Average Load', 'mean')],
            'load_std': stats[('Average Load', 'std')],
            'selection_rate': stats[('MAB Selection Rate (%)', 'mean')],
            'reachability': stats[('Reachability Rate (%)', 'mean')]
        }
        for protocol, stats in protocol_stats.iterrows()
    }

def plot_metrics_evolution(df: pd.DataFrame, save_path: Optional[Path] = None):
    """Plot the evolution of metrics over time for both protocols."""
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Protocol Performance Evolution')
    
    metrics = {
        'Average Delay (s)': (0, 0),
        'Loss Rate (%)': (0, 1),
        'Average Load': (1, 0),
        'MAB Selection Rate (%)': (1, 1)
    }
    
    for metric, (row, col) in metrics.items():
        for protocol in ['V2V', 'V2I']:
            protocol_data = df[df['Protocol'] == protocol]
            axs[row, col].plot(protocol_data['Time'], protocol_data[metric], 
                             label=protocol, 
                             alpha=0.7)
        
        axs[row, col].set_xlabel('Time')
        axs[row, col].set_ylabel(metric)
        axs[row, col].grid(True)
        axs[row, col].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def analyze_simulation_results(results_path: Path, save_plots: bool = True) -> Dict[str, Dict[str, float]]:
    """
    Analyze the results from a MAB simulation.
    
    Args:
        results_path: Path to the CSV file containing simulation results
        save_plots: Whether to save plots to disk
        
    Returns:
        Dict containing performance statistics for each protocol
    """
    # Load results
    df = pd.read_csv(results_path)
    
    # Get algorithm name from filename
    algorithm = results_path.stem.split('_')[1] if '_' in results_path.stem else 'unknown'
    
    logger.info(f"\nAnalyzing results for {algorithm} algorithm:")
    
    # Analyze protocol performance
    stats = analyze_protocol_performance(df)
    
    for protocol, metrics in stats.items():
        logger.info(f"\n{protocol} Performance:")
        logger.info(f"  Average Delay: {metrics['avg_delay']:.3f} ± {metrics['delay_std']:.3f} s")
        logger.info(f"  Loss Rate: {metrics['loss_rate']:.1f} ± {metrics['loss_std']:.1f} %")
        logger.info(f"  Average Load: {metrics['avg_load']:.3f} ± {metrics['load_std']:.3f}")
        logger.info(f"  Selection Rate: {metrics['selection_rate']:.1f} %")
        logger.info(f"  Reachability: {metrics['reachability']:.1f} %")
    
    if save_plots:
        # Create plots directory if it doesn't exist
        plots_dir = Path('plots')
        plots_dir.mkdir(exist_ok=True)
        
        # Plot metrics evolution
        plot_metrics_evolution(df, plots_dir / f'metrics_evolution_{algorithm}.png')
        logger.info(f"\nPlots saved in {plots_dir}/")
    
    return stats
