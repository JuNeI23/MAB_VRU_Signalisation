# MAB-VRU Signalization

A Python-based simulation framework for optimizing V2V (Vehicle-to-Vehicle) and V2I (Vehicle-to-Infrastructure) communication protocols for Vulnerable Road Users (VRU) using Multi-Armed Bandit algorithms.

## Overview

This project implements an intelligent protocol selection system for vehicular communications, specifically focused on VRU safety. It uses Multi-Armed Bandit (MAB) algorithms to dynamically choose between V2V and V2I protocols based on performance metrics such as delay, range, and network load.

### Key Features

- Dynamic protocol selection between V2V and V2I using MAB algorithms
- Support for multiple MAB implementations:
  - UCB (Upper Confidence Bound)
  - ε-greedy
  - Thompson Sampling
- Integration with SUMO traffic simulator
- Comprehensive metrics collection and analysis
- Visualization of simulation results

## Requirements

- Python ≥ 3.8
- SUMO (Simulation of Urban MObility)

### Python Dependencies

Core dependencies:
```
numpy>=1.20.0
pandas>=1.3.0
matplotlib>=3.4.0
traci
sumolib
tqdm>=4.65.0
```

Development dependencies:
```
pytest>=7.0.0
pytest-cov>=4.0.0
pytest-xdist>=3.0.0
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/MAB_VRU_Signalisation.git
cd MAB_VRU_Signalisation
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Unix/macOS
# or
.\venv\Scripts\activate  # On Windows
```

3. Install the package and dependencies:
```bash
pip install -e ".[dev]"
```

## Usage

### Running a Simulation

The main simulation can be run with default parameters:

```python
from mab_vru.main import main, SimulationConfig

# Use default configuration
config = SimulationConfig()
main(config)
```

Or with custom parameters:

```python
config = SimulationConfig(
    v2v_network_load=0.1,
    v2v_packet_loss=0.1,
    v2v_transmission_time=0.1,
    v2i_network_load=0.1,
    v2i_packet_loss=0.05,
    v2i_transmission_time=0.5,
    mab_algorithms=['ucb', 'epsilon-greedy', 'thompson'],
    epsilon_value=0.1
)
main(config)
```

### Directory Structure

```
MAB_VRU_Signalisation/
├── src/
│   └── mab_vru/
│       ├── MAB/           # MAB algorithm implementations
│       └── simulation/    # Simulation components
├── tests/                 # Test suite
├── results/              # Simulation results
├── logs/                 # Simulation logs
└── plots/                # Generated visualizations
```

## Testing

Run the full test suite:

```bash
pytest
```

Run tests with coverage report:

```bash
pytest --cov=simulation --cov=MAB --cov-report=html
```

## Results Analysis

Simulation results are saved in the `results/` directory in CSV format. Each run generates:
- Main results file: `results/resultats_{algorithm}.csv`
- Protocol-specific results: `results/resultats_{algorithm}_v2v.csv` and `results/resultats_{algorithm}_v2i.csv`

Performance metrics include:
- Average Delay (s)
- Loss Rate (%)
- Average Network Load
- Average Distance
- Reachability Rate (%)
- MAB Selection Rate (%)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run the test suite
5. Submit a pull request

## License

[Add your license information here]

## Authors

- Sorre Antonin
