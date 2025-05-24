# MAB-VRU Signalization 🚗🛡️

**A Python-based simulation framework for optimizing V2V (Vehicle-to-Vehicle) and V2I (Vehicle-to-Infrastructure) communication protocols for Vulnerable Road Users (VRU) using Multi-Armed Bandit algorithms.**

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Key Features](#-key-features)
3. [Project Status](#-project-status)
4. [Requirements](#-requirements)
5. [Installation](#-installation)
6. [Usage](#-usage)
7. [Project Structure](#-project-structure)
8. [Testing](#-testing)
9. [Results & Performance](#-results--performance)
10. [Implementation Details](#-implementation-details)
11. [Authors](#-authors)

---

## 🌟 Project Overview

This project implements an intelligent protocol selection system for vehicular communications, specifically focused on **Vulnerable Road Users (VRU) safety**. It uses **Multi-Armed Bandit (MAB) algorithms** to dynamically choose between V2V and V2I protocols based on performance metrics such as delay, range, and network load.

The system processes real-world SUMO traffic simulation data and provides comprehensive performance analysis and visualization of different communication strategies.

---

## 🚀 Key Features

- **Dynamic Protocol Selection**: Real-time choice between V2V and V2I using MAB algorithms
- **Multiple MAB Implementations**:
  - **UCB (Upper Confidence Bound)**: Optimistic exploration strategy
  - **ε-greedy**: Simple exploration-exploitation balance
  - **Thompson Sampling**: Bayesian approach to arm selection
- **SUMO Integration**: Works with real traffic simulation data
- **Comprehensive Metrics**: Delay, loss rate, network load, reachability analysis
- **Advanced Visualization**: Performance plots and comparative analysis
- **Robust Error Handling**: Graceful degradation and comprehensive logging
- **Security Features**: Input validation, path traversal protectio

---

## 📦 Requirements

### System Requirements
- **Python** ≥ 3.8
- **SUMO** (Simulation of Urban MObility) - Optional for trace generation
- **Memory**: ~200MB RAM for typical simulations
- **Storage**: ~50MB for full installation with dependencies

### Python Dependencies

**Core Dependencies:**
```
numpy>=1.20.0,<2.0.0        # Numerical computations
pandas>=1.3.0,<3.0.0        # Data manipulation
matplotlib>=3.4.0,<4.0.0    # Visualization
traci>=1.19.0                # SUMO interface
sumolib>=1.19.0              # SUMO utilities
tqdm>=4.65.0,<5.0.0         # Progress bars
psutil>=5.8.0,<6.0.0        # System monitoring
```

**Development Dependencies:**
```
pytest>=7.0.0,<8.0.0        # Testing framework
pytest-cov>=4.0.0,<5.0.0    # Coverage reporting
pytest-xdist>=3.0.0,<4.0.0  # Parallel testing
black>=22.0.0,<24.0.0       # Code formatting
flake8>=6.0.0,<7.0.0        # Linting
mypy>=1.0.0,<2.0.0          # Type checking
```

---

## 🔧 Installation

### Option 1: Quick Installation (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/MAB_VRU_Signalisation.git
cd MAB_VRU_Signalisation

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify installation
python3 verify_project.py
```

### Option 2: Development Installation

```bash
# 1. Clone and setup virtual environment
git clone https://github.com/yourusername/MAB_VRU_Signalisation.git
cd MAB_VRU_Signalisation
python -m venv venv
source venv/bin/activate  # Unix/macOS
# or .\venv\Scripts\activate  # Windows

# 2. Install in development mode
pip install -e ".[dev]"

# 3. Run tests to verify
pytest tests/ -v
```

### Option 3: Using Make (Advanced)

```bash
# Install dependencies and setup development environment
make install

# Run all quality checks
make quality

# Run tests with coverage
make test
```

---

## 🎮 Usage

### Basic Usage

**Run simulation with default settings:**
```bash
python3 src/main.py --file sumoTraceCroisement.csv
```

**Run with specific algorithm:**
```bash
python3 src/main.py --file sumoTraceCroisement.csv --algorithm ucb
```

### Programmatic Usage

```python
from mab_vru.main import main, SimulationConfig

# Default configuration
config = SimulationConfig()
main(config)

# Custom configuration
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

### Command Line Options

```bash
python3 src/main.py [OPTIONS]

Options:
  --file FILE        Path to SUMO trace CSV file
  --algorithm ALG    MAB algorithm: ucb, epsilon-greedy, thompson, or all
  --output DIR       Output directory for results (default: results/)
  --config FILE      Configuration file path
  --verbose          Enable verbose logging
  --no-plots         Disable plot generation
  --parallel         Enable parallel execution
```

---

## 📁 Project Structure

```
MAB_VRU_Signalisation/
├── 📄 README.md                    # This comprehensive documentation
├── 📄 requirements.txt             # All dependencies
├── 📄 pyproject.toml              # Modern Python packaging
├── 📄 setup.py                    # Alternative setup
├── 📄 Makefile                    # Build automation
├── 📄 verify_project.py           # Automated verification
├── 📄 pytest.ini                 # Test configuration
├── 📄 .gitignore                  # Version control
│
├── 📁 src/                        # Source code
│   ├── 📄 main.py                 # Main simulation entry point
│   └── 📁 mab_vru/               # Core package
│       ├── 📄 __init__.py         # Package initialization
│       ├── 📁 MAB/               # Multi-Armed Bandit algorithms
│       │   ├── 📄 __init__.py
│       │   ├── 📄 base_mab.py     # Base MAB class
│       │   ├── 📄 MAB_u.py        # UCB algorithm
│       │   ├── 📄 MAB_e.py        # Epsilon-Greedy algorithm
│       │   └── 📄 MAB_Ts.py       # Thompson Sampling algorithm
│       └── 📁 simulation/         # Simulation framework
│           ├── 📄 __init__.py
│           ├── 📄 models.py       # Data models (User, Infrastructure, Message)
│           ├── 📄 protocols.py    # V2V/V2I protocol implementations
│           ├── 📄 analytics.py    # Results analysis and visualization
│           ├── 📄 metric.py       # Performance metrics calculation
│           ├── 📄 spatial_index.py # Spatial data structures
│           └── 📄 pickleable_queue.py # Thread-safe queue implementation
│
├── 📁 tests/                      # Complete test suite
│   ├── 📄 conftest.py            # Test configuration
│   ├── 📄 requirements.txt       # Test dependencies
│   ├── 📁 fixtures/              # Test data
│   │   └── 📄 sample_trace.csv   # Sample SUMO data
│   ├── 📁 unit/                  # Unit tests (23 files)
│   │   ├── 📄 test_mab.py        # MAB algorithm tests
│   │   ├── 📄 test_models.py     # Data model tests
│   │   ├── 📄 test_main.py       # Main functionality tests
│   │   ├── 📄 test_configuration.py # Configuration tests
│   │   ├── 📄 test_security.py   # Security validation tests
│   │   └── 📄 test_context_managers.py # Context manager tests
│   └── 📁 integration/           # Integration tests
│       ├── 📄 test_simulation.py # Full simulation tests
│       └── 📄 test_system_integration.py # System integration tests
│
├── 📁 results/                    # Generated simulation results
│   ├── 📄 resultats.csv          # Consolidated results
│   ├── 📄 resultats_ucb.csv      # UCB algorithm results
│   ├── 📄 resultats_epsilon-greedy.csv # Epsilon-greedy results
│   └── 📄 resultats_thompson.csv # Thompson sampling results
│
├── 📁 plots/                      # Generated visualizations
│   ├── 📄 performance_comparison.png
│   ├── 📄 protocol_selection.png
│   └── 📄 algorithm_metrics.png
│
├── 📁 logs/                       # Execution logs
│   └── 📄 simulation_YYYYMMDD_HHMMSS.log
│
├── 📁 Data Files
│   ├── 📄 sumoTraceCroisement.csv # Real SUMO trace (1460 users, 203 infrastructure)
│   └── 📄 sumoTrace_edge.csv     # Additional trace data
│
└── 📁 htmlcov/                    # Test coverage reports
    └── 📄 index.html              # Coverage dashboard
```

---

## 🧪 Testing

### Running Tests

**Quick test run:**
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test categories
pytest tests/unit/ -v          # Unit tests only
pytest tests/integration/ -v   # Integration tests only
```

**Using Make:**
```bash
make test          # Run tests with coverage
make test-unit     # Unit tests only
make test-integration # Integration tests only
```

### Test Structure

**📊 Test Statistics:**
- **Total Tests**: 78 tests
- **Success Rate**: 100% (78/78 passing)
- **Coverage**: Comprehensive coverage of all modules
- **Test Types**: Unit tests, Integration tests, System tests

**🔬 Test Categories:**

1. **Unit Tests** (65 tests):
   - MAB algorithm correctness
   - Data model validation
   - Configuration handling
   - Security features
   - Error handling

2. **Integration Tests** (13 tests):
   - Full simulation workflows
   - System component integration
   - Performance validation
   - End-to-end scenarios

### Test Data

- **Real Data**: `sumoTraceCroisement.csv` (1460 users, 203 infrastructure)
- **Test Fixtures**: `tests/fixtures/sample_trace.csv` (controlled test data)
- **Generated Data**: Dynamic test scenarios

---

## 📊 Results & Performance

### Performance Metrics

**Simulation Performance:**
- **Processing Speed**: ~2.5 seconds for full simulation
- **Memory Usage**: ~154MB peak (efficient management)
- **Data Scale**: 1460 users + 203 infrastructure nodes
- **Timesteps**: 205 simulation steps

**Algorithm Comparison (Real Data Results):**

| Algorithm | V2V Selection | V2I Selection | Avg V2V Delay | Avg V2I Delay | V2V Loss Rate | V2I Loss Rate |
|-----------|---------------|---------------|---------------|---------------|---------------|---------------|
| **UCB** | 65.6% | 34.4% | 0.103s | 0.513s | 11.5% | 10.4% |
| **Epsilon-Greedy** | 78.2% | 21.8% | 0.105s | 0.500s | 11.3% | 7.9% |
| **Thompson Sampling** | 66.6% | 33.4% | 0.105s | 0.498s | 9.2% | 5.9% |

### Output Files

**Generated Results:**
```
results/
├── resultats.csv                    # Consolidated results across all algorithms
├── resultats_ucb.csv               # UCB algorithm detailed results
├── resultats_ucb_v2v.csv           # UCB V2V protocol specific results
├── resultats_ucb_v2i.csv           # UCB V2I protocol specific results
├── resultats_epsilon-greedy.csv    # Epsilon-greedy results
├── resultats_epsilon-greedy_v2v.csv
├── resultats_epsilon-greedy_v2i.csv
├── resultats_thompson.csv          # Thompson sampling results
├── resultats_thompson_v2v.csv
└── resultats_thompson_v2i.csv
```

**Performance Metrics Include:**
- **Average Delay** (seconds)
- **Loss Rate** (percentage)
- **Average Network Load**
- **Average Distance** (meters)
- **Reachability Rate** (percentage)
- **MAB Selection Rate** (percentage)
- **Protocol Distribution**

### Visualization

**Generated Plots:**
- Protocol selection distribution
- Performance comparison across algorithms
- Delay and loss rate analysis
- Network load visualization
- Temporal performance evolution

---

## 🔧 Implementation Details

### Multi-Armed Bandit Algorithms

**1. Upper Confidence Bound (UCB):**
```python
# Optimistic approach with confidence intervals
arm_value = average_reward + sqrt(2 * log(total_pulls) / arm_pulls)
```

**2. Epsilon-Greedy:**
```python
# Simple exploration-exploitation balance
if random() < epsilon:
    return random_arm()  # Explore
else:
    return best_arm()    # Exploit
```

**3. Thompson Sampling:**
```python
# Bayesian approach with Beta distributions
sampled_value = beta_distribution(alpha, beta).sample()
```

### Data Processing Pipeline

**1. CSV Loading:**
- Dynamic column detection (`person/0/_id`, `person/1/_id`, etc.)
- Robust error handling for malformed data
- Memory-efficient processing

**2. Spatial Indexing:**
- Efficient neighbor finding for V2V communication
- Infrastructure proximity calculation for V2I
- Optimized range-based queries

**3. Protocol Selection:**
- Real-time MAB arm selection
- Performance metric collection
- Adaptive parameter adjustment

### Security Features

- **Path Validation**: Prevents directory traversal attacks
- **Input Sanitization**: Validates CSV content and structure
- **File Size Limits**: Prevents memory exhaustion
- **Error Recovery**: Graceful handling of malformed inputs


---

## 👥 Authors

- **Sorres Antonin** - Lead Developer
- **Project**: N7 2A Academic Project
- **Institution**: ENSEEIHT (École Nationale Supérieure d'Électrotechnique, d'Électronique, d'Informatique, d'Hydraulique et des Télécommunications)
- **Date**: May 2025

---

## 📄 License

This project is developed for academic purposes as part of the N7 2A curriculum.

