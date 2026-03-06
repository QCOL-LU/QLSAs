# QLSA Benchmarking Suite

A large-scale benchmarking suite for Quantum Linear Systems Algorithms (QLSAs), supporting multiple algorithms across different quantum hardware providers.

## Overview

This project benchmarks quantum linear systems algorithms including:
- **HHL** (Harrow-Hassidim-Lloyd)
- **QSVT** (Quantum Singular Value Transformation)
- **VQLSA** (Variational Quantum Linear Systems Algorithm)
- **QHD** (Quantum Hamiltonian Descent)

Supported backends:
- **IBM** (via Qiskit)
- **Quantinuum** (via qnexus)

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Access to quantum computing backends (IBM Quantum, Quantinuum)

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd QLSAs
```

### 2. Create Virtual Environment

**Option A: Using venv (Recommended)**

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

**Option B: Using conda**

```bash
# Create conda environment
conda create -n qlsa python=3.10
conda activate qlsa

# Note: If working with qnexus, deactivate conda base environment first
conda deactivate  # if base is active
conda activate qlsa
```

### 3. Install Dependencies

```bash
# Ensure virtual environment is activated
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Backend-Specific Setup

**For IBM Quantum (Qiskit):**
```bash
# No additional setup required if using Qiskit
# Configure IBM Quantum credentials by navigating to
save_qiskit_account.ipynb
```

**For Quantinuum (qnexus):**
```bash
# Authenticate with Quantinuum through terminal
qnx login
# Alternatively, navigate to
save_nexus_account.ipynb
```

## Project Structure

```
QLSAs/
├── src/qlsas/                      # Main Python package
│   ├── algorithms/                # QLSA implementations and shared interfaces
│   │   ├── base.py                # Abstract QLSA base class
│   │   └── hhl/
│   │       ├── hhl.py             # HHL circuit construction
│   │       └── hhl_helpers.py     # HHL helper routines and parameter utilities
│   ├── data_loader.py             # State preparation utilities
│   ├── solver.py                  # Main end-to-end solver entry point
│   ├── transpiler.py              # Backend-aware circuit optimization
│   ├── executer.py                # Circuit execution and runtime sessions
│   ├── ibm_options.py             # IBM error-mitigation configuration
│   ├── post_processor.py          # Counts-to-solution post-processing
│   └── refiner.py                 # Iterative refinement built on the solver
├── examples/                      # Demo and experiment notebooks
├── tests/                         # Pytest suite
├── linear_systems_problems/       # Problem generation utilities
├── data/                          # Saved experiment outputs and datasets
├── pyproject.toml                 # Package metadata and dependencies
└── README.md                      # This file
```

## Codebase Flow

The core package lives under `src/qlsas` and is organized around a simple
pipeline: define a QLSA, build a circuit, transpile for a backend, execute, and
post-process the sampled result. `Refiner` wraps that pipeline in an iterative
refinement loop for harder linear systems.

Key modules:

- `src/qlsas/algorithms/`: algorithm definitions; `HHL` is the main implemented QLSA today.
- `src/qlsas/data_loader.py`: state preparation utilities for loading `b` into a circuit.
- `src/qlsas/solver.py`: the main orchestration entry point used by examples and notebooks.
- `src/qlsas/transpiler.py`: backend-aware circuit optimization before execution.
- `src/qlsas/executer.py`: runtime submission, IBM session handling, and sampler execution.
- `src/qlsas/ibm_options.py`: optional IBM error-mitigation settings such as DD and gate twirling.
- `src/qlsas/post_processor.py`: reconstructs solution data from sampled counts.
- `src/qlsas/refiner.py`: iterative refinement loop built on repeated solver calls.

## Usage

### Basic Solver Flow

The main entry point is `QuantumLinearSolver`, which builds an algorithm circuit,
transpiles it for the selected backend, executes it, and post-processes sampled
counts into a solution estimate.

Typical IBM backend setup looks like:

```python
from qiskit_ibm_runtime import QiskitRuntimeService

from qlsas.algorithms.hhl.hhl import HHL
from qlsas.data_loader import StatePrep
from qlsas.solver import QuantumLinearSolver

service = QiskitRuntimeService(name="QLSAs")
backend = service.backend("ibm_brisbane")

hhl = HHL(
    state_prep=StatePrep(method="default"),
    readout="measure_x",
    num_qpe_qubits=4,
    eig_oracle="classical",
)

solver = QuantumLinearSolver(
    qlsa=hhl,
    backend=backend,
    shots=2048,
    optimization_level=3,
)
```

### IBM Error Mitigation

IBM hardware execution uses `SamplerV2` and post-processes raw counts. Because of
that, the most practical first-line mitigation is suppression-oriented:

- Use `dynamical decoupling` first for deep HHL/QPE circuits with idle windows.
- Use `gate twirling` as an optional second knob when you can afford extra shot cost.
- Do not expect `TREX`, `ZNE`, or `PEC` in the current solver path; those fit
  `EstimatorV2` workflows more naturally than this counts-based `SamplerV2` flow.

Mitigation is opt-in and disabled by default, even when an IBM backend is selected.

```python
from qlsas.ibm_options import IBMExecutionOptions
from qlsas.solver import QuantumLinearSolver

ibm_options = IBMExecutionOptions(
    enable_error_mitigation=True,
    enable_dynamical_decoupling=True,
    dd_sequence_type="XX",
)

solver = QuantumLinearSolver(
    qlsa=hhl,
    backend=backend,
    ibm_options=ibm_options,
    shots=2048,
)
```

To also enable light gate twirling:

```python
ibm_options = IBMExecutionOptions(
    enable_error_mitigation=True,
    enable_dynamical_decoupling=True,
    dd_sequence_type="XX",
    enable_gate_twirling=True
)
```

## Development

### Testing

The project uses [pytest](https://pytest.org/) for testing. Install the package with test dependencies, then run the suite:

```bash
pip install -e ".[test]"
pytest
```

Useful options:

- `pytest -m "not slow"` — skip slow tests (e.g. 8×8 problems)
- `pytest --cov=qlsas` — run tests with a coverage report
- `pytest -v` — verbose output
- `pytest --run-hardware` — explicitly enable tests marked `hardware`

Tests live in the top-level `tests/` directory and mirror the `src/qlsas/` package structure.

Important test-suite policy:

- The default test suite does not talk to real quantum backends.
- Current tests use `AerSimulator`, fake backends, or configuration objects only.
- Any future test that requires a paid or remote backend must be marked `@pytest.mark.hardware`.
- `hardware` tests are skipped by default and only run when `--run-hardware` or `QLSAS_RUN_HARDWARE_TESTS=1` is set explicitly.


```

## License

See [LICENSE](LICENSE) file for details.