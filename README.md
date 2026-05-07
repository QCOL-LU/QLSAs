# QLSA Toolkit

A toolkit for running Quantum Linear Systems Algorithms (QLSAs) on quantum hardware, supporting multiple algorithms across different quantum hardware providers.

## Overview

Supported algorithms:
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
│   ├── readout/                   # Pluggable readout strategies
│   │   ├── base.py                # Readout / MultiCircuitReadout ABCs, QLSACircuit, SuccessCriterion, TomographyResult
│   │   ├── measure_x.py           # Direct Z-basis tomography
│   │   ├── hrf_readout.py         # Hadamard Random Forest tomography (multi-circuit)
│   │   └── swap_test.py           # Inner-product estimation via swap test
│   ├── state_prep.py              # State preparation utilities
│   ├── solver.py                  # End-to-end solver: build → readout → execute → post-process
│   ├── transpiler.py              # Backend-aware circuit optimization
│   ├── executer.py                # Circuit execution and runtime sessions
│   ├── measurement_result.py      # Backend-agnostic measurement-result wrapper + post-selection
│   ├── ibm_options.py             # IBM error-mitigation configuration
│   ├── post_processor.py          # Counts-to-solution post-processing (norm_estimation, tomography_from_counts)
│   └── refiner.py                 # Iterative refinement built on the solver
├── docs/                          # Architecture and algorithm documentation
│   ├── architecture.md            # Pipeline, components, extension points, migration notes
│   └── hrf_readout.md             # HRF tomography algorithm details
├── examples/                      # Demo and experiment notebooks
├── tests/                         # Pytest suite
├── linear_systems_problems/       # Problem generation utilities
├── data/                          # Saved experiment outputs and datasets
├── pyproject.toml                 # Package metadata and dependencies
└── README.md                      # This file
```

## Codebase Flow

The core package lives under `src/qlsas` and is organized around a single
pipeline: define a **QLSA**, append a **readout**, transpile, execute, and
**post-process** the sampled result into a `SolveResult`. `Refiner` wraps that
pipeline in an iterative-refinement loop for harder linear systems.

Three swappable strategy components plug into `QuantumLinearSolver`:

- **State preparation** (`state_prep.py`) — how `b` is loaded into a register.
- **QLSA algorithm** (`algorithms/`) — produces a `QLSACircuit` carrying the core circuit + a `SuccessCriterion` that defines which classical-register values mark a successful shot.
- **Readout** (`readout/`) — appends measurement gates to the QLSA circuit and post-processes the results. Single-circuit readouts (`MeasureXReadout`, `SwapTestReadout`) implement `apply` + `process`. Multi-circuit readouts (`HRFReadout`) subclass `MultiCircuitReadout` and implement `build_circuits` + `combine_results`. The solver dispatches on the protocol, so adding a new readout — or a new QLSA like QSVT with a multi-register success criterion — needs zero changes to the orchestrator.

For a full architectural reference (data flow diagram, component
responsibilities, how to add new QLSA algorithms or readout strategies, and
the migration notes from the recent refactor), see
[docs/architecture.md](docs/architecture.md).

Key modules:

- `src/qlsas/algorithms/`: algorithm definitions; `HHL` is the main implemented QLSA today.
- `src/qlsas/readout/`: pluggable readout strategies and the `Readout` / `MultiCircuitReadout` ABCs.
- `src/qlsas/state_prep.py`: state preparation utilities for loading `b` into a circuit.
- `src/qlsas/solver.py`: the main orchestration entry point used by examples and notebooks.
- `src/qlsas/measurement_result.py`: backend-agnostic measurement-result wrapper and the single source of truth for shot post-selection.
- `src/qlsas/transpiler.py`: backend-aware circuit optimization before execution.
- `src/qlsas/executer.py`: runtime submission, IBM session handling, and sampler execution.
- `src/qlsas/ibm_options.py`: optional IBM error-mitigation settings such as DD and gate twirling.
- `src/qlsas/post_processor.py`: reconstructs solution data from sampled counts (free functions: `norm_estimation`, `tomography_from_counts`, `swap_test_from_counts`).
- `src/qlsas/refiner.py`: iterative refinement loop built on repeated solver calls.

## Usage

### Basic Solver Flow

The main entry point is `QuantumLinearSolver`. It builds an algorithm circuit,
appends a readout, transpiles for the selected backend, executes, and
post-processes sampled counts into a `SolveResult` carrying both the
unit-norm direction and the physically-scaled solution.

State preparation, the QLSA, and the readout are independent, swappable
components. Typical IBM backend setup:

```python
from qiskit_ibm_runtime import QiskitRuntimeService

from qlsas.algorithms.hhl import HHL
from qlsas.readout import MeasureXReadout
from qlsas.solver import QuantumLinearSolver
from qlsas.state_prep import DefaultStatePrep

service = QiskitRuntimeService(name="QLSAs")
backend = service.backend("ibm_brisbane")

solver = QuantumLinearSolver(
    qlsa=HHL(num_qpe_qubits=4),  # default eig_oracle=UCRYEigOracle()
    readout=MeasureXReadout(),
    backend=backend,
    state_prep=DefaultStatePrep(),
    shots=2048,
    optimization_level=3,
)

result = solver.solve(A, b)
result.solution    # physically-scaled vector (= alpha * direction)
result.direction   # unit-norm direction
result.alpha       # least-squares scale
result.residual    # ‖A·solution − b‖
```

Swap in a different readout strategy by replacing `MeasureXReadout()`. For
example, for honest end-to-end quantum sign recovery via Hadamard Random
Forest tomography (see [docs/hrf_readout.md](docs/hrf_readout.md)):

```python
from qlsas.readout import HRFReadout

solver = QuantumLinearSolver(
    qlsa=HHL(num_qpe_qubits=4),
    readout=HRFReadout(num_trees=20),
    backend=backend,
    shots=4096,
)
```

The solver dispatches single- vs multi-circuit readouts automatically.

Inversion oracles other than the `UCRYEigOracle` default (`MCRYEigOracle`,
`ExactReciprocalEigOracle`) are available — see
[docs/eigenvalue_inversion.md](docs/eigenvalue_inversion.md) for the full
trade-off analysis and decision guide.

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