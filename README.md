# QLSA Benchmarking Suite

A large-scale benchmarking suite for Quantum Linear Systems Algorithms (QLSAs), supporting multiple algorithms across different quantum hardware providers.

## Overview

This project benchmarks quantum linear systems algorithms including:
- **HHL** (Harrow-Hassidim-Lloyd) algorithm
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
# Configure IBM Quantum credentials when needed
```

**For Quantinuum (qnexus):**
```bash
# Authenticate with Quantinuum
qnx login
```

## Project Structure

```
QLSAs/
├── circuits/              # Algorithm circuit implementations
│   ├── hhl/              # HHL algorithm circuits
│   ├── vqlsa/            # VQLSA algorithm circuits
│   └── qhd/              # QHD algorithm circuits
├── solvers/              # Solver implementations
├── iterative_refinement/ # Iterative refinement utilities
├── linear_systems_problems/ # Problem generation and datasets
├── qiskit/               # Qiskit-specific implementations
├── qnexus/               # qnexus-specific implementations
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Usage

[Usage instructions will be added as the project structure is finalized]

## Development

### Working with Virtual Environments

- Always activate your virtual environment before running scripts
- If using conda, ensure the base environment is deactivated when working with qnexus
- Keep virtual environments separate for different backend requirements if needed

### Adding Dependencies

```bash
# Activate virtual environment first
source venv/bin/activate  # or: conda activate qlsa

# Install new package
pip install <package-name>

# Update requirements.txt
pip freeze > requirements.txt
```

## License

See [LICENSE](LICENSE) file for details.