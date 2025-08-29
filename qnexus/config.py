"""
Configuration file for the qnexus HHL quantum linear solver.
Centralizes all configurable parameters to improve maintainability.
"""

import os
from typing import Set, Dict, Any

# ============================================================================
# BACKEND CONFIGURATION
# ============================================================================

# Available emulator backends
EMULATOR_BACKENDS: Set[str] = {"H1-1E", "H2-1E", "H2-2E"}

# Available hardware backends (require access)
HARDWARE_BACKENDS: Set[str] = {"H1-1", "H2-1", "H2-2"}

# Default backend for testing
DEFAULT_BACKEND: str = "H1-1E"

# ============================================================================
# QUANTUM JOB CONFIGURATION
# ============================================================================

# Default number of shots for quantum measurements
DEFAULT_SHOTS: int = 1024

# Default timeout for quantum job waiting (in seconds)
DEFAULT_TIMEOUT: int = 3600  # 1 hour

# Poll interval for job status checking (in seconds)
DEFAULT_POLL_INTERVAL: int = 5

# ============================================================================
# PROBLEM GENERATION CONFIGURATION
# ============================================================================

# Default condition number for generated matrices
DEFAULT_CONDITION_NUMBER: float = 5.0

# Default sparsity for generated matrices
DEFAULT_SPARSITY: float = 0.5

# Default random seed for reproducibility
DEFAULT_SEED: int = 42

# Maximum problem size for testing (to prevent memory issues)
MAX_TEST_PROBLEM_SIZE: int = 8

# ============================================================================
# HHL ALGORITHM CONFIGURATION
# ============================================================================

# Default number of QPE qubits
DEFAULT_QPE_QUBITS: int = 2

# Default time parameter for HHL
DEFAULT_T0: float = 2.0

# ============================================================================
# ITERATIVE REFINEMENT CONFIGURATION
# ============================================================================

# Default precision for iterative refinement
DEFAULT_PRECISION: float = 1e-5

# Default maximum iterations for iterative refinement
DEFAULT_MAX_ITERATIONS: int = 5

# Default scaling parameters for IR
DEFAULT_NABLA: float = 1.0
DEFAULT_RHO: float = 2.0

# ============================================================================
# COMPILATION CONFIGURATION
# ============================================================================

# Default optimization level for circuit compilation
DEFAULT_OPTIMIZATION_LEVEL: int = 2

# Default attempt batching for emulators
DEFAULT_ATTEMPT_BATCHING: bool = True

# Default noisy simulation for emulators
DEFAULT_NOISY_SIMULATION: bool = True

# ============================================================================
# FILE AND OUTPUT CONFIGURATION
# ============================================================================

# Default data directory
DEFAULT_DATA_DIR: str = "data"

# Default output format for plots
DEFAULT_PLOT_FORMAT: str = "png"

# Default DPI for plots
DEFAULT_PLOT_DPI: int = 300

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

# Default log level
DEFAULT_LOG_LEVEL: str = "INFO"

# Whether to enable verbose output
DEFAULT_VERBOSE: bool = True

# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_backend(backend: str) -> bool:
    """Validate that the backend is supported."""
    return backend in EMULATOR_BACKENDS or backend in HARDWARE_BACKENDS

def validate_problem_size(size: int) -> bool:
    """Validate that the problem size is a power of 2."""
    return size > 0 and (size & (size - 1)) == 0

def validate_shots(shots: int) -> bool:
    """Validate that the number of shots is positive."""
    return shots > 0

def validate_qpe_qubits(qpe_qubits: int) -> bool:
    """Validate that the number of QPE qubits is positive."""
    return qpe_qubits > 0

def validate_precision(precision: float) -> bool:
    """Validate that the precision is positive."""
    return precision > 0

def validate_max_iterations(max_iter: int) -> bool:
    """Validate that the maximum iterations is positive."""
    return max_iter > 0

# ============================================================================
# CONFIGURATION GETTERS
# ============================================================================

def get_backend_config(backend: str, noisy: bool = None) -> Dict[str, Any]:
    """Get configuration for a specific backend."""
    if noisy is None:
        noisy = DEFAULT_NOISY_SIMULATION
    
    if backend in EMULATOR_BACKENDS:
        return {
            "device_name": backend,
            "attempt_batching": DEFAULT_ATTEMPT_BATCHING,
            "no_opt": False,
            "simplify_initial": True,
            "noisy_simulation": noisy
        }
    else:
        return {
            "device_name": backend,
            "attempt_batching": False,  # Hardware doesn't support batching
            "no_opt": False,
            "simplify_initial": True
        }

def get_quantum_solver_config(
    shots: int = None,
    timeout: int = None,
    poll_interval: int = None,
    qpe_qubits: int = None,
    t0: float = None
) -> Dict[str, Any]:
    """Get configuration for quantum linear solver."""
    return {
        "shots": shots or DEFAULT_SHOTS,
        "timeout": timeout or DEFAULT_TIMEOUT,
        "poll_interval": poll_interval or DEFAULT_POLL_INTERVAL,
        "qpe_qubits": qpe_qubits or DEFAULT_QPE_QUBITS,
        "t0": t0 or DEFAULT_T0
    }

def get_ir_config(
    precision: float = None,
    max_iter: int = None,
    nabla: float = None,
    rho: float = None
) -> Dict[str, Any]:
    """Get configuration for iterative refinement."""
    return {
        "precision": precision or DEFAULT_PRECISION,
        "max_iter": max_iter or DEFAULT_MAX_ITERATIONS,
        "nabla": nabla or DEFAULT_NABLA,
        "rho": rho or DEFAULT_RHO
    }

# ============================================================================
# ENVIRONMENT VARIABLES
# ============================================================================

# Allow environment variables to override defaults
def get_env_config() -> Dict[str, Any]:
    """Get configuration from environment variables."""
    config = {}
    
    # Backend configuration
    if os.getenv("QNEXUS_DEFAULT_BACKEND"):
        config["DEFAULT_BACKEND"] = os.getenv("QNEXUS_DEFAULT_BACKEND")
    
    # Quantum job configuration
    if os.getenv("QNEXUS_DEFAULT_SHOTS"):
        config["DEFAULT_SHOTS"] = int(os.getenv("QNEXUS_DEFAULT_SHOTS"))
    
    if os.getenv("QNEXUS_DEFAULT_TIMEOUT"):
        config["DEFAULT_TIMEOUT"] = int(os.getenv("QNEXUS_DEFAULT_TIMEOUT"))
    
    # Problem generation configuration
    if os.getenv("QNEXUS_DEFAULT_CONDITION_NUMBER"):
        config["DEFAULT_CONDITION_NUMBER"] = float(os.getenv("QNEXUS_DEFAULT_CONDITION_NUMBER"))
    
    if os.getenv("QNEXUS_DEFAULT_SPARSITY"):
        config["DEFAULT_SPARSITY"] = float(os.getenv("QNEXUS_DEFAULT_SPARSITY"))
    
    # IR configuration
    if os.getenv("QNEXUS_DEFAULT_PRECISION"):
        config["DEFAULT_PRECISION"] = float(os.getenv("QNEXUS_DEFAULT_PRECISION"))
    
    if os.getenv("QNEXUS_DEFAULT_MAX_ITERATIONS"):
        config["DEFAULT_MAX_ITERATIONS"] = int(os.getenv("QNEXUS_DEFAULT_MAX_ITERATIONS"))
    
    return config

# Apply environment configuration
env_config = get_env_config()
for key, value in env_config.items():
    globals()[key] = value 