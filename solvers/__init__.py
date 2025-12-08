"""
Solver interfaces and backend-specific implementations.
"""

from .solver import Solver, SolverResult
from .hhl_solver import HHLBaseSolver
from .qiskit_hhl_solver import QiskitHHLSolver
from .quantinuum_hhl_solver import QuantinuumHHLSolver
from .factory import create_solver

__all__ = [
    "Solver",
    "SolverResult",
    "HHLBaseSolver",
    "QiskitHHLSolver",
    "QuantinuumHHLSolver",
    "create_solver",
]


