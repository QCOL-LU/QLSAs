"""
General solver interface for quantum linear system solvers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict
import numpy as np

Context = Dict[str, Any] # Context is a dictionary of key-value pairs that are passed to the solver.


@dataclass
class SolverResult:
    """
    Container for solver outputs.

    Attributes:
        solution: The (normalised, all positive valued) solution vector obtained from the solver.
        metadata: Backend- or solver-specific information such as circuit statistics.
        raw_result: Optional handle to the raw backend result object.
    """

    solution: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_result: Any | None = None

    @property
    def vector(self) -> np.ndarray:
        """Alias for the solution vector."""
        return self.solution


class Solver(ABC):
    """
    Abstract base class that implements a template method for quantum solvers.
    """

    def solve(self, A: np.ndarray, b: np.ndarray, **options: Any) -> SolverResult:
        """
        Solve the linear system Ax = b.

        Args:
            A: Coefficient matrix.
            b: Right-hand side vector.
            **options: Backend- or solver-specific keyword arguments.

        Returns:
            A SolverResult containing the approximated solution and metadata.
        """

        context = self._prepare_problem(A, b, **options)
        circuit = self._build_circuit(context)
        execution = self._execute_circuit(circuit, context)
        return self._postprocess(execution, context)

    
    
    def _prepare_problem(self, A: np.ndarray, b: np.ndarray, **options: Any) -> Context:
        """
        Validate and preprocess the inputs. Subclasses may extend this context.
        """

        A_arr = np.asarray(A, dtype=np.float64)
        b_arr = np.asarray(b, dtype=np.float64)
        # TODO: Hermitian check should be done here, not in the _build_circuit method.
        

        

    @abstractmethod
    def _build_circuit(self, context: Context) -> Any:
        """
        Construct the quantum circuit (or backend-specific program) for the problem.
        """

    @abstractmethod
    def _execute_circuit(self, circuit: Any, context: Context) -> Any:
        """
        Run the circuit on the target backend and return an execution artefact.
        """

    @abstractmethod
    def _postprocess(self, execution_result: Any, context: Context) -> SolverResult:
        """
        Convert the execution artefact into a SolverResult.
        """

    def solve_vector(self, A: np.ndarray, b: np.ndarray, **options: Any) -> np.ndarray:
        """
        Convenience helper returning only the solution vector.
        """

        return self.solve(A, b, **options).solution