from abc import ABC, abstractmethod
from qlsas.backend.base import Backend
from qlsas.transpiler.base import Transpiler
from qlsas.executer.base import Executer
from qlsas.result.base import Result
from qiskit import QuantumCircuit
import numpy as np

class QuantumLinearSolver(ABC):
    """
    Abstract base class for quantum linear solvers.
    Combines a state preparation circuit, a QLSA, a readout method, and a backend.
    """

    @abstractmethod
    def build_circuit(
        self, 
        state_prep_circuit: QuantumCircuit,
        qlsa: QuantumCircuit,
        readout_circuit: QuantumCircuit,
    ) -> QuantumCircuit:
        """
        Build the circuit for the quantum linear solver.  
        Uses Qiskit QuantumCircuit objects as building blocks for now.
        """
        pass

    @abstractmethod
    def transpile(
        self,
        circuit: QuantumCircuit,
        backend: Backend,
        transpiler: Transpiler,
    ) -> QuantumCircuit:
        """
        Transpile the Quantum Linear Solver circuit to a backend.
        Delegates to the Transpiler object.
        """
        pass

    @abstractmethod
    def execute(
        self,
        circuit: QuantumCircuit,
        backend: Backend
    ) -> Result:
        """
        Execute the circuit on the backend.
        Delegates to the Executer object.
        """
        pass

    @abstractmethod
    def solve(
        self, 
        circuit: QuantumCircuit, 
        backend: Backend,
        transpiler: Transpiler,
        executer: Executer
    ) -> np.ndarray:
        """
        User facing wrapper method to solve the linear system Ax = b in one go.
        Builds the circuit, transpiles it, executes it, postprocesses, and returns the classicalresult.
        """
        pass