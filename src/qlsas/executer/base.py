from abc import ABC, abstractmethod
from backend.base import Backend
from qiskit import QuantumCircuit
from qlsas.result.base import Result

class Executer(ABC):
    """
    Abstract base class for executer.
    """

    @abstractmethod
    def run(self, 
        circuit: QuantumCircuit, 
        backend: Backend,
        shots: int,
    ) -> Result:
        """
        Execute the circuit on the backend.
        """
        pass