from abc import ABC, abstractmethod
from qiskit import QuantumCircuit

class Transpiler(ABC):
    """
    Abstract base class for transpiler.
    """

    @abstractmethod
    def optimize(self, circuit: QuantumCircuit, backend_name: str) -> QuantumCircuit:
        """
        Optimize the circuit for target hardware.
        """
        raise NotImplementedError