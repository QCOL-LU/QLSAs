from abc import ABC, abstractmethod
from qiskit import QuantumCircuit
import numpy as np

class QLSA(ABC):

    @abstractmethod
    def build_circuit(self, A: np.ndarray, b: np.ndarray) -> QuantumCircuit:
        raise NotImplementedError