from abc import ABC, abstractmethod
from qiskit import QuantumCircuit

class QLSA(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def build_circuit(self, state_prep_circuit: QuantumCircuit, readout_circuit: QuantumCircuit, **kwargs) -> QuantumCircuit:
        pass