import numpy as np
import math
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import StatePreparation

class StatePrep:
    """
    StatePrep class for constructing the state preparation circuit for the QLSA.
    """
    def __init__(self, method='default'):
        self.method = method

    def load_state(self, state: np.ndarray) -> QuantumCircuit:
        if self.method == 'default':
            return self.load_state_default(state)
        else:
            raise ValueError(f"Invalid method: {self.method}")

    def load_state_default(self, state: np.ndarray) -> QuantumCircuit:
        """Load a state into a quantum circuit using StatePreparation (unitary, no reset).
        Uses StatePreparation instead of initialize() to avoid the reset gate, which is not
        supported by restricted backends like IBM Miami (Nighthawk).
        """
        if not math.log2(len(state)).is_integer():
            raise ValueError(f"State must be a power of two: {len(state)}")
        if not np.isclose(np.linalg.norm(state), 1):
            raise ValueError(f"State must have unit norm, instead has norm: {np.linalg.norm(state)}")

        register_size = int(math.log2(len(state)))

        # Initialize the circuit
        b_register = QuantumRegister(register_size)
        circuit = QuantumCircuit(b_register)

        # Load the state using StatePreparation (unitary gate, no reset)
        sp = StatePreparation(list(state), normalize=True)
        circuit.append(sp, b_register)
        return circuit