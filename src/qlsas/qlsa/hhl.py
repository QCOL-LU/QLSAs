from qlsas.qlsa.base import QLSA
from typing import Optional
import warnings
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import Initialize, RYGate, HamiltonianGate, QFT
import numpy as np
import math
from qlsas.state_prep.prepare import StatePrep

class HHL(QLSA):
    def __init__(
        self,
        A: np.ndarray,
        b: np.ndarray,
        state_prep_circuit: StatePrep,
        readout: str,  # should be either 'measure_x' or 'swap_test'
        num_qpe_qubits: int,
        t0: float = 2*np.pi,
        swap_test_vector: Optional[np.ndarray] = None
    ):
        """
        Initialize the HHL QLSA.
        Args:
            A: The matrix representing the linear system.
            b: The vector representing the right-hand side of the linear system.
            state_prep_circuit: The state preparation circuit.
            readout: The readout method to use.
            num_qpe_qubits: The number of qubits to use for the QPE.
            t0: The time parameter used in the controlled-Hamiltonian operations.
            swap_test_vector: The vector to use for the swap test. Only used if readout is 'swap_test'.
        """
        # Check if readout method is valid
        if readout not in ("measure_x", "swap_test"):
            raise ValueError("readout must be either 'measure_x' or 'swap_test'")
        if readout == "swap_test" and swap_test_vector is None:
            raise ValueError("swap_test requires `swap_test_vector`.")
        if readout == "measure_x" and swap_test_vector is not None:
            warnings.warn("swap_test_vector provided but readout is 'measure_x'; ignoring.")

        # Check if A is a square matrix and b is a vector of matching size
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(f"A must be a square matrix, got shape {A.shape}")
        if b.ndim != 1:
            raise ValueError(f"b must be a vector, got shape {b.shape}")
        if A.shape[0] != b.shape[0]:
            raise ValueError(f"Dimension mismatch: A is {A.shape[0]}x{A.shape[1]}, but b has shape {b.shape}")
        
        # Check if A is Hermitian
        if not np.allclose(A, A.T.conjugate()):
            raise ValueError("A must be Hermitian")
        
        # Check if b has unit norm
        if not np.isclose(np.linalg.norm(b), 1):
            raise ValueError(f"b should have unit norm, instead has norm: {np.linalg.norm(b)}")
        
        super().__init__()
        self.A = A
        self.b = b
        self.state_prep_circuit = state_prep_circuit
        self.readout = readout
        self.num_qpe_qubits = num_qpe_qubits
        self.t0 = t0
        self.swap_test_vector = swap_test_vector

    
    def build_circuit(self) -> QuantumCircuit:
        """
        Compose the HHL circuit out of the state preparation circuit, the QLSA, and the readout circuit.
        Either calls measure_x_circuit or swap_test_circuit, depending on the readout method.

        Returns:
            QuantumCircuit: The composed HHL circuit.
        """
        if self.readout == "measure_x":
            return self.measure_x_circuit()

        elif self.readout == "swap_test":
            return self.swap_test_circuit()

        else:
            raise ValueError(f"Invalid readout method: {self.readout}")


    def measure_x_circuit(self) -> QuantumCircuit:
        """
        Build the circuit for measuring the x register.
        """
        data_register_size = int(math.log2(len(self.b)))
        
        # Quantum registers
        ancilla_flag_register = QuantumRegister(1, name='ancilla_flag_register') 
        qpe_register          = QuantumRegister(self.num_qpe_qubits, name='qpe_register') # increase for precision
        b_to_x_register       = QuantumRegister(data_register_size, name='b_to_x_register') # register to load b into
        # Classical registers
        ancilla_flag_result   = ClassicalRegister(1, name='ancilla_flag_result')
        x_result              = ClassicalRegister(data_register_size, name='x_result')
        # num_qubits = len(qpe_register) + len(b_to_x_register) + 1

        # Initialize the circuit
        circ = QuantumCircuit(
            ancilla_flag_register, 
            qpe_register, 
            b_to_x_register, 
            ancilla_flag_result, 
            x_result,
            name=f"HHL {len(self.b)} by {len(self.b)}"
        )

        # init = initialize(list(self.b))
        # circ.append(init, b_to_x_register)
        
        circ.compose( # load b into the circuit
            self.state_prep_circuit.load_state(self.b), 
            b_to_x_register, 
            inplace=True) 
        
        circ.barrier() #==============================================================
        circ.h(qpe_register)
        
        circ.barrier() #==============================================================
        for i in range(len(qpe_register)):
            time = self.t0 / (2**(len(qpe_register) - 1 - i))
            U = HamiltonianGate(self.A, time)
            G = U.control(1)
            qubits = [qpe_register[i]] + b_to_x_register[:]
            circ.append(G, qubits)
        
        circ.barrier() #==============================================================
        iqft = QFT(
            len(qpe_register), 
            approximation_degree=0, 
            do_swaps=True, 
            inverse=True, 
            name='IQFT'
            )
        circ.append(iqft, qpe_register)
        
        circ.barrier() #==============================================================
        # Eigenvalue-based rotation
        # A common simplification is to use a constant rotation
        # C = 1 / cond(A)
        # angle = 2 * np.arcsin(C) # This is one approach
        # For simplicity, let's assume a fixed rotation angle as a placeholder
        # This part requires careful calibration based on eigenvalue estimates
        # WARNING: This is a simplified implementation. For production use, 
        # the rotation angles should be calculated based on the actual eigenvalues.
        for i in range(len(qpe_register)):
            angle = (2*np.pi) / (2**(i+1)) # Simplified rotation, not eigenvalue-dependent
            circ.cry(angle, qpe_register[i], ancilla_flag_register[0])
        
        circ.barrier() #==============================================================
        circ.measure(ancilla_flag_register, ancilla_flag_result)
        
        circ.barrier() #==============================================================
        # Uncomputation
        qft = QFT(
            len(qpe_register), 
            approximation_degree=0, 
            do_swaps=True, 
            inverse=False, 
            name='QFT'
            )
        circ.append(qft, qpe_register)
        
        circ.barrier() #==============================================================
        for i in range(len(qpe_register))[::-1]:
            time = self.t0 / (2**(len(qpe_register) - 1 - i))
            U = HamiltonianGate(self.A, -time)
            G = U.control(1)
            qubits = [qpe_register[i]] + b_to_x_register[:]
            circ.append(G, qubits)
        
        circ.barrier() #==============================================================
        circ.h(qpe_register)
        
        circ.barrier() #==============================================================
        circ.measure(b_to_x_register, x_result)
        return circ
        

    def swap_test_circuit(self) -> QuantumCircuit:
        """
        Build the circuit for the swap test. Estimates the inner product of x and v.
        """
        data_register_size = int(math.log2(len(self.b)))
        
        # Quantum registers
        ancilla_flag_register      = QuantumRegister(1, name='ancilla_flag_register') 
        qpe_register               = QuantumRegister(self.num_qpe_qubits, name='qpe_register') # increase for precision
        b_to_x_register            = QuantumRegister(data_register_size, name='b_to_x_register') # register to load b into
        # quantum registers for the swap test
        swap_test_ancilla_register = QuantumRegister(1, name='swap_test_ancilla_register')
        v_register                 = QuantumRegister(data_register_size, name='v_register') # register to load v into
        # Classical registers
        ancilla_flag_result        = ClassicalRegister(1, name='ancilla_flag_result')
        swap_test_ancilla_result   = ClassicalRegister(1, name='swap_test_result')
        # num_qubits = len(qpe_register) + len(b_to_x_register) + 1

        # Initialize the circuit
        circ = QuantumCircuit(
            ancilla_flag_register, 
            qpe_register, 
            b_to_x_register, 
            swap_test_ancilla_register, 
            v_register, 
            ancilla_flag_result, 
            swap_test_ancilla_result,
            name=f"HHL Swap Test{len(self.b)} by {len(self.b)}"
        )

        # init = initialize(list(self.b))
        # circ.append(init, b_to_x_register)
        
        circ.compose( # load b into the circuit
            self.state_prep_circuit.load_state(self.b), 
            b_to_x_register, 
            inplace=True) 
        
        circ.barrier() #==============================================================
        circ.h(qpe_register)
        
        circ.barrier() #==============================================================
        for i in range(len(qpe_register)):
            time = self.t0 / (2**(len(qpe_register) - 1 - i))
            U = HamiltonianGate(self.A, time)
            G = U.control(1)
            qubits = [qpe_register[i]] + b_to_x_register[:]
            circ.append(G, qubits)
        
        circ.barrier() #==============================================================
        iqft = QFT(
            len(qpe_register), 
            approximation_degree=0, 
            do_swaps=True, 
            inverse=True, 
            name='IQFT'
            )
        circ.append(iqft, qpe_register)
        
        circ.barrier() #==============================================================
        # Eigenvalue-based rotation
        # A common simplification is to use a constant rotation
        # C = 1 / cond(A)
        # angle = 2 * np.arcsin(C) # This is one approach
        # For simplicity, let's assume a fixed rotation angle as a placeholder
        # This part requires careful calibration based on eigenvalue estimates
        # WARNING: This is a simplified implementation. For production use, 
        # the rotation angles should be calculated based on the actual eigenvalues.
        for i in range(len(qpe_register)):
            angle = (2*np.pi) / (2**(i+1)) # Simplified rotation, not eigenvalue-dependent
            circ.cry(angle, qpe_register[i], ancilla_flag_register[0])
        
        circ.barrier() #==============================================================
        circ.measure(ancilla_flag_register, ancilla_flag_result)
        
        circ.barrier() #==============================================================
        # Uncomputation
        qft = QFT(
            len(qpe_register), 
            approximation_degree=0, 
            do_swaps=True, 
            inverse=False, 
            name='QFT'
            )
        circ.append(qft, qpe_register)
        
        circ.barrier() #==============================================================
        for i in range(len(qpe_register))[::-1]:
            time = self.t0 / (2**(len(qpe_register) - 1 - i))
            U = HamiltonianGate(self.A, -time)
            G = U.control(1)
            qubits = [qpe_register[i]] + b_to_x_register[:]
            circ.append(G, qubits)
        
        circ.barrier() #==============================================================
        circ.h(qpe_register)
        
        circ.barrier() #==============================================================
        circ.compose( # load v into the circuit
            self.state_prep_circuit.load_state(self.swap_test_vector),
            v_register,
            inplace=True
        )

        circ.barrier() #==============================================================
        circ.h(swap_test_ancilla_register)
        for i in range(len(b_to_x_register)):
            circ.cswap(swap_test_ancilla_register[0], b_to_x_register[i], v_register[i])
        circ.h(swap_test_ancilla_register)
        circ.barrier() #==============================================================

        circ.measure(swap_test_ancilla_register, swap_test_ancilla_result)
        
        return circ
        
