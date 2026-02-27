from qlsas.qlsa.base import QLSA
from typing import Optional
import warnings
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import HamiltonianGate, QFT
import numpy as np
import math
from numpy.linalg import cond
from qlsas.data_loader import StatePrep
from qlsas.qlsa.hhl.hhl_helpers import classical_eig_inversion_oracle, quantum_eig_inversion_oracle, dynamic_t0, C_factor

class HHL(QLSA):
    def __init__(
        self,
        state_prep: StatePrep,
        readout: str,
        num_qpe_qubits: int,
        eig_oracle: str = "classical",
    ):
        """
        Initialize the HHL QLSA configuration.
        Args:
            state_prep: The state preparation method to use with load_state().
            readout: The readout method to use. Should be either 'measure_x' or 'swap_test'.
            num_qpe_qubits: The number of qubits to use for the QPE.
            eig_oracle: The eigenvalue inversion oracle to use. Either 'classical' (default) or
                'quantum'. The classical oracle uses classically computed eigenvalues to construct
                controlled-RY rotations; the quantum oracle uses Qiskit's ExactReciprocalGate.
        """
        super().__init__()
        self.state_prep = state_prep
        self.readout = readout
        self.num_qpe_qubits = num_qpe_qubits
        self.eig_oracle = eig_oracle

    def build_circuit(
        self, 
        A: np.ndarray, 
        b: np.ndarray, 
        swap_test_vector: Optional[np.ndarray] = None,
        t0: Optional[float] = None, 
        C: Optional[float] = None
    ) -> QuantumCircuit:
        """
        Compose the HHL circuit out of the state preparation circuit, the QLSA, and the readout circuit.
        Either calls measure_x_circuit or swap_test_circuit, depending on the readout method.
        
        Args:
            A: The matrix representing the linear system.
            b: The vector representing the right-hand side of the linear system.
            swap_test_vector: The vector to use for the swap test. Only used if readout is 'swap_test'.
            t0: The time parameter used in the controlled-Hamiltonian operations.
            C: The scaling factor used in the controlled-Hamiltonian operations.

        Returns:
            QuantumCircuit: The composed HHL circuit.
        """
        # Check if readout method is valid
        if self.readout not in ("measure_x", "swap_test"):
            raise ValueError("readout must be either 'measure_x' or 'swap_test'")
        if self.eig_oracle not in ("classical", "quantum"):
            raise ValueError("eig_oracle must be either 'classical' or 'quantum'")
            
        # Swap test validation
        if self.readout == "swap_test" and swap_test_vector is None:
            raise ValueError("swap_test requires `swap_test_vector`.")
        if self.readout == "measure_x" and swap_test_vector is not None:
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

        # Dynamically calculate t0 and C factor for the HHL circuit
        self.t0 = dynamic_t0(A) if t0 is None else t0
        self.C = C_factor(A) if C is None else C

        # Build circuit based on readout method
        if self.readout == "measure_x":
            return self.measure_x_circuit(A, b)
        elif self.readout == "swap_test":
            return self.swap_test_circuit(A, b, swap_test_vector)

    def _apply_qpe(
        self, 
        circ: QuantumCircuit, 
        A: np.ndarray, 
        qpe_register: QuantumRegister, 
        target_register: QuantumRegister, 
        inverse: bool = False
    ) -> None:
        """Applies Phase Estimation (or Inverse Phase Estimation) for the HHL algorithm."""
        if not inverse:
            circ.h(qpe_register)
            circ.barrier() 
            
            for i in range(len(qpe_register)):
                time = self.t0 * (2**i)
                U = HamiltonianGate(A, -time, label=f"H_{i}")
                G = U.control(1)
                qubits = [qpe_register[i]] + target_register[:]
                circ.append(G, qubits)
            
            circ.barrier() 
            iqft = QFT(len(qpe_register), approximation_degree=0, do_swaps=True, inverse=True, name='IQFT')
            circ.append(iqft, qpe_register)
        else:
            qft = QFT(len(qpe_register), approximation_degree=0, do_swaps=True, inverse=False, name='QFT')
            circ.append(qft, qpe_register)
            circ.barrier() 
            
            for i in reversed(range(len(qpe_register))):
                time = self.t0 * (2**i)
                U = HamiltonianGate(A, time, label=f"H_{i}")
                G = U.control(1)
                qubits = [qpe_register[i]] + target_register[:]
                circ.append(G, qubits)
            
            circ.barrier()
            circ.h(qpe_register)

    def _apply_eig_oracle(
        self,
        circ: QuantumCircuit,
        qpe_register: QuantumRegister,
        ancilla_qubit,
        A: np.ndarray,
    ) -> None:
        """Dispatch to the selected eigenvalue inversion oracle."""
        if self.eig_oracle == "classical":
            classical_eig_inversion_oracle(
                circ, qpe_register, ancilla_qubit,
                A=A, t0=self.t0, C=self.C,
            )
        elif self.eig_oracle == "quantum":
            quantum_eig_inversion_oracle(
                circ, qpe_register, ancilla_qubit,
                A=A, t0=self.t0, C=self.C,
            )

    def _build_base_hhl_circuit(self, A: np.ndarray, b: np.ndarray):
        """
        Build the core operations of the HHL circuit shared among all readout methods.
        Includes state preparation, forward QPE, eigenvalue inversion, and uncomputation.
        """
        data_register_size = int(math.log2(len(b)))
        
        # Quantum registers
        ancilla_flag_register = QuantumRegister(1, name='ancilla_flag_register') 
        qpe_register          = QuantumRegister(self.num_qpe_qubits, name='qpe_register') 
        b_to_x_register       = QuantumRegister(data_register_size, name='b_to_x_register') 
        
        # Classical registers
        ancilla_flag_result   = ClassicalRegister(1, name='ancilla_flag_result')

        # Initialize the circuit
        circ = QuantumCircuit(
            ancilla_flag_register, 
            qpe_register, 
            b_to_x_register, 
            ancilla_flag_result, 
        )

        # 1. State Preparation
        circ.compose(self.state_prep.load_state(b), b_to_x_register, inplace=True) 
        circ.barrier() 
        
        # 2. Forward QPE
        self._apply_qpe(circ, A, qpe_register, b_to_x_register, inverse=False)
        circ.barrier() 
        
        # 3. Eigenvalue-based rotation
        self._apply_eig_oracle(circ, qpe_register, ancilla_flag_register[0], A)
        circ.barrier() 

        # 4. Measure Ancilla Flag
        circ.measure(ancilla_flag_register, ancilla_flag_result) 
        circ.barrier() 
        
        # 5. Uncomputation (Inverse QPE)
        self._apply_qpe(circ, A, qpe_register, b_to_x_register, inverse=True)
        circ.barrier() 
        
        return circ, b_to_x_register

    def measure_x_circuit(self, A: np.ndarray, b: np.ndarray) -> QuantumCircuit:
        """
        Build the circuit for measuring the x register.
        """
        circ, b_to_x_register = self._build_base_hhl_circuit(A, b)
        circ.name = f"HHL {len(b)} by {len(b)}"
        
        # Add custom registers needed for x measurement
        x_result = ClassicalRegister(len(b_to_x_register), name='x_result')
        circ.add_register(x_result)
        
        # Measure
        circ.measure(b_to_x_register, x_result)
        
        return circ
        
    def swap_test_circuit(self, A: np.ndarray, b: np.ndarray, swap_test_vector: np.ndarray) -> QuantumCircuit:
        """
        Build the circuit for the swap test. Estimates the inner product of x and v.
        """
        circ, b_to_x_register = self._build_base_hhl_circuit(A, b)
        circ.name = f"HHL Swap Test {len(b)} by {len(b)}"
        
        # Add custom registers needed for the swap test
        swap_test_ancilla_register = QuantumRegister(1, name='swap_test_ancilla_register')
        v_register                 = QuantumRegister(len(b_to_x_register), name='v_register') 
        swap_test_ancilla_result   = ClassicalRegister(1, name='swap_test_result')
        
        circ.add_register(swap_test_ancilla_register, v_register, swap_test_ancilla_result)
        
        # Load v into the circuit
        circ.compose(
            self.state_prep.load_state(swap_test_vector),
            v_register,
            inplace=True
        )
        circ.barrier()

        # Swap test operations
        circ.h(swap_test_ancilla_register)
        for i in range(len(b_to_x_register)):
            circ.cswap(swap_test_ancilla_register[0], b_to_x_register[i], v_register[i])
        circ.h(swap_test_ancilla_register)
        circ.barrier()

        # Measure
        circ.measure(swap_test_ancilla_register, swap_test_ancilla_result)
        
        return circ