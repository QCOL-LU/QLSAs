from qlsas.qlsa.base import QLSA
from typing import Optional
import warnings
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import Initialize, RYGate, HamiltonianGate, QFT
import numpy as np
import math
from numpy.linalg import cond
from qlsas.data_loader import StatePrep


def eig_inversion_oracle(
    circ,
    qpe_register,
    ancilla_qubit,
    t0: float,
    eigs: np.ndarray,
    C: float,
    unwrap_phase: bool = False,
    lam_floor: float = 1e-12,
):
    m = len(qpe_register)
    eigs = np.real_if_close(eigs)
    eigs = np.real(eigs)
    eigs = np.sort(np.abs(eigs))  # SPD-friendly
    eigs[eigs < lam_floor] = lam_floor

    for k in range(2**m):
        phi = k / (2**m)
        if unwrap_phase and phi >= 0.5:
            phi -= 1.0
        lam_est = abs((2*np.pi*phi) / t0)
        lam_est = max(lam_est, lam_floor)

        # near-term solution: snap phase-bin estimate to nearest true eigenvalue
        lam = eigs[np.argmin(np.abs(eigs - lam_est))]

        ratio = C / lam
        ratio = min(max(ratio, 0.0), 1.0)
        theta = 2*np.arcsin(ratio)

        ctrl_state = format(k, f"0{m}b")
        mc_ry = RYGate(theta).control(m, ctrl_state=ctrl_state)
        circ.append(mc_ry, list(qpe_register) + [ancilla_qubit])

class HHL(QLSA):
    def __init__(
        self,
        state_prep: StatePrep,
        readout: str,
        num_qpe_qubits: int,
        t0: float,
        swap_test_vector: Optional[np.ndarray] = None
    ):
    # TODO: add all attributes as args to the build_circuit method instead of initializing the class
        """
        Initialize the HHL QLSA.
        Args:
            A: The matrix representing the linear system.
            b: The vector representing the right-hand side of the linear system.
            state_prep: The state preparation method to use with load_state().
            readout: The readout method to use. Should be either 'measure_x' or 'swap_test'.
            num_qpe_qubits: The number of qubits to use for the QPE.
            t0: The time parameter used in the controlled-Hamiltonian operations.
            swap_test_vector: The vector to use for the swap test. Only used if readout is 'swap_test'.
        """
        
        super().__init__()
        self.state_prep = state_prep
        self.readout = readout
        self.num_qpe_qubits = num_qpe_qubits
        self.t0 = t0
        self.swap_test_vector = swap_test_vector

    
    def build_circuit(self, A: np.ndarray, b: np.ndarray) -> QuantumCircuit:
        """
        Compose the HHL circuit out of the state preparation circuit, the QLSA, and the readout circuit.
        Either calls measure_x_circuit or swap_test_circuit, depending on the readout method.

        Returns:
            QuantumCircuit: The composed HHL circuit.
        """

        # Check if readout method is valid
        if self.readout not in ("measure_x", "swap_test"):
            raise ValueError("readout must be either 'measure_x' or 'swap_test'")
        if self.readout == "swap_test" and self.swap_test_vector is None:
            raise ValueError("swap_test requires `swap_test_vector`.")
        if self.readout == "measure_x" and self.swap_test_vector is not None:
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

        if self.readout == "measure_x":
            return self.measure_x_circuit(A, b)

        elif self.readout == "swap_test":
            return self.swap_test_circuit(A, b)

        else:
            raise ValueError(f"Invalid readout method: {self.readout}")


    def measure_x_circuit(self, A: np.ndarray, b: np.ndarray) -> QuantumCircuit:
        """
        Build the circuit for measuring the x register.
        """
        data_register_size = int(math.log2(len(b)))
        
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
            name=f"HHL {len(b)} by {len(b)}"
        )

        # init = initialize(list(b))
        # circ.append(init, b_to_x_register)
        
        circ.compose( # load b into the circuit
            self.state_prep.load_state(b),
            b_to_x_register, 
            inplace=True) 
        
        circ.barrier() #==============================================================
        circ.h(qpe_register)
        
        circ.barrier() #==============================================================
        for i in range(len(qpe_register)):
            time = self.t0 / (2**(len(qpe_register) - 1 - i))
            U = HamiltonianGate(A, time)
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
        # for i in range(len(qpe_register)):
        #     #angle = (2*np.pi) / (2**(i+1)) # Simplified rotation, not eigenvalue-dependent
        #     circ.cry(angle, qpe_register[i], ancilla_flag_register[0])
        
        # eigs = np.linalg.eigvals(A)
        # for i in range(1, len(qpe_register) + 1):
        #     U = RYGate((2*np.pi)/eigs[i-1]).control(1)  # or 2**(len(q_reg)+1-i) factor?
        #     circ.append(U, [i, 0])

        eigs = np.linalg.eigvalsh(A)          # Hermitian → stable
        eig_inversion_oracle(
        circ, 
        qpe_register, 
        ancilla_flag_register[0], 
        self.t0, 
        eigs, 
        C = 0.9 * np.min(np.abs(eigs)),        # safe choice
        unwrap_phase=False
        )

        
        circ.barrier() #==============================================================
        circ.measure(ancilla_flag_register, ancilla_flag_result) # TODO: add support for mid-circuit measurements to postselect on ancilla flag
        
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
            U = HamiltonianGate(A, -time)
            G = U.control(1)
            qubits = [qpe_register[i]] + b_to_x_register[:]
            circ.append(G, qubits)
        
        circ.barrier() #==============================================================
        circ.h(qpe_register)
        
        circ.barrier() #==============================================================
        circ.measure(b_to_x_register, x_result)
        return circ
        

    def swap_test_circuit(self, A: np.ndarray, b: np.ndarray) -> QuantumCircuit:
        """
        Build the circuit for the swap test. Estimates the inner product of x and v.
        """
        data_register_size = int(math.log2(len(b)))
        
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
            name=f"HHL Swap Test{len(b)} by {len(b)}"
        )

        # init = initialize(list(b))
        # circ.append(init, b_to_x_register)
        
        circ.compose( # load b into the circuit
            self.state_prep.load_state(b), 
            b_to_x_register, 
            inplace=True) 
        
        circ.barrier() #==============================================================
        circ.h(qpe_register)
        
        circ.barrier() #==============================================================
        for i in range(len(qpe_register)):
            time = self.t0 / (2**(len(qpe_register) - 1 - i))
            U = HamiltonianGate(A, time)
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
        eigs = np.linalg.eigvalsh(A)          # Hermitian → stable
        eig_inversion_oracle(
        circ, 
        qpe_register, 
        ancilla_flag_register[0], 
        self.t0, 
        eigs, 
        C = 0.9 * np.min(np.abs(eigs)),        # safe choice
        unwrap_phase=False
        )
        
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
            U = HamiltonianGate(A, -time)
            G = U.control(1)
            qubits = [qpe_register[i]] + b_to_x_register[:]
            circ.append(G, qubits)
        
        circ.barrier() #==============================================================
        circ.h(qpe_register)
        
        circ.barrier() #==============================================================
        circ.compose( # load v into the circuit
            self.state_prep.load_state(self.swap_test_vector),
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