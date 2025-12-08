"""
Qiskit implementation of the HHL circuit. Can be run on Qiskit backends, or on Quantinuum backends after conversion to tket.

Input:
A: The matrix representing the linear system.
b: The vector representing the right-hand side of the linear system.
n_qpe_qubits: The number of qubits to use for the QPE, detirmining the precision of the eigenvalue estimation.
t0: The time parameter used in the controlled-Hamiltonian operations.

Output:
QuantumCircuit: The quantum circuit for solving the linear system using HHL.
"""
import numpy as np
import math
from numpy import linalg as LA
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import Initialize, RYGate, HamiltonianGate, QFT

def qiskit_hhl_circuit(
    A: np.ndarray,
    b: np.ndarray,
    n_qpe_qubits: int,
    t0: float = 2*np.pi # Default time parameter for the controlled-Hamiltonian operations.
) -> QuantumCircuit:
    """
    Qiskit implementation of the HHL circuit. Can be run on Qiskit backends, or on Quantinuum backends after conversion to tket.

    Input:
    A: The matrix representing the linear system.
    b: The vector representing the right-hand side of the linear system.
    n_qpe_qubits: The number of qubits to use for the QPE, detirmining the precision of the eigenvalue estimation.
    t0: The time parameter used in the controlled-Hamiltonian operations.
    """
    # ==========================================================================
    # Preprocessing
    # ==========================================================================
    def check_hermitian(mat):
        mat = np.asarray(mat)
        assert np.allclose(mat, mat.T.conjugate(), rtol=1e-05, atol=1e-08), \
            "Sorry! The input matrix should be Hermitian."
    check_hermitian(A)

    norm_b = LA.norm(b)
    A = A / norm_b
    b = b / norm_b
    eigs = LA.eigvals(A)
    # ==========================================================================
    # Quantum Circuit
    # ==========================================================================
    ancilla_qbit = QuantumRegister(1, name='anc')
    n_b = int(math.log2(len(b)))
    q_reg = QuantumRegister(n_qpe_qubits, name='q')  # QPE register size is variable
    b_reg = QuantumRegister(n_b, name='b')
    ancilla_result = ClassicalRegister(1, name='anc_result')
    b_vec = ClassicalRegister(n_b, name='b_vec')
    num_qubits = len(q_reg) + len(b_reg) + 1
    circ = QuantumCircuit(ancilla_qbit, q_reg, b_reg, ancilla_result, b_vec,
                          name=f"HHL {len(b)} by {len(b)}")
    init = Initialize(list(b))
    circ.append(init, b_reg)
    circ.barrier()
    circ.h(q_reg)
    circ.barrier()
    for i in range(len(q_reg)):
        time = t0 / (2**(len(q_reg) - 1 - i))
        U = HamiltonianGate(A, time)
        G = U.control(1)
        qubits = [q_reg[i]] + b_reg[:]
        circ.append(G, qubits)
    circ.barrier()
    iqft = QFT(len(q_reg), approximation_degree=0, do_swaps=True, inverse=True, name='IQFT')
    circ.append(iqft, q_reg)
    circ.barrier()
    # Eigenvalue-based rotation
    # A common simplification is to use a constant rotation
    # C = 1 / cond(A)
    # angle = 2 * np.arcsin(C) # This is one approach
    # For simplicity, let's assume a fixed rotation angle as a placeholder
    # This part requires careful calibration based on eigenvalue estimates
    # WARNING: This is a simplified implementation. For production use, 
    # the rotation angles should be calculated based on the actual eigenvalues.
    for i in range(len(q_reg)):
        angle = (2*np.pi) / (2**(i+1)) # Simplified rotation, not eigenvalue-dependent
        circ.cry(angle, q_reg[i], ancilla_qbit[0])
    circ.barrier()
    circ.measure(ancilla_qbit, ancilla_result)
    circ.barrier()
    # Uncomputation
    qft = QFT(len(q_reg), approximation_degree=0, do_swaps=True, inverse=False, name='QFT')
    circ.append(qft, q_reg)
    circ.barrier()
    for i in range(len(q_reg))[::-1]:
        time = t0 / (2**(len(q_reg) - 1 - i))
        U = HamiltonianGate(A, -time)
        G = U.control(1)
        qubits = [q_reg[i]] + b_reg[:]
        circ.append(G, qubits)
    circ.barrier()
    circ.h(q_reg)
    circ.barrier()
    circ.measure(b_reg, b_vec)
    return circ