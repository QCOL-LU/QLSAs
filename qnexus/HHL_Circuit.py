import numpy as np
from numpy import linalg as LA
import math
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import Initialize, RYGate, HamiltonianGate, QFT

def hhl_circuit(A, b, t0=2*np.pi, n_qpe_qubits=None):
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
    qpe_qubits = n_qpe_qubits if n_qpe_qubits is not None else n_b
    q_reg = QuantumRegister(qpe_qubits, name='q')  # QPE register size is now variable
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
    # This part can be tricky. A common simplification is to use a constant rotation
    # C = 1 / cond(A)
    # angle = 2 * np.arcsin(C) # This is one approach
    # For simplicity, let's assume a fixed rotation angle as a placeholder
    # This part requires careful calibration based on eigenvalue estimates
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