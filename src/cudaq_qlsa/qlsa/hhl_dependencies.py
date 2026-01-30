import cudaq
import numpy as np
import math

# Modified version of QFT available on CudaQ documentation: swaps added.
@cudaq.kernel
def QFT(qubits: cudaq.qview):
    '''Args:
    qubits (cudaq.qview): specifies the quantum register to which apply the QFT.'''
    qubit_count = len(qubits)
    
    for i in range(qubit_count // 2):
        swap(qubits[i], qubits[qubit_count - 1 - i])

    # Apply Hadamard gates and controlled rotation gates.
    for i in range(qubit_count):
        h(qubits[i])
        for j in range(i + 1, qubit_count):
            angle = -(2 * np.pi) / (2**(j - i + 1))
            cr1(angle, [qubits[j]], qubits[i])

# Inverse of QFT can be accessed by applying adjoint operator.
@cudaq.kernel
def invQFT(qubits: cudaq.qview):
    '''Args:
    qubits (cudaq.qview): specifies the quantum register to which apply the inverse QFT.'''
    cudaq.adjoint(QFT, qubits)



@cudaq.kernel
def A_exp_pauli(qubits: cudaq.qview, coefficients: list[float], words: list[cudaq.pauli_word], Time: float):
    for i in range(len(coefficients)):
        exp_pauli((1)*coefficients[i]*Time, qubits, words[i])

@cudaq.kernel
def NA_exp_pauli(qubits: cudaq.qview, coefficients: list[float], words: list[cudaq.pauli_word], Time: float):
    for i in range(len(coefficients)): 
        exp_pauli((-1)*coefficients[i]*Time, qubits, words[i])


@cudaq.kernel
def EIO(qReg: cudaq.qview, qAnc: cudaq.qubit, n_qpe_qubits: int, thetas: list[float]):
   
    m = n_qpe_qubits

    n_states = 2**m

    for k in range(n_states):
        theta = thetas[k]

        for j in range(m):
            bit_val = (k // (2**(m - 1 - j))) % 2
            if bit_val == 0:
                x(qReg[j])

        ry.ctrl(theta, qReg, qAnc)

        for j in range(m):
            bit_val = (k // (2**(m - 1 - j))) % 2
            if bit_val == 0:
                x(qReg[j])


def eigs_processor(A: np.ndarray, lam_floor:float = 1e-12):
   
    eigs = np.linalg.eigvalsh(A)
    
    eigs = np.real_if_close(eigs)
    eigs = np.real(eigs)
    eigs = np.sort(np.abs(eigs))
    
    eigs[eigs < lam_floor] = lam_floor
    
    return eigs


def ry_angle(n_qpe_qubits: int, t0: float, eigs: list[float], C: float, unwrap_phase: bool = False, lam_floor: float = 1e-12):
    
    thetas = []
    m = n_qpe_qubits
    
    for k in range(2**m):
        phi = k / (2**m)
       
        if unwrap_phase and phi >= 0.5:
            phi -= 1.0
       
        lam_est = abs((2*math.pi*phi)/t0)
        lam_est = max(lam_est, lam_floor)
       
        lam = min(eigs, key=lambda x: abs(x - lam_est))
       
        ratio = min(max(C/lam, 0.0), 1.0)
        theta = 2*math.asin(ratio)
       
        thetas.append(theta)

    return thetas