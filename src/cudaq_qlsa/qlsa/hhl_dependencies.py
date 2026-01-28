import cudaq

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
